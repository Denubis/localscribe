# pattern: Imperative Shell
"""Process-level runtime concerns: cache selection and CUDA memory hygiene.

Guardrail #3: choose the model cache deterministically before importing any
Hugging Face consumer. A LocalScribe-specific override wins, then the standard
Hugging Face setting, then Hugging Face's XDG-compatible default.
"""

from __future__ import annotations

import logging
import os
import warnings
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from .errors import MissingExtraError

logger = logging.getLogger(__name__)

def configure_cache() -> None:
    """Select ``HF_HOME`` before any Hugging Face import.

    ``LOCALSCRIBE_HF_HOME`` is the application override. Without it, preserve a
    standard ``HF_HOME`` chosen by the user. If neither is set, follow Hugging
    Face's own default: ``$XDG_CACHE_HOME/huggingface`` or
    ``~/.cache/huggingface``.
    """
    override = os.environ.get("LOCALSCRIBE_HF_HOME")
    standard = os.environ.get("HF_HOME")
    if override:
        hf_home = Path(override).expanduser()
    elif standard:
        hf_home = Path(standard).expanduser()
    else:
        cache_root = os.environ.get("XDG_CACHE_HOME")
        hf_home = (
            Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
        ) / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    logger.debug("HF_HOME set to %s", hf_home)


#: Set before the heavy imports, because these libraries read the environment at
#: import time. Every entry silences startup chatter only; none of it changes what
#: the models do. ``setdefault`` throughout, so an explicit value from the caller's
#: shell always wins.
_QUIET_ENV = {
    # NeMo initialises OpenTelemetry, which announces "No exporters were provided".
    "OTEL_SDK_DISABLED": "true",
    # nemo-toolkit depends on wandb, which introduces itself on import. Nothing
    # here ever logs a run.
    "WANDB_SILENT": "true",
    "WANDB_MODE": "disabled",
    # Otherwise every fork after a tokenizer is used prints a parallelism warning.
    "TOKENIZERS_PARALLELISM": "false",
    # In case a transitive import drags TensorFlow in.
    "TF_CPP_MIN_LOG_LEVEL": "3",
}

#: Loggers that chatter at import and during model loading.
_NOISY_LOGGERS = (
    "nemo_logger",
    "nemo",
    # NVIDIA's telemetry, pulled in by nemo. Source of both "No exporters were
    # provided" and the OneLogger error_handling_strategy line; each is a
    # `_logger.warning`, so setting the parent level filters the children.
    "nv_one_logger",
    "lightning",
    "lightning.pytorch",
    "pytorch_lightning",
    "pyannote",
    "speechbrain",
    "matplotlib",
)


class _BelowError(logging.Filter):
    """Drop anything under ERROR, whatever the logger's level currently is.

    A level is the obvious lever and it loses a race: NeMo reinstates its own
    verbosity when it loads a model, so a ``setLevel`` made beforehand is undone
    by the time the dataloader warns. ``logging`` never resets a logger's filters,
    so attaching one holds for the life of the process.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def quiet_third_party() -> None:
    """Silence third-party startup noise so progress stays readable.

    Call before the heavy imports. This suppresses *reporting*, never a failure:
    everything raised as an error still surfaces, and localscribe's own progress
    lines are untouched. The pydub ``SyntaxWarning`` filter matters only on the
    first run after an install, when Python compiles the bytecode.
    """
    for key, value in _QUIET_ENV.items():
        os.environ.setdefault(key, value)

    warnings.filterwarnings("ignore", category=SyntaxWarning)
    warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")

    for name in _NOISY_LOGGERS:
        noisy = logging.getLogger(name)
        noisy.setLevel(logging.ERROR)
        if not any(isinstance(existing, _BelowError) for existing in noisy.filters):
            noisy.addFilter(_BelowError())


def quiet_nemo_logging() -> None:
    """Turn NeMo's own logger down, after it has been imported.

    NeMo installs its logger on import and overrides the level set beforehand, so
    this has to run afterwards. Best-effort: a failure here is cosmetic.
    """
    try:
        nemo_logging = import_module("nemo.utils").logging

        nemo_logging.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001 - cosmetic only, never worth failing a run
        logger.debug("could not lower the NeMo log level", exc_info=True)


def require_gpu_extra() -> None:
    """Fail with an actionable message when the model stack is not installed.

    Uses ``find_spec`` rather than an import: this runs before every transcribing
    command, and importing torch to discover that torch exists costs seconds.
    """
    if find_spec("torch") is None:
        raise MissingExtraError(
            "this command needs the model stack, which this install does not have.\n"
            "  from a LocalScribe checkout, run: ./install.sh --gpu\n"
            "  a machine that only records does not need it: scribe-record and "
            "scribe-send work without."
        )


def free_cuda() -> None:
    """Release cached CUDA memory between model stages to stay under 24 GB.

    Safe to call when torch is absent or CUDA is unavailable.
    """
    import gc

    gc.collect()
    try:
        torch = import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
