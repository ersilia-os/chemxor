"""ChemXor."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore

from warnings import filterwarnings

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

filterwarnings(action="ignore", category=DeprecationWarning, module="tensorboard")
filterwarnings(action="ignore", category=DeprecationWarning, module="torchvision")
