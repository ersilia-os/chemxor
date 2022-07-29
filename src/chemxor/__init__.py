"""ChemXor."""

from importlib.metadata import PackageNotFoundError, version  # type: ignore
from warnings import filterwarnings

try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

from .part_net_client import PartitionNetClient  # noqa: F401
from .part_net_service import PartitionNetService  # noqa: F401

filterwarnings(action="ignore", category=DeprecationWarning, module="tensorboard")
filterwarnings(action="ignore", category=DeprecationWarning, module="torchvision")
