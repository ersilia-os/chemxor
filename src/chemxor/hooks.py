"""Project hooks."""
from typing import Any, Dict, Iterable, Optional

from kedro.config import ConfigLoader
from kedro.framework.hooks import hook_impl
from kedro.io import DataCatalog
from kedro.versioning import Journal


class ProjectHooks:
    """Project hooks."""

    @hook_impl
    def register_config_loader(
        self: "ProjectHooks",
        conf_paths: Iterable[str],
        env: str,
        extra_params: Dict[str, Any],
    ) -> ConfigLoader:
        """Register config loader.

        Args:
            conf_paths (Iterable[str]): [description]
            env (str): [description]
            extra_params (Dict[str, Any]): [description]

        Returns:
            ConfigLoader: [description]
        """
        return ConfigLoader(conf_paths)

    @hook_impl
    def register_catalog(
        self: "ProjectHooks",
        catalog: Optional[Dict[str, Dict[str, Any]]],
        credentials: Dict[str, Dict[str, Any]],
        load_versions: Dict[str, str],
        save_version: str,
        journal: Journal,
    ) -> DataCatalog:
        """Register data catalog.

        Args:
            catalog (Optional[Dict[str, Dict[str, Any]]]): [description]
            credentials (Dict[str, Dict[str, Any]]): [description]
            load_versions (Dict[str, str]): [description]
            save_version (str): [description]
            journal (Journal): [description]

        Returns:
            DataCatalog: [description]
        """
        return DataCatalog.from_config(
            catalog, credentials, load_versions, save_version, journal
        )
