"""Project settings."""
from pathlib import Path

from kedro_viz.integrations.kedro.sqlite_store import SQLiteStore

from chemxor.hooks import ProjectHooks

# Instantiate and list your project hooks here
HOOKS = (ProjectHooks(),)

# List the installed plugins for which to disable auto-registry
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Define where to store data from a KedroSession. Defaults to BaseSessionStore.
# from kedro.framework.session.store import ShelveStore
# SESSION_STORE_CLASS = ShelveStore

SESSION_STORE_CLASS = SQLiteStore
SESSION_STORE_ARGS = {"path": str(Path(__file__).parents[2] / "data")}

# Define keyword arguments to be passed to `SESSION_STORE_CLASS` constructor
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Define custom context class. Defaults to `KedroContext`
# CONTEXT_CLASS = KedroContext

# Define the configuration folder. Defaults to `conf`
CONF_ROOT = "src/chemxor/config"
