"""This module contains an example test.

Tests should be placed in ``src/tests``, in modules that mirror your
project's structure, and in files named test_*.py. They are simply functions
named ``test_*`` which test a unit of logic.

To run the tests, run ``kedro test`` from the project root directory.
"""

from pathlib import Path
from typing import Any

from kedro.framework.context import KedroContext
import pytest


@pytest.fixture
def project_context() -> None:
    return KedroContext(package_name="chemxor", project_path=Path.cwd())


# The tests below are here for the demonstration purpose
# and should be replaced with the ones testing the project
# functionality
class TestProjectContext:
    def test_package_name(self: "TestProjectContext", project_context: Any) -> None:
        assert project_context.package_name == "chemxor"
