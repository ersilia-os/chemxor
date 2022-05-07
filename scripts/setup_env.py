# flake8: noqa
"""Setup script."""

import os
from pathlib import Path
import platform
from subprocess import run

import urllib3

project_root = Path(__file__).absolute().parents[1]
tools_path = project_root.joinpath("scratch", "tools")
os.makedirs(tools_path, exist_ok=True)
current_system = platform.system()
current_machine = platform.machine()

client = urllib3.PoolManager()


def write_env() -> None:
    """Write .env in project root."""
    print("\n\n+++++++++++ Env File +++++++++++++\n")
    data = f"""
export CONDA_SHELL_PATH={(tools_path / 'miniconda' / 'etc' / 'profile.d' / 'conda.sh')}
export CONDA_ENV_PATH={(tools_path / 'miniconda' / 'envs' / 'imcloud')}
export PATH={(tools_path / 'miniconda' / 'envs' / 'imcloud' / 'bin')}:{(tools_path / 'miniconda' / 'condabin')}:${{PATH:+:${{PATH}}}}
export _CE_M=
export _CE_CONDA=
export CONDA_EXE={(tools_path / 'miniconda' / 'bin' / 'conda')}
export CONDA_PYTHON_EXE={(tools_path / 'miniconda' / 'bin' / 'conda' / "python")}
export CONDA_SHLVL=2
export CONDA_PREFIX={(tools_path / 'miniconda' / 'envs' / 'imcloud')}
export CONDA_DEFAULT_ENV=imcloud
export CONDA_PROMPT_MODIFIER=(imcloud)
export CONDA_PREFIX_1={(tools_path / 'miniconda')}
export PS1=\"(imcloud) ${{PS1:+${{PS1}}}}\"
    """
    with open((project_root / ".setup.env"), "w") as f:
        f.write(data)


def setup_miniconda() -> None:
    """Setup miniconda."""
    print("+++++++++++ Miniconda +++++++++++++")
    if os.path.isfile((tools_path / "miniconda" / "bin" / "conda")) is False:
        print("Isolated conda not found!\nInstalling Conda\n")
        if current_system == "Darwin":
            local_system = "MacOSX"
        else:
            local_system = current_system

        # Decide platform architecture
        local_machine = current_machine

        # Build url
        url = f"https://repo.anaconda.com/miniconda/Miniconda3-latest-{local_system}-{local_machine}.sh"

        resp = client.request("GET", url, preload_content=False)

        # Download file
        with open((tools_path.parent / "miniconda.sh"), "wb") as f:
            while True:
                data = resp.read()
                if not data:
                    break
                f.write(data)
        resp.release_conn()
        run(["chmod", "+x", "miniconda.sh"], cwd=tools_path.parent)
        run(
            ["./miniconda.sh", "-b", "-p", f"{(tools_path / 'miniconda')}"],
            cwd=tools_path.parent,
        )
        run(["chmod", "+x", "conda"], cwd=(tools_path / "miniconda" / "bin"))

    # Print the info about Conda
    out = run(
        ["./conda", "info"], capture_output=True, cwd=(tools_path / "miniconda" / "bin")
    )
    print(
        f"Using Conda at: {(tools_path / 'miniconda' / 'bin')} \n{out.stdout.decode('utf-8')}"
    )


def main() -> None:
    """Setup runner."""
    setup_miniconda()
    write_env()


if __name__ == "__main__":
    main()
