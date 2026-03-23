from __future__ import annotations
from shutil import copytree
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_PROFILES_DIR = PROJECT_ROOT / "profiles"
TARGET_PACKAGE = "reachy_talk_data"
TARGET_SUBDIR = "profiles"


class BuildPyWithProfiles(build_py):
    """Copy built-in profiles into the wheel data package at build time."""

    def run(self) -> None:
        """Build Python modules, then copy root-level profiles into reachy_talk_data."""
        super().run()

        target_root = Path(self.build_lib) / TARGET_PACKAGE / TARGET_SUBDIR
        copytree(SOURCE_PROFILES_DIR, target_root, dirs_exist_ok=True)


setup(
    cmdclass={"build_py": BuildPyWithProfiles},
)
