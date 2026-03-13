import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from contextlib import contextmanager


@contextmanager
def chdir(path):
    prev_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


class CustomBuildExt(build_ext):
    def _origami_available(self):
        """Check if origami is already installed with grouped GEMM support."""
        try:
            import importlib
            origami = importlib.import_module("origami")
            return hasattr(origami, "select_config_grouped")
        except ImportError:
            return False

    def run(self):
        if self._origami_available():
            print("origami already installed with grouped GEMM support, skipping build.")
        else:
            # Remove existing _origami directory if it exists
            if os.path.exists("_origami"):
                print("Removing existing _origami directory...")
                shutil.rmtree("_origami")

            # Clone rocm-libraries repo (grouped origami branch)
            print("Cloning rocm-libraries (grouped origami)...")
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--branch",
                    "ryaswann/grouped_origami",
                    "https://github.com/ROCm/rocm-libraries.git",
                    "_origami",
                ]
            )

            # Use custom chdir context manager to run sparse-checkout
            with chdir("_origami"):
                subprocess.check_call(["git", "sparse-checkout", "set", "shared/origami"])

            # Build the origami Python extension (uses pyproject.toml + scikit-build-core)
            origami_setup_path = os.path.join("_origami", "shared", "origami", "python")
            print(f"Building origami in {origami_setup_path}...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "."], cwd=origami_setup_path
            )

        print("Running build_ext for main package...")
        super().run()


setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=[Extension("_trigger_ext", sources=[])],
)
