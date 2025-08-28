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
    def run(self):
        # Remove existing _origami directory if it exists
        if os.path.exists("_origami"):
            print("Removing existing _origami directory...")
            shutil.rmtree("_origami")

        # Clone hipBLASLt repo
        print("Cloning hipBLASLt...")
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--branch",
                "users/ibrahimw1/origami-standalone",
                "https://github.com/ryanswann-amd/rocm-libraries.git",
                "_origami",
            ]
        )

        # Use custom chdir context manager to run sparse-checkout
        with chdir("_origami"):
            subprocess.check_call(["git", "sparse-checkout", "set", "shared/origami"])

        # Build the nested origami setup.py
        origami_setup_path = os.path.join("_origami", "shared", "origami", "python")
        print(f"Building origami setup.py in {origami_setup_path}...")
        subprocess.check_call(
            [sys.executable, "setup.py", "install"], cwd=origami_setup_path
        )

        print("Running build_ext for main package...")
        super().run()


setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=[Extension("_trigger_ext", sources=[])],
)
