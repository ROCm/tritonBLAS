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


def _find_triton_root():
    """Discover the triton source tree root from the installed package."""
    try:
        import triton
        pkg_dir = triton.__path__[0]  # e.g. /workspace/triton/python/triton
        # Editable install: walk up to the repo root
        candidate = os.path.normpath(os.path.join(pkg_dir, "..", ".."))
        if os.path.isfile(os.path.join(candidate, "third_party", "amd", "backend", "include", "hip", "hip_runtime.h")):
            return candidate
    except ImportError:
        pass
    return None


def _hip_cmake_env():
    """Build environment for origami's cmake when no ROCm SDK is installed.

    If /opt/rocm already has hip headers, returns an empty dict (no shim
    needed).  Otherwise, points CMAKE_PREFIX_PATH at our bundled
    hip-config.cmake and passes TRITON_ROOT so the shim can discover
    headers from the triton source tree.
    """
    if os.path.isfile("/opt/rocm/include/hip/hip_runtime.h"):
        return {}

    env = os.environ.copy()

    # Point cmake at our fallback hip-config.cmake
    cmake_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cmake")
    existing = env.get("CMAKE_PREFIX_PATH", "")
    env["CMAKE_PREFIX_PATH"] = cmake_dir + (":" + existing if existing else "")

    # Help the shim find HIP headers in the triton source tree
    triton_root = _find_triton_root()
    if triton_root:
        env["TRITON_ROOT"] = triton_root

    return env


class CustomBuildExt(build_ext):
    def run(self):
        # Check if origami is already installed (e.g. by clone_and_build.sh)
        try:
            import origami
            print("origami already installed, skipping clone/build...")
        except ImportError:
            # Remove existing _origami directory if it exists
            if os.path.exists("_origami"):
                print("Removing existing _origami directory...")
                shutil.rmtree("_origami")

            # Clone origami from rocm-libraries
            print("Cloning rocm-libraries (origami)...")
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    "--branch",
                    "users/pghysels/origami_gfx1250",
                    "git@github.com:AMD-ROCm-Internal/rocm-libraries.git",
                    "_origami",
                ]
            )

            # Use custom chdir context manager to run sparse-checkout
            with chdir("_origami"):
                subprocess.check_call(["git", "sparse-checkout", "set", "shared/origami"])

            # Build the nested origami package (uses pyproject.toml / scikit-build-core)
            origami_setup_path = os.path.join("_origami", "shared", "origami", "python")
            print(f"Building origami in {origami_setup_path}...")
            env = _hip_cmake_env()
            cmd = ["pip", "install", "."]
            subprocess.check_call(cmd, cwd=origami_setup_path, env=env or None)

        print("Running build_ext for main package...")
        super().run()


setup(
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=[Extension("_trigger_ext", sources=[])],
)
