# Standalone GGUF K-quant ops for MLX, packaged as a C++/Metal extension.
#
# All project metadata lives in pyproject.toml ([project]). This file carries
# only the imperative bits that PEP 621 can't express: the CMake-driven Metal
# extension and its build command.

import os
import sys

from mlx import extension
from setuptools import setup

# Pin CMake's find_package(Python) to the backend interpreter, which always has
# the build deps; under build isolation its own search can pick one without them.
os.environ["CMAKE_ARGS"] = (
    f"{os.environ.get('CMAKE_ARGS', '')} -DPython_EXECUTABLE={sys.executable}".strip()
)

# KQ_PROBE_KERNELS is a cached CMake option, so every build names its value:
# without it a build after a probe build keeps the probe kernels. The
# environment variable KQ_PROBE_KERNELS=1 asks for a probe build; a
# -DKQ_PROBE_KERNELS in CMAKE_ARGS is left alone.
if "-DKQ_PROBE_KERNELS" not in os.environ["CMAKE_ARGS"]:
    probe = os.environ.get("KQ_PROBE_KERNELS", "").lower() in ("1", "on", "true", "yes")
    os.environ["CMAKE_ARGS"] += f" -DKQ_PROBE_KERNELS={'ON' if probe else 'OFF'}"

# Force the Metal deployment target to 26.2 on macOS.
# https://github.com/ml-explore/mlx/issues/3586
if sys.platform == "darwin":
    os.environ["MACOSX_DEPLOYMENT_TARGET"] = "26.2"

if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("mlx_kquant._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
    )
