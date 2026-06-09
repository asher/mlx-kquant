# Standalone GGUF K-quant ops for MLX, packaged as a C++/Metal extension.
#
# All project metadata lives in pyproject.toml ([project]). This file carries
# only the imperative bits that PEP 621 can't express: the CMake-driven Metal
# extension and its build command.

from mlx import extension
from setuptools import setup

if __name__ == "__main__":
    setup(
        ext_modules=[extension.CMakeExtension("mlx_kquant._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
    )
