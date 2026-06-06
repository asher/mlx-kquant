# Standalone GGUF K-quant ops for MLX, packaged as a C++/Metal extension.

from setuptools import setup

from mlx import extension

if __name__ == "__main__":
    setup(
        name="mlx_kquant",
        version="0.0.1",
        description="GGUF K-quant dequantize / quantized-matmul / gather-qmm / quantize "
        "ops for MLX, via custom Metal kernels (no MLX-core fork required).",
        ext_modules=[extension.CMakeExtension("mlx_kquant._ext")],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["mlx_kquant"],
        package_data={"mlx_kquant": ["*.so", "*.dylib", "*.metallib"]},
        install_requires=["mlx==0.31.2"],
        # This package is the low-level op/kernel layer (the kq.* namespace plus
        # the C++ GGUF wire-byte reader). The full GGUF -> MLX loader/runtime
        # lives in the separate `gguf-mlx` package, which depends on this one.
        zip_safe=False,
        python_requires=">=3.9",
    )
