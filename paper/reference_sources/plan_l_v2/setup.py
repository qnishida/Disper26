from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import glob
import os

_here = os.path.dirname(os.path.abspath(__file__))

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "disper26py",
        ["src/bindings.cpp"],
        # Include paths:
        # 1. Eigen3 (Standard location)
        # 2. autodiff (fetched via CMake, in build/_deps/)
        # 3. Pybind11 (Handled by Pybind11Extension)
        include_dirs=[
            "/usr/include/eigen3",
            os.path.join(os.path.dirname(__file__), "third_party/autodiff"),
        ],
        # bindings.cpp #includes the other src/ files, so they must be listed
        # as depends or editing them does not trigger a rebuild.
        depends=sorted(
            glob.glob(os.path.join(_here, "src", "*.cpp"))
            + glob.glob(os.path.join(_here, "src", "*.hpp"))
        ),
        # -fcx-limited-range: emit complex multiply/divide inline instead of
        #   calling libgcc's __muldc3/__divdc3. The layer propagators are
        #   saturated with complex arithmetic and the NaN-recovery fixups those
        #   helpers perform never fire on finite layer parameters.
        # -fno-math-errno: let sqrt() and friends compile to a single
        #   instruction instead of a call that also has to set errno.
        # Neither relaxes real-valued IEEE semantics the way -ffast-math would.
        extra_compile_args=['-O3', '-march=native', '-DNDEBUG',
                            '-fcx-limited-range', '-fno-math-errno'],
        cxx_std=17, # C++17 required for autodiff
    ),
]

setup(
    name="disper26py",
    version="0.1.104",
    author="Kiwamu Nishida",
    description="Python bindings for Disper26 VTI Dispersion Engine via pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.6",
)
