"""Build Python bindings through the same CMake configuration as the C++ core."""
import os
from pathlib import Path
import shlex
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        import pybind11

        source = Path(__file__).resolve().parent
        output = Path(self.get_ext_fullpath(ext.name)).resolve().parent
        build_dir = Path(self.build_temp).resolve() / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)
        config = "Debug" if self.debug else "Release"
        args = [
            "cmake", "-S", str(source), "-B", str(build_dir),
            f"-DCMAKE_BUILD_TYPE={config}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={output}{os.sep}",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{config.upper()}={output}{os.sep}",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            "-DDISPER26_BUILD_TESTS=OFF",
            "-DDISPER26_BUILD_FORTRAN=OFF",
            "-DDISPER26_BUILD_PYTHON=ON",
            "-DDISPER26_REQUIRE_PYTHON=ON",
        ]
        # CMake selects the explicit override, bundled source, or pinned fetch.
        args += shlex.split(os.environ.get("CMAKE_ARGS", ""))
        subprocess.check_call(args)
        subprocess.check_call([
            "cmake", "--build", str(build_dir), "--config", config,
            "--target", "disper26py",
            "--parallel", str(self.parallel or 2),
        ])


setup(
    name="disper26py",
    version="0.1.105",
    author="Kiwamu Nishida",
    license="GPL-2.0-or-later",
    options={"sdist": {"owner": "root", "group": "root"}},
    license_files=["LICENSE", "NOTICE"],
    description="Python bindings for Disper26 VTI Dispersion Engine via pybind11",
    ext_modules=[Extension("disper26py", sources=[])],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(include=["disper26", "disper26.*"]) + ["disper26_inv"],
    package_dir={"disper26_inv": "inversion/disper26_inv"},
    extras_require={
        "inversion": ["scipy>=1.6"],
        "paper": ["scipy>=1.6", "matplotlib"],
        "test": ["pytest", "mpmath"],
    },
    package_data={"disper26": ["py.typed", "*.pyi"], "disper26.data": ["*.csv"]},
    install_requires=["numpy>=1.20"],
    zip_safe=False,
    python_requires=">=3.8",
)
