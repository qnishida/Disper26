# Building Disper26

CMake 3.18+, a C++17 compiler, and Eigen 3.4.x are required. Ubuntu builds
use Eigen 3.4.x from the system package or an explicit `Eigen3_DIR`.
On macOS, Homebrew `eigen@3` is required, and the
complex automatic-differentiation code currently requires GNU GCC/libstdc++.
Apple Clang/libc++ cannot compile these instantiations. Install the native
prerequisites with Homebrew (skip packages already installed):

```bash
brew install gcc eigen@3 cmake
```

For a new native macOS build, CMake automatically selects Homebrew GCC unless
`CXX`, `CMAKE_CXX_COMPILER`, or a toolchain file is explicitly set. Linux keeps
CMake's usual compiler selection. Existing build directories retain their
compiler; use a new directory when changing compilers. An explicit Apple
Clang selection produces a configuration error explaining the requirement.

CMake locates `eigen@3` with `brew --prefix eigen@3` on macOS. This package
can coexist with Homebrew `eigen` (Eigen 5); it does not need to be linked
globally. Each configuration selects its package directory, replacing any
cached or explicit `Eigen3_DIR` on macOS. This also prevents Conda/uv from
selecting another Eigen installation. On Linux, normal CMake package discovery
and `Eigen3_DIR` overrides remain available. All builds reject Eigen versions
outside 3.4.x during configuration.

## C++ library, executable, and tests

Run these commands from the project root:

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j4
(cd build-native && ctest --output-on-failure)
```

To include the Fortran interface and its tests, install a Fortran compiler and
add `-DDISPER26_BUILD_FORTRAN=ON` to the configure command.

To omit the C++ tests, configure with `-DDISPER26_BUILD_TESTS=OFF`. To omit the
optional Python module, use `-DDISPER26_BUILD_PYTHON=OFF`. The latter is otherwise
built when the selected Python provides pybind11 and development headers.
Select that interpreter explicitly with `-DPython_EXECUTABLE=/path/to/python`.

On platforms where `long double` has no more precision than `double` (including
Apple Silicon), CMake reports and excludes the four Taylor/shared-hyperbolic
reference tests. Their high-precision assertions are preserved. Other test
targets remain enabled; this exclusion is not a claim of numerical validation.

CTest includes the following regression checks. They can also be run directly:

```bash
./build-native/tests/test_brent_known_values
./build-native/tests/test_romberg_root_trace
./build-native/tests/verify_mode_count
```

Runtime assertions remain enabled for the C++ test targets in Release builds.
Production library and executable compiler flags are unchanged.

## Python extension: Conda or uv

Use a dedicated environment. From the project root containing `setup.py` and
`pyproject.toml`, choose one of the following. `disper26` below is an example
Conda environment name, not a required pre-existing environment. Choose any
name, or activate an existing dedicated environment:

```bash
conda create -c conda-forge -n disper26 python=3.12 pip
conda activate disper26
python -m pip install .
```

Or use uv:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python .
```

The remaining `python` commands use the selected environment's interpreter.
Activate the uv environment with `source .venv/bin/activate`, or use its Python
path explicitly. For non-interactive Conda commands, use
`conda run -n ENV_NAME python ...`, replacing `ENV_NAME` with your environment
name. Python 3.8+ is supported; 3.12 above is an example choice.

The Python build uses CMake and the same compiler/header discovery as the C++
build. `pyproject.toml` declares setuptools, wheel, CMake, and pybind11 as build
dependencies, so pip/uv prepare them in an isolated build environment. Only
`disper26py` is built; C++ tests and the Fortran interface are disabled. Existing
CMake binaries are not reused. The module is installed into the selected Python
environment, not into every Python on the machine.

For custom locations or offline dependencies, pass CMake options through
`CMAKE_ARGS`, for example on Linux (the Eigen directory must provide 3.4.x):

```bash
CMAKE_ARGS="-DCMAKE_CXX_COMPILER=/path/to/g++ -DEigen3_DIR=/path/to/eigen3/cmake -DFETCHCONTENT_SOURCE_DIR_AUTODIFF=/path/to/autodiff" python -m pip install .
```

For `python setup.py build_ext --inplace`, install the build dependencies into
the active environment first. From the project root run:

```bash
python -m pip install 'setuptools>=64' wheel 'cmake>=3.18' 'pybind11>=2.10'
python setup.py build_ext --inplace
```

## Dependency sources and validation

Native and Python builds select autodiff in the same order:

1. An explicit `FETCHCONTENT_SOURCE_DIR_AUTODIFF` source directory.
2. A populated `third_party/autodiff` submodule or the source bundled in the
   Python source distribution.
3. The public upstream commit
   `b0a4feff5b2a61262e94305452ac53369fe35e75`, fetched by CMake.

The submodule is pinned to that same commit. Previous build directories are
not searched for dependency sources. An explicit override is available for
local experiments; use the pinned source when reproducing release results.
The build prints the selected source directory.

GitHub source ZIPs do not contain submodule contents, so their first build
requires Git and network access for the fixed upstream fetch. A Python source
distribution includes the autodiff source and does not need that fetch. The
native Eigen and Python build dependencies must still be installed. No build
requires a private research repository. Eigen remains a native dependency:
Eigen 3.4.x on Ubuntu, and Homebrew `eigen@3` on macOS, including inside
Conda/uv.

Eigen 5 changes `Eigen::internal::SingleRange` into a template, which
autodiff v1.1.2 cannot compile unmodified. The build therefore requires
Eigen 3.4.x instead of patching autodiff headers for Eigen 5. No compatibility
header is generated or used. For an existing build reporting
`expected a type, got 'SingleRange'`, install `eigen@3` on macOS and re-run
the configuration and build commands above; the old Eigen cache is replaced.
The `autodiff_eigen_indexed` CTest checks row/column indexed-view gradients
against `2*x` with absolute tolerance 1e-12.

## Verify the Python installation

Install the test and paper extras to run the full suite, including the
long-period mode diagnostics that use SciPy for trapezoidal integration.

```bash
python -m pip install '.[test,paper]'
python -c "import disper26, disper26py; print(disper26py.__file__)"
python -m pytest -q tests/python/test_examples.py
python -m pytest -q tests/python
python tests/python/verify_integer_alignment.py
```

Some optional historical workflows also require the `paper` extra, which
provides SciPy and Matplotlib. Inverse modelling uses the `inversion` extra:

```bash
python -m pip install '.[inversion]'
python -c "import disper26_inv; print(disper26_inv.__version__)"
```

The forward API and practice examples need only NumPy at runtime. Importing
`disper26_inv` without SciPy reports how to install the extra. The existing
`paper` extra continues to include SciPy for compatibility.
See [paper reproduction](../paper/README.md).

Linux Python builds use `-march=native`, and wheels depend on the compiler
runtime. A wheel built locally is not a portable binary release.
