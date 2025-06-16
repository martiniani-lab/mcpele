import os
import sys
import subprocess
import shutil
import argparse
import shlex

import numpy as np
import pele
import sysconfig
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as old_build_ext

# Numpy header files
numpy_lib = os.path.split(np.__file__)[0]
numpy_include = np.get_include()

# find pele path
try:
    pelepath = os.path.dirname(pele.__file__)[:-5]
except Exception:
    sys.stderr.write("WARNING: couldn't find path to pele\n")
    sys.exit(1)

# argument parsing
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-j", type=int, default=4)
parser.add_argument("-c", "--compiler", type=str, default=None)
parser.add_argument(
    "--opt-report",
    action="store_true",
    help="Print optimization report (for Intel compiler). Default: False",
    default=False,
)
parser.add_argument(
    "--build-type",
    type=str,
    default="Release",
    help="Build type: Release, Debug, RelWithDebInfo, MemCheck",
)

jargs, remaining_args = parser.parse_known_args(sys.argv)
# record compiler choice
idcompiler = None
if not jargs.compiler or jargs.compiler in ("unix", "gnu", "gcc"):
    idcompiler = "unix"
    remaining_args += ["-c", idcompiler]
elif jargs.compiler in ("intelem", "intel", "icc", "icpc"):
    idcompiler = "intel"
    remaining_args += ["-c", idcompiler]

# reset argv
sys.argv = remaining_args
print(jargs, remaining_args)
cmake_parallel_args = [] if jargs.j is None else [f"-j{jargs.j}"]

# compiler flags by build type
build_type = jargs.build_type
flags = {
    "Release": ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-O3", "-fPIC", "-DNDEBUG", "-march=native"],
    "Debug":   ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-ggdb3", "-O0", "-fPIC"],
    "RelWithDebInfo": ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-g", "-O3", "-fPIC"],
    "MemCheck": ["-std=c++2a", "-Wall", "-Wextra", "-pedantic", "-g", "-O0", "-fPIC", "-fsanitize=address", "-fsanitize=leak"],
}
if build_type not in flags:
    raise ValueError(f"Unknown build type: {build_type}")
cmake_compiler_extra_args = flags[build_type]
if idcompiler.lower() == "unix":
    cmake_compiler_extra_args += ["-march=native", "-flto", "-fopenmp"]
else:
    cmake_compiler_extra_args += ["-axCORE-AVX2", "-ipo", "-qopenmp", "-ip", "-unroll"]
    if jargs.opt_report:
        cmake_compiler_extra_args.append("-qopt-report=5")

# write version

def git_version():
    def _minimal_ext_cmd(cmd):
        env = {k: os.environ[k] for k in ("SYSTEMROOT", "PATH") if k in os.environ}
        env.update({"LANGUAGE": "C", "LANG": "C", "LC_ALL": "C"})
        return subprocess.check_output(cmd, env=env)
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"]);
        return out.strip().decode()
    except Exception:
        return "Unknown"

os.makedirs(os.path.dirname("mcpele/version.py"), exist_ok=True)
with open("mcpele/version.py", "w") as f:
    f.write(f"# GENERATED\ngit_revision = '{git_version()}'\n")

# cythonize

def generate_cython():
    cwd = os.path.dirname(__file__)
    print("Cythonizing sources")
    cmd = [sys.executable, os.path.join(cwd, "cythonize.py"), "mcpele", "-I", f"{pelepath}/pele/potentials/"]
    if subprocess.call(cmd, cwd=cwd) != 0:
        raise RuntimeError("Running cythonize failed!")

generate_cython()

# CMake build dir
cmake_build_dir = os.path.join(os.getcwd(), "build", "cmake")

cxx_files = [
    "mcpele/monte_carlo/_pele_mc.cxx",
    "mcpele/monte_carlo/_monte_carlo_cpp.cxx",
    "mcpele/monte_carlo/_takestep_cpp.cxx",
    "mcpele/monte_carlo/_accept_test_cpp.cxx",
    "mcpele/monte_carlo/_conf_test_cpp.cxx",
    "mcpele/monte_carlo/_action_cpp.cxx",
    "mcpele/monte_carlo/_nullpotential_cpp.cxx",
]

def get_ldflags():
    gv = sysconfig.get_config_var
    pyver = gv("VERSION")
    libs = gv("LIBS").split() + gv("SYSLIBS").split() + [f"-lpython{pyver}"]
    if not gv("Py_ENABLE_SHARED"):
        libs.insert(0, "-L" + gv("LIBDIR"))
    if not gv("PYTHONFRAMEWORK"):
        libs += gv("LINKFORSHARED").split()
    return " ".join(libs)

# generate CMakeLists.txt
with open("CMakeLists.txt.in") as fin:
    template = fin.read()
cmake_txt = (template
    .replace("__PELE_DIR__", pelepath)
    .replace("__PYTHON_INCLUDE__", " ".join(sysconfig.get_paths()["include"].split(os.pathsep)))
    .replace("__NUMPY_INCLUDE__", numpy_include)
    .replace("__PYTHON_LDFLAGS__", get_ldflags())
    .replace("__COMPILER_EXTRA_ARGS__", '"%s"' % " ".join(cmake_compiler_extra_args))
)
with open("CMakeLists.txt", "w") as fout:
    fout.write(cmake_txt + "\n")
    for src in cxx_files:
        fout.write(f"make_cython_lib(${{CMAKE_CURRENT_SOURCE_DIR}}/{src})\n")

# prepare compiler environment
def set_compiler_env(cid):
    cc = shutil.which("gcc" if cid == "unix" else "icc")
    cxx = shutil.which("g++" if cid == "unix" else "icpc")
    ld = shutil.which("ld" if cid == "unix" else "xild")
    ar = shutil.which("ar" if cid == "unix" else "xiar")
    env = os.environ.copy()
    env.update({"CC": cc, "CXX": cxx, "LD": ld, "AR": ar})
    args = [f"-D CMAKE_C_COMPILER={cc}",
            f"-D CMAKE_CXX_COMPILER={cxx}",
            f"-D CMAKE_LINKER={ld}",
            f"-D CMAKE_AR={ar}"]
    return env, args

# run CMake and build
os.makedirs(cmake_build_dir, exist_ok=True)

def run_cmake():
    cwd = os.getcwd()
    env, cmake_args = set_compiler_env(idcompiler)
    subprocess.check_call(["cmake", *cmake_args, cwd], cwd=cmake_build_dir, env=env)
    subprocess.check_call(["make", *cmake_parallel_args], cwd=cmake_build_dir, env=env)
    print("CMake build completed")

run_cmake()

# custom build_ext to copy .so artifacts
class build_ext_precompiled(old_build_ext):
    def build_extension(self, ext):
        ext_path = self.get_ext_fullpath(ext.name)
        lib = ext.sources[0]
        if not (lib.endswith(".so") and os.path.isfile(lib)):
            raise RuntimeError(f"Invalid library: {lib}")
        os.makedirs(os.path.dirname(ext_path), exist_ok=True)
        shutil.copy2(lib, ext_path)

# setuptools setup
extensions = [Extension(src.replace("/", ".").rsplit(".",1)[0], [os.path.join(cmake_build_dir, os.path.basename(src).replace('.cxx','.so'))])
              for src in cxx_files]

setup(
    name="mcpele",
    version="0.1",
    description="mcpele: Monte Carlo and parallel tempering on pele foundation",
    url="https://github.com/pele-python/mcpele",
    packages=[
        "mcpele", "mcpele.monte_carlo", "mcpele.parallel_tempering",
        "mcpele.monte_carlo.tests", "mcpele.parallel_tempering.tests"
    ],
    cmdclass={"build_ext": build_ext_precompiled},
    ext_modules=extensions,
)
