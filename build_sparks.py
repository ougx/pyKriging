#!/usr/bin/env python3
"""
build_sparks.py
============
Compile the SPARKS Fortran sources into an executable.

Usage
-----
python build_sparks.py               # auto-detect compiler
python build_sparks.py --compiler gfortran
python build_sparks.py --compiler ifx
python build_sparks.py --compiler ifort
python build_sparks.py --opt debug  # no optimisation, add -g
python build_sparks.py --no-openmp  # Disable OpenMP parallelization

The compiled library is placed in:
    bin/sparks (Linux / macOS)
    bin/sparks.exe (Windows)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Source files in dependency order (each module compiled before its users)
# ---------------------------------------------------------------------------
SOURCES = [
    "common.f90",
    "utils.F90",
    "progress_bar.F90",
    "rotation.f90",
    "variogram.f90",
    "kdtree2_maxidx.f90",
    "gaussian_quadrature.f90",
    "lapack.f",
    "solver.f90",
    "kriging.F90",
    "../sparks/f90getopt.F90",
    "../sparks/io.f90",
    "../sparks/sparks.f90",
]

# ---------------------------------------------------------------------------
# Compiler flag sets
# ---------------------------------------------------------------------------
_ON_WINDOWS = sys.platform == "win32"

def _intel_flags(opt_win, opt_linux, debug_win, debug_linux, shared_win, shared_linux):
    """Return platform-correct Intel release/debug/shared/implib flag lists."""
    if _ON_WINDOWS:
        return {
            "release": opt_win,
            "debug": debug_win,
            "shared": shared_win,
            "implib": ["/link", "/FORCE:MULTIPLE", "/OPT:REF", "/OPT:ICF"],
        }
    else:
        return {
            "release": opt_linux,
            "debug": debug_linux,
            "shared": shared_linux,
            "implib": [],
        }

FLAGS = {
    "gfortran": {
        "release": ["-O2", "-fdefault-real-8", "-fopenmp", "-cpp", "-fbacktrace", "-ffree-line-length-none"],
        "debug": ["-O0", "-g", "-fdefault-real-8", "-fopenmp", "-Wall", "-fcheck=all", "-fbacktrace", "-cpp", "-ffree-line-length-none"],
        "shared": [],
        "implib": [],
    },
    "ifx": _intel_flags(
        opt_win = ["/O2", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/warn:all", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "-check all"],
        shared_win = ['/nologo', '/MT'],
        shared_linux = ['-nologo', '-static', '-MT']
    ),
    "ifort": _intel_flags(
        opt_win = ["/O2", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/fpp"],
        opt_linux = ["-O2", "-real-size:64", "-qopenmp", "-traceback", "-fpp"],
        debug_win = ["/Od", "/debug:full", "/real-size:64", "/Qopenmp", "/heap-arrays:0", "/traceback", "/warn:all", "/fpp", "/check:all"],
        debug_linux=["-O0", "-g", "-real-size:64", "-qopenmp", "-traceback", "-fpp", "-warn all", "-check all"],
        shared_win = ['/nologo', '/MT'],
        shared_linux = ['-nologo', '-static', '-MT']
    ),
}

def detect_compiler():
    for compiler in ("ifx", "gfortran", "ifort"):
        if shutil.which(compiler):
            return compiler
    raise RuntimeError(
        "No Fortran compiler found. Install gfortran (Linux/macOS) or "
        "Intel oneAPI (ifx/ifort) and ensure it is on PATH."
    )

def output_name(compiler: str) -> str:
    return "sparks"

def build(compiler: str, arg: argparse.ArgumentParser, fortran_dir: Path, out_dir: Path):
    flag_set = FLAGS.get(compiler)
    if flag_set is None:
        raise ValueError(f"Unknown compiler {compiler!r}. Choose: gfortran, ifx, ifort")

    if arg.no_openmp:
        openmp_flags = {
            "gfortran": ["fopenmp"],
            "ifx": ["qopenmp", "Qopenmp"],
            "ifort": ["qopenmp", "Qopenmp"],
        }
        for flag in flag_set[arg.opt]:
            if flag[1:] in openmp_flags.get(compiler, []):
                flag_set[arg.opt].remove(flag)

    out_name = output_name(compiler)
    out_path = out_dir / out_name
    sources = [str(fortran_dir / s) for s in SOURCES]

    cmd = (
        [compiler] + flag_set[arg.opt] + flag_set["shared"] + sources + ["-o", str(out_path)] + flag_set["implib"]
    )

    print("Compiling with:")
    print("  ", " ".join(cmd))
    print()

    result = subprocess.run(cmd, capture_output=False, cwd=out_dir if sys.platform == "win32" else None)
    if result.returncode != 0:
        print(f"\nCompilation failed (exit code {result.returncode})")
        sys.exit(result.returncode)

    print(f"\nSuccess: {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Build the pykriging Fortran library.")
    parser.add_argument("--compiler", default=None, help="Fortran compiler: gfortran, ifx, ifort (default: auto-detect)")
    parser.add_argument("--opt", default="release", choices=["release", "debug"], help="Optimisation level (default: release)")
    parser.add_argument("--no-openmp", action="store_true", help="Disable OpenMP parallelization")
    args = parser.parse_args()

    compiler = args.compiler or detect_compiler()
    print(f"Compiler: {compiler}")
    print(f"Mode: {args.opt}")

    root = Path(__file__).parent
    fortran_dir = root / "src" / "libkriging"
    out_dir = root / "bin"

    if not fortran_dir.exists():
        raise FileNotFoundError(f"Fortran source directory not found: {fortran_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    build(compiler, args, fortran_dir, out_dir)

if __name__ == "__main__":
    main()
