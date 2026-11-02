"""C++ compilation of leaf boxes (native/compile_boxes.cpp), loaded through ctypes.

Build-time only: the boxes are identical to those of ``scm._compile_trees``, in the same order (checked by
``tests/test_native.py`` and ``experiments/native_check.py``), so queries are not affected.  The shared
library is built next to the source on first use; without a compiler :func:`available` is False and
``scm.bind_scm`` falls back to Python.
"""

from __future__ import annotations

import ctypes as C
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DIR = Path(__file__).parent / "native"
_SRC = _DIR / "compile_boxes.cpp"
_LIB = None
_P_I64, _P_F64, _P_I32 = C.POINTER(C.c_int64), C.POINTER(C.c_double), C.POINTER(C.c_int32)


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    so = _DIR / f"compile_boxes_{os.uname().machine}.so"
    if not so.exists() or so.stat().st_mtime < _SRC.stat().st_mtime:
        tmp = so.with_suffix(f".{os.getpid()}.tmp")
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC", "-ffp-contract=off", "-pthread",
                        str(_SRC), "-o", str(tmp)], check=True, capture_output=True)
        os.replace(tmp, so)   # atomic: concurrent jobs may build at the same time
    lib = C.CDLL(str(so))
    lib.tw_compile.restype = C.c_void_p
    lib.tw_compile.argtypes = [
        C.c_int, C.c_int64, _P_I64, _P_I64, _P_F64, _P_I64, _P_I64, _P_I32, _P_F64, _P_I64, C.c_int64, _P_I64,
        C.POINTER(_P_F64), C.POINTER(_P_F64), C.POINTER(_P_I64), C.POINTER(_P_F64), _P_F64, C.c_int, C.c_double,
        C.c_int64, C.c_int, _P_I64,
    ]
    lib.tw_count.restype = C.c_int64
    lib.tw_count.argtypes = [C.c_void_p]
    lib.tw_copy.restype = None
    lib.tw_copy.argtypes = [C.c_void_p, _P_I64, _P_I64, _P_F64, _P_F64]
    lib.tw_free.restype = None
    lib.tw_free.argtypes = [C.c_void_p]
    _LIB = lib
    return lib


def available() -> bool:
    try:
        _load()
        return True
    except (OSError, subprocess.CalledProcessError, FileNotFoundError):
        return False


@dataclass
class Boxes:
    tree: np.ndarray    # n
    node: np.ndarray    # n, node index inside its tree
    lo: np.ndarray      # n x d
    hi: np.ndarray      # n x d


@dataclass
class NativeMediator:
    """The arrays of ``scm._Mediator``: trees with a single box always select the unit's own leaf and are dropped."""

    lo: np.ndarray
    hi: np.ndarray
    tree: np.ndarray
    diff: np.ndarray
    observed: float


def flat_trees(ens) -> dict:
    ptr = np.zeros(len(ens.trees) + 1, dtype=np.int64)
    ptr[1:] = np.cumsum([len(t.feature) for t in ens.trees])
    cat = lambda name, dtype: np.concatenate([np.asarray(getattr(t, name), dtype=dtype) for t in ens.trees])
    return dict(ptr=ptr, feature=cat("feature", np.int64), threshold=cat("threshold", np.float64),
                left=cat("left", np.int64), right=cat("right", np.int64), value=cat("value", np.float64),
                native_leaf=cat("native_leaf", np.int64))


def compile_trees(ens, flat: dict, inputs: list, d: int, max_boxes: int, threads: int = 1) -> tuple[Boxes | None, int]:
    """Returns ``(boxes, status)``; status 1: more than ``max_boxes`` boxes.  The result does not depend on ``threads``."""
    lib = _load()
    p = len(inputs)
    kind = np.zeros(p, dtype=np.int32)
    fixed = np.zeros(p, dtype=np.float64)
    arg = np.zeros(p, dtype=np.int64)
    meds: list[NativeMediator] = []
    for j, (k, a) in enumerate(inputs):
        if k == "fixed":
            kind[j], fixed[j] = 0, a
        elif k == "axis":
            kind[j], arg[j] = 1, a
        else:
            kind[j], arg[j] = 2, len(meds)
            meds.append(a)
    los = (_P_F64 * max(1, len(meds)))(*[m.lo.ctypes.data_as(_P_F64) for m in meds])
    his = (_P_F64 * max(1, len(meds)))(*[m.hi.ctypes.data_as(_P_F64) for m in meds])
    trs = (_P_I64 * max(1, len(meds)))(*[m.tree.ctypes.data_as(_P_I64) for m in meds])
    dfs = (_P_F64 * max(1, len(meds)))(*[m.diff.ctypes.data_as(_P_F64) for m in meds])
    ns = np.array([len(m.tree) for m in meds] or [0], dtype=np.int64)
    obs = np.array([m.observed for m in meds] or [0.0], dtype=np.float64)
    status = C.c_int64(0)
    h = lib.tw_compile(
        d, len(ens.trees), flat["ptr"].ctypes.data_as(_P_I64), flat["feature"].ctypes.data_as(_P_I64),
        flat["threshold"].ctypes.data_as(_P_F64), flat["left"].ctypes.data_as(_P_I64),
        flat["right"].ctypes.data_as(_P_I64), kind.ctypes.data_as(_P_I32), fixed.ctypes.data_as(_P_F64),
        arg.ctypes.data_as(_P_I64), len(meds), ns.ctypes.data_as(_P_I64), los, his, trs, dfs,
        obs.ctypes.data_as(_P_F64), int(ens.input_dtype == np.float32), float(ens.zero_threshold), max_boxes,
        threads, C.byref(status))
    try:
        if status.value != 0:
            return None, int(status.value)
        n = lib.tw_count(h)
        tree, node = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
        lo, hi = np.empty((n, d), dtype=np.float64), np.empty((n, d), dtype=np.float64)
        lib.tw_copy(h, tree.ctypes.data_as(_P_I64), node.ctypes.data_as(_P_I64), lo.ctypes.data_as(_P_F64),
                    hi.ctypes.data_as(_P_F64))
        return Boxes(tree, node, lo, hi), 0
    finally:
        lib.tw_free(h)


def mediator_from(boxes: Boxes, flat: dict, observed: float, observed_leaf: np.ndarray) -> NativeMediator:
    counts = np.bincount(boxes.tree, minlength=len(observed_leaf))
    keep = counts[boxes.tree] > 1
    tree = boxes.tree[keep]
    value = flat["value"][flat["ptr"][tree] + boxes.node[keep]]
    return NativeMediator(boxes.lo[keep], boxes.hi[keep], tree, value - observed_leaf[tree], float(observed))
