"""ctypes bridge to the compiled Mojo Aho-Corasick kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a
64-bit address, so the argtypes below must stay `c_int64` for addresses;
`c_int` truncates them and segfaults.
"""

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-ahocorapy.so"

# byte alphabet, as pyahocorasick uses for KEY_STRING
SIGMA = 256


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))
    lib.ac_build.restype = None
    lib.ac_build.argtypes = [ctypes.c_int64] * 10
    lib.ac_search.restype = ctypes.c_int64
    lib.ac_search.argtypes = [ctypes.c_int64] * 10
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    return a.ctypes.data


def build_automaton(nstates, child_ptr, child_byte, child_to, pattern_of):
    """Transition table, failure links and output links for a trie.

    `nstates` includes the root at state 0. Returns
    `(table, fail, out)` with `table` shaped (nstates, 256).
    """
    child_ptr = np.ascontiguousarray(child_ptr, dtype=np.int64)
    child_byte = np.ascontiguousarray(child_byte, dtype=np.int64)
    child_to = np.ascontiguousarray(child_to, dtype=np.int64)
    pattern_of = np.ascontiguousarray(pattern_of, dtype=np.int64)
    table = np.zeros((nstates, SIGMA), dtype=np.int64)
    fail = np.zeros(nstates, dtype=np.int64)
    out = np.full(nstates, -1, dtype=np.int64)
    queue = np.zeros(nstates, dtype=np.int64)
    lib.ac_build(
        nstates, _addr(child_ptr), _addr(child_byte), _addr(child_to), SIGMA,
        _addr(table), _addr(fail), _addr(out), _addr(pattern_of), _addr(queue),
    )
    return table, fail, out


def search(text, table, out, pattern_of, cap):
    """Every match of the automaton in `text`, as (end, pattern index) pairs.

    The start of a match is not reported by the kernel: the automaton is a
    trie, and the caller knows the byte length of each pattern, so the start is
    `end - len(pattern) + 1`. The shim reconstructs it, which is why this
    returns end positions only.
    """
    text = np.frombuffer(text, dtype=np.uint8).astype(np.int64)
    res_end = np.zeros(cap, dtype=np.int64)
    res_pattern = np.zeros(cap, dtype=np.int64)
    found = lib.ac_search(
        _addr(text), text.size, _addr(table), SIGMA, _addr(out),
        _addr(pattern_of), 0, _addr(res_end), _addr(res_pattern), cap,
    )
    return res_end[:found], res_pattern[:found]
