"""Correctness-gated benchmark for mojo-ahocorapy.

Every case compares the Mojo match list with the real ahocorapy match list
before timing, so a wrong automaton shows up as a correctness failure rather
than as a suspiciously good number. The baseline is the C extension itself:
ahocorapy wraps pyahocorasick, which is a hand-written C automaton and a
strong baseline for a single-pattern-per-byte scan.
"""

from __future__ import annotations

import pathlib
import random
import sys
import time

import ahocorasick as ref

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_ahocorapy  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _words(count, rng, min_len=4, max_len=12):
    """Patterns over a 16-character alphabet: enough overlap to exercise the
    failure links, small enough that the table stays modest."""
    alphabet = "abcdefghijklmnop"
    seen = set()
    out = []
    while len(out) < count:
        length = rng.randint(min_len, max_len)
        word = "".join(rng.choice(alphabet) for _ in range(length))
        if word in seen:
            continue
        seen.add(word)
        out.append(word)
    return out


def _text(nbytes, rng):
    alphabet = "abcdefghijklmnop"
    return "".join(rng.choice(alphabet) for _ in range(nbytes))


def bench_search(n_words, n_bytes, seed=0):
    rng = random.Random(seed)
    words = _words(n_words, rng)
    text = _text(n_bytes, rng)

    mine = mojo_ahocorapy.Automaton()
    theirs = ref.Automaton(ref.STORE_ANY, ref.KEY_STRING)
    for index, word in enumerate(words):
        mine.add_word(word, index)
        theirs.add_word(word, index)
    mine.make_automaton()
    theirs.make_automaton()

    got = mine.iter(text)
    expect = list(theirs.iter(text))
    assert got == expect, "match lists differ"

    ref_time = _time(lambda: list(theirs.iter(text)))
    mojo_time = _time(lambda: mine.iter(text))
    return f"search {n_words} patterns / {n_bytes} bytes", ref_time, mojo_time


def bench_build(n_words, seed=0):
    rng = random.Random(seed + 100)
    words = _words(n_words, rng)
    def build_mine():
        automaton = mojo_ahocorapy.Automaton()
        for index, word in enumerate(words):
            automaton.add_word(word, index)
        automaton.make_automaton()
        return automaton

    def build_theirs():
        automaton = ref.Automaton(ref.STORE_ANY, ref.KEY_STRING)
        for index, word in enumerate(words):
            automaton.add_word(word, index)
        automaton.make_automaton()
        return automaton

    # both sides are timed over the whole construction: pyahocorasick builds
    # its automaton during add_word, so timing make_automaton alone would
    # measure a no-op
    mojo_time = _time(build_mine, 3)
    ref_time = _time(build_theirs, 3)
    return f"build {n_words} patterns", ref_time, mojo_time


def main():
    print(f"{'case':<38}{'ahocorapy (C)':>16}{'mojo-ahocorapy':>16}{'ratio':>9}")
    print("-" * 79)
    cases = [
        lambda: bench_search(8, 1 << 20, seed=1),
        lambda: bench_search(64, 1 << 20, seed=2),
        lambda: bench_search(512, 1 << 20, seed=3),
        lambda: bench_search(4096, 1 << 20, seed=4),
        lambda: bench_build(512, seed=5),
        lambda: bench_build(4096, seed=6),
    ]
    for case in cases:
        label, ref_t, got_t = case()
        ratio = ref_t / got_t if got_t else float("nan")
        print(f"{label:<38}{ref_t*1e3:>14.2f}ms{got_t*1e3:>14.2f}ms{ratio:>8.2f}x")


if __name__ == "__main__":
    main()
