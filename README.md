# mojo-ahocorapy

Mojo port of the compute core of [ahocorapy](https://github.com/WoLpH/ahocorapy)
— the C extension that wraps `pyahocorasick` — version 1.8.0. The Python
package is named `mojo_ahocorapy`, so it installs alongside the real
`ahocorapy` and the parity tests import both and compare them match for match.

## What the compute core is

ahocorapy is a thin binding. Its own Python surface is `add_word`,
`make_automaton`, `iter`, `iter_long`, `find_all`, `exists`, `keys`, `pop`,
`remove_word`, `save`, and a couple of `STORE_*` options, and all of it is a
handful of lines of C over two data structures. Those two data structures are
the whole computational content of the library, and both are ported here:

- **`make_automaton`**, the breadth-first pass that computes the failure
  function and materialises the transition table where a missing edge is
  replaced by the corresponding edge of the failure state. That is
  `build_automaton_table` in the C extension and it is O(states x sigma).
- **the search**, one table lookup per text byte, reporting every pattern that
  ends at each position through the output links. That is `next_state` plus
  the output walk.

Trie insertion stays in Python, because it is string work over the caller's
patterns rather than a numeric loop.

## Covered subset

| area | implemented API |
| --- | --- |
| construction | `add_word` (str, bytes or a sequence of ints), `add_wordlist`, `make_automaton`, `clear`, `kind`, `__len__` |
| search | `iter` (every match, in pyahocorasick's order), `iter_positions` (the full span), `exists`, `find_all` |
| inspection | `keys`, `values`, `items` |
| kernels | `ac_build` (table, failure links, output links), `ac_search` |

Details that the parity tests pin down:

- the alphabet is the 256 byte values, as for a `KEY_STRING` automaton, and a
  `str` pattern or text is taken one character per byte (`latin-1`), which is
  how pyahocorasick treats a wide string. `'\x80'` is the single byte 0x80, not
  its two-byte UTF-8 encoding;
- at one end position the current state is reported first, then its output
  chain, so the longest match at that position comes first: `iter('ushers')`
  over `he, she, his, hers` yields `[(3, 'she'), (3, 'he'), (5, 'hers')]`;
- every match is reported, including matches that are suffixes of longer ones
  and overlapping matches;
- a repeated pattern keeps its position in the insertion order and its last
  value, as upstream does;
- `kind` is `TRIE` (1) before `make_automaton` and `AHOCORASICK` (2) after,
  matching this version's constants.

## Not implemented

- **`iter_long`**. pyahocorasick's "non-overlapping longest match" selection
  could not be pinned down from its observable behaviour: on `['ab', 'a']`
  over `'ab'` it reports `a`, and on `['a', 'ab']` it reports `ab`, with the
  same trie in both cases, so the choice is not a function of the automaton,
  the text or the pattern lengths. Rather than ship a near-miss, the method is
  left out. Use the real `ahocorapy` for it.
- `save`/`load` (pickle of the compiled automaton), `remove_word`, `pop`,
  `match`, `longest_prefix`, `dump` and `get_stats`: dictionary and
  serialisation plumbing, not a numeric loop.
- Non-string keys (`KEY_SEQUENCE` with arbitrary integer tokens) and the
  `STORE_INTS` / `STORE_ANY` storage options. A sequence pattern works here
  only if its items fit in a byte, because the table is byte-indexed.
- Empty patterns, which `add_word` rejects with a `ValueError` rather than
  handling.

## Install

```bash
pixi install
pixi run build     # -> dist/libmojo-ahocorapy.so
pixi run test
pixi run bench
```

## Tests

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

18 tests, 17 of them match-for-match against the real `ahocorapy` on a
`KEY_STRING` automaton: the classic `he/she/his/hers` example, patterns that
are prefixes and suffixes of one another, overlapping and repeated matches,
texts shorter than the patterns, a 26-pattern automaton over English prose, a
12-deep chain of patterns that forces the failure walk to iterate, bytes above
127, duplicate patterns, and 40 randomised pattern-set/text pairs. The one
non-reference test is the byte-text path, which a `KEY_STRING` reference
automaton refuses to `iter`; it is checked against an explicit expectation.

## Performance

Best-of-five, same process, against the C extension itself, which is the
strongest available baseline for a one-table-lookup-per-byte scan. Every case
compares the full match lists before timing.

| case | ahocorapy (C) | mojo-ahocorapy | result |
| --- | ---: | ---: | ---: |
| search, 8 patterns, 1 MiB text | 20.60 ms | 7.78 ms | 2.65x faster |
| search, 64 patterns, 1 MiB text | 35.38 ms | 9.46 ms | 3.74x faster |
| search, 512 patterns, 1 MiB text | 43.52 ms | 23.69 ms | 1.84x faster |
| search, 4096 patterns, 1 MiB text | 84.33 ms | 56.82 ms | 1.48x faster |
| build, 512 patterns | 0.94 ms | 12.73 ms | 0.07x, slower |
| build, 4096 patterns | 7.12 ms | 1203.23 ms | 0.01x, slower |

The search is 1.5x to 3.7x faster, and the reason is the table layout. The
table is dense over all 256 byte values in `int64`, so the inner loop is one
load and no branch on whether an edge exists: the failure transition is
already in the table. pyahocorasick keeps a sparse edge list and walks failure
links explicitly, which is cheaper in memory and more work per byte.

That is also why the build is much slower, and the two facts are the same
decision: completing a state's row is a 256-cell copy, so construction is
O(states x 256) — 30,000 states is 7.7 million cell writes and a 61 MB table
— where the C side is O(edges). For search-heavy use, which is what an
Aho-Corasick automaton is for, that is the right way round. For a
build-once-never-search workload it is the wrong way round, and a sparse
transition structure would be the fix.

## How it works

`src/kernels.mojo` is one compilation unit with two exported kernels, compiled
to `dist/libmojo-ahocorapy.so`. Buffers cross the C ABI as 64-bit addresses
and are rebuilt in Mojo as `Pointer[Int64, AnyOrigin[mut=True]]`.

`ac_build` takes the trie as a CSR child list — state `s` owns
`child_byte[child_ptr[s]:child_ptr[s + 1]]` and `child_to[...]` — and writes
the transition table, the failure links and the output links. The queue holds
the root, and a state's row is completed the moment the state is discovered,
so the row of a failure state is always final before a deeper state needs to
read it. The trie itself is assembled in the Python shim.

`ac_search` is the scan: one `Int64` load per byte, then the output-link walk
for the matches at that position. It returns the number of matches found and
writes into caller-owned arrays; a non-raising `abi("C")` function cannot
signal an overflow, so the shim sizes the output for the worst case (one match
per pattern per position) and the kernel returns early rather than writing out
of bounds if even that is not enough.

## License

MIT
