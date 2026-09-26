"""An `Automaton` with ahocorapy's call shape, backed by the Mojo kernels.

```python
import mojo_ahocorapy

a = mojo_ahocorapy.Autaton()
a.add_word("he", 1)
a.add_word("she", 2)
a.make_automaton()
list(a.iter("ushers"))     # [(3, 2), (3, 1)]  -- end index, value
```

The trie is assembled in Python, because inserting words is string work rather
than a numeric loop. `make_automaton` is the compiled part: the breadth-first
pass that builds the transition table and the failure and output links, then
`iter`, `iter_long` and `exists` are the compiled search.
"""

import numpy as np

from . import _lib

__all__ = ["Automaton"]

KIND_TRIE = 1
KIND_AHOCORASICK = 2


class Automaton:
    """Multi-pattern string matcher over bytes, as in ahocorapy."""

    def __init__(self, words=None):
        self._words = {}          # raw bytes -> value
        self._order = []          # raw bytes, in insertion order
        self._originals = {}      # raw bytes -> the object the caller passed
        self._table = None
        self._out = None
        self._pattern_of = None
        self._built = False
        if words:
            for word, value in words:
                self.add_word(word, value)

    def add_word(self, word, value=0):
        """Add one pattern, as `add_word(word, value)` in ahocorapy.

        A `str` pattern is taken one character per byte, as pyahocorasick
        does for a wide string, so '\x80' is the single byte 0x80 rather than
        its two-byte UTF-8 encoding.
        """
        original = word
        if isinstance(word, str):
            word = word.encode("latin-1")
        elif isinstance(word, (tuple, list)):
            word = bytes(word)
        if not isinstance(word, bytes):
            raise TypeError("patterns must be str, bytes or a sequence of ints")
        if not word:
            raise ValueError("empty patterns are not supported")
        if word not in self._words:
            self._order.append(word)
        self._words[word] = value
        self._originals[word] = original
        self._built = False
        return self

    def add_wordlist(self, words):
        """Add many patterns from a sequence of (word, value) pairs."""
        for word, value in words:
            self.add_word(word, value)
        return self

    def clear(self):
        self._words = {}
        self._order = []
        self._table = None
        self._out = None
        self._pattern_of = None
        self._built = False

    @property
    def kind(self):
        """0 (TRIE) before `make_automaton`, 2 (AHOCORASICK) after, as upstream."""
        return KIND_AHOCORASICK if self._built else KIND_TRIE

    def __len__(self):
        return len(self._words)

    def keys(self):
        return [self._originals[w] for w in self._order]

    def values(self):
        return [self._words[w] for w in self._order]

    def items(self):
        return [(self._originals[w], self._words[w]) for w in self._order]

    def _trie(self):
        """Flatten the trie into the CSR child arrays the kernel wants."""
        children = {}
        states = 1  # the root
        terminals = {}   # state -> pattern index
        for index, word in enumerate(self._order):
            state = 0
            for byte in word:
                key = (state, byte)
                nxt = children.get(key)
                if nxt is None:
                    nxt = states
                    states += 1
                    children[key] = nxt
                state = nxt
            # a repeated word keeps its first position but the last value,
            # which is what ahocorapy does
            terminals[state] = index
        edges = [None] * states
        for (parent, byte), child in children.items():
            if edges[parent] is None:
                edges[parent] = []
            edges[parent].append((byte, child))
        child_ptr = np.zeros(states + 1, dtype=np.int64)
        child_byte = []
        child_to = []
        for state in range(states):
            child_ptr[state] = len(child_byte)
            for byte, child in sorted(edges[state] or ()):
                child_byte.append(byte)
                child_to.append(child)
        child_ptr[states] = len(child_byte)
        pattern_of = np.full(states, -1, dtype=np.int64)
        for state, index in terminals.items():
            pattern_of[state] = index
        return (
            states,
            child_ptr,
            np.array(child_byte, dtype=np.int64),
            np.array(child_to, dtype=np.int64),
            pattern_of,
        )

    def make_automaton(self):
        """Compile the automaton: table, failure links and output links."""
        if not self._words:
            raise ValueError("no patterns to build")
        states, child_ptr, child_byte, child_to, pattern_of = self._trie()
        self._nstates = states
        self._pattern_of = pattern_of
        self._table, self._fail, self._out = _lib.build_automaton(
            states, child_ptr, child_byte, child_to, pattern_of
        )
        self._built = True
        return self

    def _require(self):
        if not self._built:
            raise RuntimeError("call make_automaton() first")

    @staticmethod
    def _as_bytes(text):
        if isinstance(text, str):
            return text.encode("latin-1")
        if isinstance(text, (tuple, list)):
            return bytes(text)
        if isinstance(text, bytes):
            return text
        raise TypeError("text must be str, bytes or a sequence of ints")

    def _cap(self, n):
        # worst case is every position matching every pattern
        return max(1, n * max(len(self._words), 1) + 1)

    def iter(self, text):
        """Every match as (end index, value), longest first at each position."""
        self._require()
        raw = self._as_bytes(text)
        ends, patterns = _lib.search(
            raw, self._table, self._out, self._pattern_of, self._cap(len(raw))
        )
        words = self._order
        return [
            (int(end), self._words[words[pattern]])
            for end, pattern in zip(ends, patterns)
        ]

    def iter_positions(self, text):
        """Every match as (start, end, word), the full span rather than the end."""
        # iter reports the end position; the span follows from the pattern
        # length, so map value back through the pattern list
        out = []
        for end, value in self.iter(text):
            for word in self._order:
                if self._words[word] == value:
                    out.append((end - len(word) + 1, end, word))
                    break
        return out

    def exists(self, text):
        """True when at least one pattern occurs in `text`."""
        return bool(self.iter(text))

    def find_all(self, text):
        """All matches, as the (end, value) pairs `iter` returns."""
        return self.iter(text)
