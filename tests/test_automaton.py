"""Parity tests for the Mojo Aho-Corasick kernels against the real ahocorapy.

ahocorapy is the C extension wrapping pyahocorasick; `iter` over a text is its
central operation and the automaton table is the whole of its state. Every test
here compares the two match-for-match, in the same order, including the order
within a single end position, which is the thing a wrong output-link
implementation gets wrong.

The cases are chosen to hit the parts of the construction that are easy to get
wrong: patterns that are prefixes and suffixes of each other, overlapping
matches, a pattern that is a prefix of the text, a text shorter than the
patterns, byte values above 127, and duplicate patterns.
"""

import random

import ahocorasick as ref
import pytest

import mojo_ahocorapy


def pair(words):
    """A reference automaton and a Mojo automaton over the same words."""
    theirs = ref.Automaton(ref.STORE_ANY, ref.KEY_STRING)
    mine = mojo_ahocorapy.Automaton()
    for index, word in enumerate(words):
        theirs.add_word(word, index + 1)
        mine.add_word(word, index + 1)
    theirs.make_automaton()
    mine.make_automaton()
    return theirs, mine


def compare(words, text):
    theirs, mine = pair(words)
    assert list(mine.iter(text)) == list(theirs.iter(text))


def test_classic_example():
    words = ["he", "she", "his", "hers"]
    compare(words, "ushers")
    compare(words, "she")
    compare(words, "he")
    compare(words, "")
    compare(words, "s")


def test_prefix_and_suffix_relationships():
    # 'a' is a prefix of 'ab' and a suffix of 'ba'; the output chain has to
    # report both at the same end position, deepest first
    compare(["a", "ab", "ba"], "aba")
    compare(["ab", "a", "b"], "abab")
    compare(["aa", "a"], "aaaa")
    compare(["abc", "bc", "c", "b"], "abcb")


def test_overlapping_and_repeated_matches():
    compare(["aa"], "aaaa")
    compare(["aba"], "ababa")
    compare(["a", "b", "c"], "abcabcabc")
    compare(["needle"], "needle in a haystack needle")


def test_text_shorter_than_the_pattern():
    compare(["abcdef"], "abc")
    compare(["abcdef"], "")
    compare(["a"], "")


def test_match_at_every_position():
    words = [chr(ord("a") + i) for i in range(26)]
    compare(words, "the quick brown fox jumps over the lazy dog")
    compare(words, "z")


def test_high_byte_values():
    # bytes above 127 must go through the 256-entry table, not be truncated.
    # A str is one character per byte, as pyahocorasick treats a wide string.
    words = ["\x80\xff", "\xff", "\x00a"]
    compare(words, "xx\x80\xffyy\xffa\x00a")
    compare(words, "\x80\xff")
    compare(["\x00", "\x00a"], "a\x00\x00a")


def test_bytes_text_matches_a_hand_computed_expectation():
    # a KEY_STRING reference automaton refuses a bytes object, so the byte
    # path is checked against an explicit expectation instead
    mine = mojo_ahocorapy.Automaton()
    mine.add_word(b"\xc3\xa9", 1)
    mine.add_word(b"\xff", 2)
    mine.make_automaton()
    assert list(mine.iter(b"x\xc3\xa9\xff")) == [(2, 1), (3, 2)]
    assert list(mine.iter(b"\xff\xc3\xa9")) == [(0, 2), (2, 1)]


def test_duplicate_pattern_keeps_the_last_value():
    theirs = ref.Automaton(ref.STORE_ANY, ref.KEY_STRING)
    theirs.add_word("ab", 1)
    theirs.add_word("ab", 2)
    theirs.make_automaton()
    mine = mojo_ahocorapy.Automaton()
    mine.add_word("ab", 1)
    mine.add_word("ab", 2)
    mine.make_automaton()
    assert len(mine) == len(theirs) == 1
    assert list(mine.iter("abab")) == list(theirs.iter("abab"))


def test_long_pattern_chain():
    # a chain long enough that the failure walk has to iterate
    words = ["a" * k for k in range(1, 12)]
    compare(words, "a" * 20)
    compare(words, "ba" * 5 + "a" * 7)


def test_single_character_patterns_over_binary_text():
    words = ["a"]
    rng = random.Random(4)
    text = "".join(rng.choice("ab") for _ in range(200))
    compare(words, text)
    words = ["ab", "ba", "aa", "bb"]
    compare(words, text)


def test_randomised_pattern_and_text_pairs():
    rng = random.Random(1234)
    for trial in range(40):
        alphabet = "ab"
        count = rng.randint(1, 6)
        words = []
        for _ in range(count):
            length = rng.randint(1, 5)
            words.append("".join(rng.choice(alphabet) for _ in range(length)))
        text = "".join(
            rng.choice(alphabet) for _ in range(rng.randint(0, 40))
        )
        compare(words, text)


def test_exists_and_find_all_agree():
    theirs, mine = pair(["he", "she", "his", "hers"])
    for text in ["ushers", "she", "he", "xyz", "", "hers hers"]:
        assert mine.exists(text) == bool(list(theirs.iter(text)))
        assert mine.find_all(text) == list(theirs.iter(text))


def test_iter_positions_reports_the_full_span():
    # iter gives the end; the span is end - len(word) + 1
    words = ["he", "she", "hers"]
    _, mine = pair(words)
    got = sorted(mine.iter_positions("ushers"))
    assert got == [(1, 3, b"she"), (2, 3, b"he"), (2, 5, b"hers")]


def test_keys_values_and_len_match():
    words = ["he", "she", "his"]
    _, mine = pair(words)
    assert sorted(mine.keys()) == sorted(words)
    assert len(mine) == len(words) == 3
    assert mine.values() == [1, 2, 3]
    assert sorted(mine.items()) == sorted((w, i + 1) for i, w in enumerate(words))


def test_kind_reports_trie_then_automaton():
    mine = mojo_ahocorapy.Automaton()
    mine.add_word("a", 1)
    assert mine.kind == ref.TRIE
    mine.make_automaton()
    assert mine.kind == ref.AHOCORASICK


def test_bytes_and_sequence_patterns():
    mine = mojo_ahocorapy.Automaton()
    mine.add_word(b"ab", 1)
    mine.add_word((99, 100), 2)
    mine.make_automaton()
    assert list(mine.iter(b"xabcd")) == [(2, 1), (4, 2)]
    assert list(mine.iter(b"zzcd")) == [(3, 2)]


def test_errors_are_loud():
    mine = mojo_ahocorapy.Automaton()
    with pytest.raises(RuntimeError):
        mine.iter("abc")  # not built yet
    with pytest.raises(ValueError):
        mine.add_word("", 1)
    with pytest.raises(TypeError):
        mine.add_word(1.5, 1)
    with pytest.raises(ValueError):
        mojo_ahocorapy.Automaton().make_automaton()


def test_adding_a_word_invalidates_the_automaton():
    mine = mojo_ahocorapy.Automaton()
    mine.add_word("ab", 1)
    mine.make_automaton()
    assert list(mine.iter("ab")) == [(1, 1)]
    mine.add_word("bc", 2)
    with pytest.raises(RuntimeError):
        mine.iter("abc")
    mine.make_automaton()
    assert list(mine.iter("abc")) == [(1, 1), (2, 2)]
