"""Aho-Corasick automaton construction and search: ahocorapy's compute core.

ahocorapy is a thin C wrapper around pyahocorasick. Everything it exposes
(`add_word`, `exists`, `find_all`, `iter`, `iter_long`, `keys`, `save`) is a
few lines of C over two data structures, and those two structures are the
whole computational content of the library:

- `make_automaton` walks the trie breadth first, computes the failure
  function, and materialises a dense transition table where a missing edge
  is replaced by the corresponding edge of the failure state. That is
  `build_automaton_table` in the C extension, and it is O(states * sigma).
- the search walks the text one byte at a time through that table, following
  failure links implicitly (the table already encodes them) and reporting
  every pattern that ends at each position through the output links. That is
  `next_state` in the C extension.

Both are here. Trie insertion stays in Python, because it is string work over
the caller's words rather than a numeric loop.

Buffers cross the C ABI as 64-bit addresses, `@export` rejects parametric
functions, and `AnyOrigin[mut=True]` is the only usable mutable origin, so each
export takes `Int` addresses and rebuilds its own pointer.
"""

comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


@export("ac_build")
def ac_build(nstates: Int, child_ptr_addr: Int, child_byte_addr: Int,
             child_to_addr: Int, sigma: Int, table_addr: Int, fail_addr: Int,
             out_addr: Int, pattern_of_addr: Int, queue_addr: Int) abi("C"):
    """Build the transition table, the failure links and the output links.

    The trie arrives as a CSR child list: state `s` owns
    `child_byte[child_ptr[s] : child_ptr[s + 1]]` / `child_to[...]`, one entry
    per real trie edge. `pattern_of[s]` is the pattern index of a terminal
    state, or -1.

    Rows are completed in breadth-first order, so the failure state's row is
    always finished before a state that fails into it: a state's row is its
    failure row with its own real edges written over it. That is what makes
    the search loop a single table lookup per byte, as in the C extension.

    `table` is nstates x sigma and is fully written; `fail` and `out` are
    nstates; `queue` is a scratch buffer of nstates.
    """
    var child_ptr = ip(child_ptr_addr)
    var child_byte = ip(child_byte_addr)
    var child_to = ip(child_to_addr)
    var table = ip(table_addr)
    var fail = ip(fail_addr)
    var out = ip(out_addr)
    var pattern_of = ip(pattern_of_addr)
    var queue = ip(queue_addr)
    var alphabet = Int64(sigma)

    # start from "no transition", then write the real edges
    for i in range(nstates * Int(sigma)):
        table[unsafe_offset=i] = -1
    for s in range(nstates):
        var lo = child_ptr[unsafe_offset=s]
        var hi = child_ptr[unsafe_offset=s + 1]
        for e in range(lo, hi):
            table[unsafe_offset=Int64(s) * alphabet + child_byte[unsafe_offset=e]] = (
                child_to[unsafe_offset=e]
            )

    for s in range(nstates):
        fail[unsafe_offset=s] = 0
        out[unsafe_offset=s] = -1

    # the root falls back to itself, so its missing edges are self loops
    for c in range(sigma):
        if table[unsafe_offset=c] == -1:
            table[unsafe_offset=c] = 0

    # Breadth first, with the root in the queue. A state's row is completed
    # the moment the state is discovered, so by the time a deeper state needs
    # to read the row of its failure state that row is already final.
    queue[unsafe_offset=0] = 0
    var head = 0
    var tail = 1
    while head < tail:
        var s = queue[unsafe_offset=head]
        head += 1
        var lo = child_ptr[unsafe_offset=s]
        var hi = child_ptr[unsafe_offset=s + 1]
        for e in range(lo, hi):
            var c = child_byte[unsafe_offset=e]
            var t = child_to[unsafe_offset=e]
            var g = Int64(0)
            if s != 0:
                var f = fail[unsafe_offset=s]
                var step = table[unsafe_offset=Int64(f) * alphabet + c]
                while step == -1 and f != 0:
                    f = fail[unsafe_offset=f]
                    step = table[unsafe_offset=Int64(f) * alphabet + c]
                if step != -1:
                    g = step
            fail[unsafe_offset=t] = g
            if pattern_of[unsafe_offset=g] >= 0:
                out[unsafe_offset=t] = g
            else:
                out[unsafe_offset=t] = out[unsafe_offset=g]
            # complete t's row: the failure row, then its own real edges
            var base_t = Int64(t) * alphabet
            var base_f = Int64(g) * alphabet
            for sym in range(sigma):
                table[unsafe_offset=base_t + Int64(sym)] = table[unsafe_offset=base_f + Int64(sym)]
            # the real edges of t, not of its parent
            var tlo = child_ptr[unsafe_offset=t]
            var thi = child_ptr[unsafe_offset=t + 1]
            for e2 in range(tlo, thi):
                table[unsafe_offset=base_t + child_byte[unsafe_offset=e2]] = (
                    child_to[unsafe_offset=e2]
                )
            queue[unsafe_offset=tail] = t
            tail += 1


@export("ac_search")
def ac_search(text_addr: Int, n: Int, table_addr: Int, sigma: Int,
              out_addr: Int, pattern_of_addr: Int, res_start_addr: Int,
              res_end_addr: Int, res_pattern_addr: Int,
              cap: Int) abi("C") -> Int64:
    """Run the automaton over `n` text bytes and report every match.

    The state walk is one table lookup per byte. At each position the current
    state is reported first, then its output chain, which is the order
    pyahocorasick's `iter` yields in: longest match first at a given end
    position.

    Matches are written as (end, pattern index) pairs into two arrays of
    capacity `cap`, and the number of matches found is returned. A non-raising
    `abi("C")` function cannot signal an overflow, so the caller must size
    `cap` for the worst case (one match per pattern per position); this shim
    does.
    """
    var text = ip(text_addr)
    var table = ip(table_addr)
    var out = ip(out_addr)
    var pattern_of = ip(pattern_of_addr)
    var res_end = ip(res_end_addr)
    var res_pattern = ip(res_pattern_addr)

    var alphabet = Int64(sigma)
    var limit = Int64(cap)
    var state = 0
    var found = Int64(0)
    for i in range(n):
        var c = text[unsafe_offset=i]
        state = Int(table[unsafe_offset=Int64(state) * alphabet + c])
        var p = pattern_of[unsafe_offset=state]
        if p >= 0:
            if found >= limit:
                return found
            res_end[unsafe_offset=Int(found)] = Int64(i)
            res_pattern[unsafe_offset=Int(found)] = p
            found += 1
        var t = out[unsafe_offset=state]
        while t >= 0:
            if found >= limit:
                return found
            res_end[unsafe_offset=Int(found)] = Int64(i)
            res_pattern[unsafe_offset=Int(found)] = pattern_of[unsafe_offset=t]
            found += 1
            t = out[unsafe_offset=t]
    return found
