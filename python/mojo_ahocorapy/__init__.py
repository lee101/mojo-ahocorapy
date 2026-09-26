"""Mojo port of ahocorapy's automaton construction and search.

[ahocorapy](https://github.com/WoLpH/ahocorapy) is a C extension wrapping
pyahocorasick. Its computational content is the Aho-Corasick transition table
and the search over it; both are ported here. Trie insertion, dictionary
plumbing and the `save`/`load` helpers are not.

```python
import mojo_ahocorapy

a = mojo_ahocorapy.Autaton()
a.add_word("he", 1)
a.add_word("she", 2)
a.make_automaton()
list(a.iter("ushers"))
```
"""

from .automaton import Automaton

__all__ = ["Automaton"]
__version__ = "0.1.0"
