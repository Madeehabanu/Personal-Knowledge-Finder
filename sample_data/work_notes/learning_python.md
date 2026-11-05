# Study notes: Python decorators

A decorator is a function that wraps another function to extend its behaviour without
modifying it. Common uses are logging, caching with functools.lru_cache, and access control.
Remember that functools.wraps preserves the original function's name and docstring.
