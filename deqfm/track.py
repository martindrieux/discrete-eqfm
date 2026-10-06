"""The nb experiment tracker, optional. `@nb.core` marks the functions that define the algorithms;
without the tracker installed it does nothing and training runs unchanged."""
try:
    import nb
except Exception:         # noqa: BLE001
    class nb:             # stand-in with the one decorator the library code uses
        core = staticmethod(lambda f: f)
        start = None
