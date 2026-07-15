"""`display` is an IPython builtin: it exists in a notebook cell but NOT inside an
imported module. Every ported cell that called it would raise NameError. This
shim makes the modules run identically in a notebook and in a plain script."""
try:
    from IPython.display import display  # noqa: F401
except ImportError:                       # pragma: no cover
    def display(*objs, **kwargs):
        for o in objs:
            try:
                print(o.to_string())
            except AttributeError:
                print(o)
