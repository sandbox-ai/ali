from pprint import pprint as pp
from rich import print

def head_dict(dict, n=1):
    return {k: dict[k] for k in list(dict)[:n]}


def pprint(*args, **kwargs):
    for arg in args:
        pp(arg, **kwargs)
