"""Reads a Lightning checkpoint from an outside source without running any of its code.

Only tensors, storages and plain containers are rebuilt; every other class the pickle names (omegaconf configs,
tokenizers, ...) becomes an inert stub that records its arguments.   python -I scripts/safe_ckpt.py CKPT [OUT.pt]
OUT.pt (optional) receives {"state_dict": ..., "hparams": <stub tree as text>} as a plain torch file.
"""
import collections
import pickle
import sys

import torch

_SAFE = {("collections", "OrderedDict"): collections.OrderedDict, ("collections", "defaultdict"): collections.defaultdict,
         ("__builtin__", "dict"): dict, ("__builtin__", "list"): list, ("__builtin__", "long"): int,
         ("builtins", "dict"): dict, ("builtins", "list"): list, ("builtins", "int"): int,
         ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2}
for _n in ("FloatStorage", "HalfStorage", "BFloat16Storage", "LongStorage", "IntStorage", "BoolStorage"):
    _SAFE[("torch", _n)] = getattr(torch, _n)


class Stub:
    def __init__(self, *a, **k): self.args, self.state = a, None
    def __setstate__(self, s): self.state = s
    def __call__(self, *a, **k): return Stub(*a)
    def __repr__(self): return f"Stub{self.args!r}"


def _stub(module, name):
    return type(f"{module}.{name}", (Stub,), {})


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _SAFE.get((module, name)) or _stub(module, name)


class _Mod:   # torch.load's pickle_module interface
    Unpickler = SafeUnpickler
    load = staticmethod(lambda f, **kw: SafeUnpickler(f, **kw).load())
    __name__ = "safe_pickle"


def load(path):
    return torch.load(path, map_location="cpu", pickle_module=_Mod, weights_only=False)


if __name__ == "__main__":
    ck = load(sys.argv[1])
    sd = ck["state_dict"]
    print("top-level keys:", list(ck))
    for k, v in sd.items():
        print(f"{k:70s} {tuple(v.shape)} {v.dtype}")
    if len(sys.argv) > 2:
        torch.save({"state_dict": sd, "hparams": repr(ck.get("hyper_parameters"))}, sys.argv[2])
