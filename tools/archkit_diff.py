#!/usr/bin/env python3
"""Print the differences between the hand-written GGUF tables (foreign_names) and the rows generated from llama.cpp, per architecture:
tensor names (compare_hand) and metadata keys (keys the hand config builder reads vs keys the converter writes).  python -I tools/archkit_diff.py"""
from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from kyojin_mlx import foreign_archkit as K, foreign_names as N  # noqa: E402


def hand_keys(arch: str) -> set:
    """GGUF keys (without the arch prefix) the hand config builder of `arch` reads: every `a("key")` / `a("key", default)` of the builder and of the builders it calls."""
    seen, todo, keys = set(), [N._CONFIG[N.ARCHS[arch].config]], set()
    while todo:
        fn = todo.pop()
        if fn in seen:
            continue
        seen.add(fn)
        src = inspect.getsource(fn)
        keys |= set(re.findall(r"\ba\(\s*\"([\w.]+)\"", src))
        todo += [getattr(N, m) for m in re.findall(r"\b(_cfg_\w+|_guard_common)\(", src) if hasattr(N, m) and m != fn.__name__]
    return keys


def written_keys(arch: str) -> dict:
    return {r["key"][len(arch) + 1:]: r["src"][0] for r in K.metadata_rows(arch) if r["key"].startswith(arch + ".")}


def main():
    for arch, A in N.ARCHS.items():
        t = K.compare_hand(arch, A)
        h, w = hand_keys(arch), written_keys(arch)
        direct = {k for k, s in w.items() if s in ("hparam", "block_count", "rope")}
        print(f"== {arch} (mlx-lm model_type {A.model_type}); converter flags {list(K.converter_flags(arch))}")
        print(f"   tensors: not a converter candidate {[x[0] + ' -> ' + x[1] for x in t['not_candidate']]}; unknown to gguf-py {t['unknown_kind']}")
        print(f"            read by llama.cpp's loader, no hand row: {t['not_in_hand']}")
        print(f"   metadata: converter writes {len(w)} keys ({len(direct)} direct), hand reads {len(h)}; "
              f"hand reads, converter never writes {sorted(h - set(w))}; converter writes, hand never reads {sorted(set(w) - h)}")


if __name__ == "__main__":
    main()
