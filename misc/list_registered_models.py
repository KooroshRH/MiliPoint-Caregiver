#!/usr/bin/env python3
"""
Print the canonical `--model` keys from the mmrnet model registry.

Use this before launching runs (especially via `runner.py`) to avoid
name mismatches between filenames and registry keys.
"""

from __future__ import annotations

import ast
import pathlib


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    registry_path = repo_root / "mmrnet" / "models" / "__init__.py"
    src = registry_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(registry_path))

    keys: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "model_map" for t in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise SystemExit("model_map is not a dict literal; cannot parse keys safely.")
        for k in node.value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                raise SystemExit("model_map has a non-string key; cannot parse safely.")
        break

    if not keys:
        raise SystemExit("Could not find model_map assignment in mmrnet/models/__init__.py")

    keys = sorted(keys)
    print(f"{len(keys)} registered models:")
    for k in keys:
        print(k)


if __name__ == "__main__":
    main()

