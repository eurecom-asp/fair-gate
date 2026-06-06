#!/usr/bin/env python3
from pathlib import Path
import ast

TARGETS = [
    "train.py",
    "fair_utils.py",
    "ECAPAModel.py",
    "ecapa_fair_model.py",
    "model.py",
    "loss.py",
    "dataLoader.py",
    "tools.py",
]

def list_defs(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[SKIP] {path}: {e}")
        return

    print(f"\n# {path}")
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            print(f"FUNC  {node.name}")
        elif isinstance(node, ast.ClassDef):
            print(f"CLASS {node.name}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    print(f"  METHOD {item.name}")

def main():
    for name in TARGETS:
        path = Path(name)
        if path.exists():
            list_defs(path)

if __name__ == "__main__":
    main()
