#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a compact Codex view from glyph_registry.json:

- word
- rehydration_sigil
- list of languages with entries

Used by GlyphNotes / UI.
"""

import json
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parents[1]
REG_PATH    = BASE_DIR / "core/translation/glyph_registry.json"
OUTPUT_PATH = BASE_DIR / "codex" / "Codex.json"

def build_codex():
    if not REG_PATH.exists():
        raise SystemExit(f"Missing registry: {REG_PATH}")

    with REG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise SystemExit(f"Registry is not an object at root: {type(data)}")

    codex = []
    for w, v in data.items():
        if not isinstance(v, dict):
            # ignore stray metadata like strings, etc.
            continue
        languages = v.get("languages") or {}
        if not isinstance(languages, dict):
            languages = {}
        entry = {
            "word": w,
            "rehydration_sigil": v.get("rehydration_sigil"),
            "languages": sorted(list(languages.keys())),
        }
        codex.append(entry)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(codex, f, ensure_ascii=False, indent=2)

    print(f"Codex saved → {OUTPUT_PATH} ({len(codex)} entries)")

if __name__ == "__main__":
    build_codex()
