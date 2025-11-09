#!/usr/bin/env python3
import json, os
REGISTRY_PATH = os.path.expanduser("~/Aletheis_Empatheos/core/translation/glyph_registry.json")
OUTPUT_PATH   = os.path.expanduser("~/Aletheis_Empatheos/codex/glyphnotes_codex.json")

def build_codex():
    if not os.path.exists(REGISTRY_PATH):
        print("Registry not found.")
        return
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    codex = []
    for w, v in data.items():
        entry = {
            "word": w,
            "rehydration_sigil": v.get("rehydration_sigil"),
            "languages": list(v.get("languages", {}).keys()),
        }
        codex.append(entry)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(codex, f, ensure_ascii=False, indent=2)
    print(f"Codex saved → {OUTPUT_PATH} ({len(codex)} entries)")
if __name__ == "__main__":
    build_codex()
