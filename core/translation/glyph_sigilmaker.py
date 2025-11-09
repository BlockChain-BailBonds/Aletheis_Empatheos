#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glyph_sigilmaker.py
Collapses a glyph string into a single deterministic Sigil symbol.
Reversible: each sigil can rehydrate the original glyph string via index lookup.
"""
import hashlib, base64, json
from pathlib import Path

SIGIL_MAP_FILE = Path("sigil_map.json")

# create or load lookup
if SIGIL_MAP_FILE.exists():
    sigil_map = json.loads(SIGIL_MAP_FILE.read_text())
else:
    sigil_map = {}

SIGIL_ALPHABET = "⟡⚚☿⚛☀☁☂☃☄★☆⚝⚞⚟⚠⚡⚢⚣⚤⚥⚦⚧⚨⚩⚪⚫⚬⚭⚮⚯♁♆♇♈♉♊♋♌♍♎♏♐♑♒♓♔♕♖♗♘♙♚♛♜♝♞♟"

def glyphs_to_sigil(glyph_str: str) -> str:
    """Hash a glyph string into a stable single sigil character."""
    h = hashlib.sha256(glyph_str.encode("utf-8")).digest()
    idx = int.from_bytes(h[:2], "big") % len(SIGIL_ALPHABET)
    sigil = SIGIL_ALPHABET[idx]
    sigil_map[sigil] = glyph_str
    SIGIL_MAP_FILE.write_text(json.dumps(sigil_map, ensure_ascii=False, indent=2))
    return sigil

def sigil_to_glyphs(sigil: str) -> str:
    """Reverse lookup of sigil back into glyph string."""
    return sigil_map.get(sigil, "")

if __name__ == "__main__":
    from glyph_translator import GlyphTranslator
    CODEx = "/storage/emulated/0/Download/unified_ledger.glyphstore"
    translator = GlyphTranslator(CODEx)
    prompt = "Translate 'The mind speaks through symbols.' into French."
    glyph_str = translator.translate_to_glyph(prompt)
    sigil = glyphs_to_sigil(glyph_str)
    print("\nSigil:", sigil)
    print("\nRehydrated glyphs:\n", sigil_to_glyphs(sigil))
