cat > ~/llama/glyph_translate_all.py <<'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glyph_translate_all.py
Uses the local llama.cpp server to translate a list of word definitions
into many languages, then encodes each into glyphs and a rehydration sigil.
"""
import requests, json, time, hashlib
from glyph_translator import GlyphTranslator
from glyph_sigilmaker import glyphs_to_sigil

LLAMA_URL = "http://127.0.0.1:8080/completion"
CODEx = "/storage/emulated/0/Download/unified_ledger.glyphstore"
translator = GlyphTranslator(CODEx)

languages = [
    "French", "German", "Spanish", "Italian", "Portuguese", "Russian",
    "Arabic", "Hindi", "Japanese", "Chinese", "Korean", "Swahili",
    "Turkish", "Greek", "Hebrew", "Thai", "Vietnamese", "Indonesian"
]

words = [
    "consciousness", "truth", "energy", "symbol", "language",
    "creation", "memory", "dream", "spirit", "knowledge"
]

def query_llama(text: str) -> str:
    payload = {"prompt": text, "n_predict": 128, "temperature": 0.7}
    r = requests.post(LLAMA_URL, json=payload, timeout=600)
    return r.json().get("content", "").strip()

results = {}
for w in words:
    results[w] = {}
    for lang in languages:
        prompt = f"Define '{w}' and translate the definition into {lang}."
        print(f"\n→ {prompt}")
        raw = query_llama(prompt)
        glyphs = translator.text_to_glyphs(raw)
        sigil = glyphs_to_sigil(glyphs)
        results[w][lang] = {"text": raw, "glyphs": glyphs, "sigil": sigil}
        print(f"{lang}: {sigil}")
        time.sleep(1)

with open("glyph_definitions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nAll definitions encoded → glyph_definitions.json")
EOF
chmod +x ~/llama/glyph_translate_all.py
