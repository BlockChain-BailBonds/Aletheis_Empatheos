#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glyph_translate_all.py

Termux pipeline:

- Load 1000 words from glyph_word_list.json (20 groups × 50 words).
- Use local LLaMA server to generate definitions in 100 languages.
- Encode each definition to glyph string via GlyphTranslator.
- Build per-language sigils AND one rehydration sigil per word
  from the concatenation of all language glyph strings.
- Save incremental registry to glyph_registry.json so runs can be resumed
  and new words/languages can be added later.
- Create timestamped snapshot copies on every successful run.
"""

import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
import requests

from glyph_translator import GlyphTranslator
from glyph_sigilmaker import glyphs_to_sigil

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

BASE_DIR = Path("/data/data/com.termux/files/home/llama")

WORDLIST_PATH   = BASE_DIR / "glyph_word_list.json"    # 1000-word list (you already have this)
REGISTRY_PATH   = BASE_DIR / "glyph_registry.json"     # incremental output library
LOG_PATH        = BASE_DIR / "translate_log.txt"

# Your glyphstore for GlyphTranslator
CODEX_PATH      = Path("/storage/emulated/0/Download/unified_ledger.glyphstore")

# LLaMA server
LLAMA_URL       = "http://127.0.0.1:8080/completion"
N_PREDICT       = 192
TEMPERATURE     = 0.4

# 100 languages
LANGUAGES = [
    "English","French","German","Spanish","Italian","Portuguese","Russian","Chinese","Japanese","Korean",
    "Arabic","Hindi","Bengali","Urdu","Turkish","Persian","Vietnamese","Indonesian","Thai","Swahili",
    "Zulu","Afrikaans","Amharic","Somali","Hausa","Yoruba","Igbo","Dutch","Swedish","Norwegian",
    "Danish","Finnish","Polish","Czech","Slovak","Hungarian","Romanian","Bulgarian","Serbian","Croatian",
    "Bosnian","Slovenian","Greek","Hebrew","Yiddish","Latin","Catalan","Galician","Basque","Welsh",
    "Irish","Scottish Gaelic","Breton","Ukrainian","Belarusian","Georgian","Armenian","Azerbaijani","Kazakh","Uzbek",
    "Turkmen","Kyrgyz","Tajik","Malay","Filipino","Lao","Khmer","Burmese","Nepali","Sinhala",
    "Marathi","Gujarati","Punjabi","Tamil","Telugu","Kannada","Malayalam","Odia","Mongolian","Tibetan",
    "Maori","Samoan","Tongan","Malagasy","Luxembourgish","Icelandic","Estonian","Latvian","Lithuanian","Albanian",
    "Macedonian","Kurdish","Pashto","Faroese","Haitian Creole","Esperanto","Xhosa","Quechua","Guarani","Nahuatl",
]

# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------

def log(msg: str) -> None:
    msg = msg.rstrip("\n")
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def flatten_word_list(path: Path):
    data = load_json(path, {})
    words = []
    seen = set()
    for group, lst in data.items():
        if not isinstance(lst, list):
            continue
        for w in lst:
            if not isinstance(w, str):
                continue
            key = w.strip()
            if not key:
                continue
            cap = key[0].upper() + key[1:]
            if cap not in seen:
                seen.add(cap)
                words.append(cap)
    return words

def llama_define(word: str, lang: str) -> str | None:
    prompt = (
        f"Define the single word \"{word}\" in {lang}. "
        f"Respond only with a short definition in {lang}, no examples, no quotes, no extra commentary."
    )
    payload = {"prompt": prompt, "n_predict": N_PREDICT, "temperature": TEMPERATURE}
    try:
        r = requests.post(LLAMA_URL, json=payload, timeout=600)
    except Exception as e:
        log(f"[ERROR] LLaMA request failed for {word} ({lang}): {e}")
        return None
    if not r.ok:
        log(f"[ERROR] HTTP {r.status_code} for {word} ({lang})")
        return None
    try:
        data = r.json()
    except Exception as e:
        log(f"[ERROR] JSON decode failed for {word} ({lang}): {e}")
        return None
    text = data.get("content", "")
    if not isinstance(text, str):
        log(f"[WARN] Unexpected JSON format for {word} ({lang}): {data}")
        return None
    return text.strip()

def context_hash(word: str, definition: str, lang: str) -> str:
    s = f"{word}|{lang}|{definition}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def main():
    log("=== glyph_translate_all: start ===")
    translator = GlyphTranslator(str(CODEX_PATH))
    registry = load_json(REGISTRY_PATH, {})
    if not isinstance(registry, dict):
        registry = {}
    log(f"Loaded registry with {len(registry)} words from {REGISTRY_PATH}")
    words = flatten_word_list(WORDLIST_PATH)
    log(f"Loaded word list: {len(words)} words from {WORDLIST_PATH}")

    for idx, word in enumerate(words, start=1):
        entry = registry.get(word) or {"languages": {}, "rehydration_sigil": None}
        langs_map = entry["languages"]
        log(f"\n--- [{idx}/{len(words)}] {word} ---")
        used_sigils = {v.get('sigil') for v in langs_map.values() if isinstance(v, dict)}
        used_sigils.discard(None)
        updated_any = False

        for lang in LANGUAGES:
            if lang in langs_map and "glyphs" in langs_map[lang]:
                continue
            log(f"  -> {word} [{lang}] ...")
            definition = llama_define(word, lang)
            if not definition:
                log("     [SKIP] no definition")
                continue
            glyphs = translator.text_to_glyphs(definition)
            base_hash = context_hash(word, glyphs, lang)
            attempt, sigil = 0, None
            while True:
                h = base_hash if attempt == 0 else base_hash + f"#{attempt}"
                sigil_candidate = glyphs_to_sigil(glyphs + h[:4])
                if sigil_candidate not in used_sigils:
                    sigil = sigil_candidate
                    used_sigils.add(sigil)
                    break
                attempt += 1
                if attempt > 16:
                    sigil = sigil_candidate
                    break
            langs_map[lang] = {"definition": definition, "glyphs": glyphs, "sigil": sigil}
            log(f"     len={len(definition)}  sigil={sigil}")
            updated_any = True
            time.sleep(0.05)

        if langs_map:
            concatenated = " ".join(
                langs_map[l]["glyphs"]
                for l in sorted(langs_map.keys())
                if "glyphs" in langs_map[l]
            )
            if concatenated:
                re_sig = glyphs_to_sigil(concatenated)
                entry["rehydration_sigil"] = re_sig
                log(f"  rehydration_sigil = {re_sig}")
                updated_any = True

        registry[word] = entry
        if updated_any:
            save_json(REGISTRY_PATH, registry)
            log(f"  [SAVE] registry updated ({len(registry)} words)")

    # --- session snapshot ---
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    snapshot_path = REGISTRY_PATH.with_name(f"glyph_registry_{ts}.json")
    save_json(snapshot_path, registry)
    log(f"  [SNAPSHOT] saved full copy to {snapshot_path}")

    log("\n✅ Completed pass over word list.")
    log(f"   Registry path: {REGISTRY_PATH}")
    log("=== glyph_translate_all: done ===")

if __name__ == "__main__":
    main()
