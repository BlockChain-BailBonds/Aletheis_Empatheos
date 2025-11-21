cat > multilingual_sigil_engine.py << 'EOF'
#!/usr/bin/env python3
# ================================================================================
# GlyphMatics • Universal Multilingual Sigil Engine (ULS Engine)
# Matthew Blake Ward (Nine1Eight) • SigilOS • GNC2 Layer
#
# TRUE PIPELINE (PER LANGUAGE):
#   WORD → G2P → IPA → IPA-GLYPHS → TEXT-GLYPH → DEFINITION-GLYPH → L-SIGIL
#
# GLOBAL PIPELINE:
#   COMBINE ALL LANGUAGES →
#       {1} JSON SUPERSIGIL
#       {2} MERGED GLYPH SUPERSTRING
# ================================================================================

import os, json, re, unicodedata, hashlib
from datetime import datetime

# ================================================================================
# LOAD 111-GLYPH ALPHABET
# ================================================================================
GLYPH_ALPHABET_111 = [
    "𖤐","𖤓","𖤔","𖤕","𖤖","𖤗","𖤘","𖤙","𖤚","𖤛","𖤜",
    "𖤝","𖤞","𖤟","𖤠","𖤡","𖤢","𖤣","𖤤","𖤥","𖤦",
    "𖤧","𖤨","𖤩","𖤪","𖤫","𖤬","𖤭","𖤮","𖤯","𖤰",
    "𖤱","𖤲","𖤳","𖤴","𖤵","𖤶","𖤷","𖤸","𖤹","𖤺",
    "𖤻","𖤼","𖤽","𖤾","𖤿","𖥀","𖥁","𖥂","𖥃","𖥄",
    "𖥅","𖥆","𖥇","𖥈","𖥉","𖥊","𖥋","𖥌","𖥍","𖥎",
    "𖥏","𖥐","𖥑","𖥒","𖥓","𖥔","𖥕","𖥖","𖥗","𖥘",
    "𖥙","𖥚","𖥛","𖥜","𖥝","𖥞","𖥟","𖥠","𖥡","𖥢",
    "𖥣","𖥤","𖥥","𖥦","𖥧","𖥨","𖥩","𖥪","𖥫","𖥬",
    "𖥭","𖥮","𖥯","𖥰","𖥱","𖥲","𖥳","𖥴","𖥵","𖥶",
    "𖥷","𖥸","𖥹","𖥺","𖥻","𖥼","𖥽","𖥾","𖥿"
]
ALEN = len(GLYPH_ALPHABET_111)

# ================================================================================
# DIRECTORIES
# ================================================================================
DICT_DIR = os.path.expanduser("~/GlyphMatics/dictionaries")
PHON_DIR = os.path.expanduser("~/GlyphMatics/languages_spoken")
ARCHIVE_DIR = os.path.expanduser("~/.glyphmatics")
os.makedirs(ARCHIVE_DIR, exist_ok=True)
ARCHIVE_FILE = os.path.join(ARCHIVE_DIR, "universal_sigil_archive.jsonl")

# ================================================================================
# TEXT → GLYPH
# ================================================================================
def text_to_glyphs(text: str) -> str:
    b = unicodedata.normalize("NFC", text).encode("utf-8")
    return "".join(GLYPH_ALPHABET_111[v % ALEN] for v in b)

# ================================================================================
# LOAD DICTIONARIES + PHONEMES
# ================================================================================
def load_dictionary(lang):
    path = os.path.join(DICT_DIR, f"{lang.lower()}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path,"r",encoding="utf-8"))

def load_phonemes(lang):
    path = os.path.join(PHON_DIR, f"{lang.lower()}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path,"r",encoding="utf-8"))

# ================================================================================
# G2P ENGINE (uses your g2p_engine module)
# ================================================================================
from g2p_engine import g2p as G2P

# ================================================================================
# IPA → GLYPHS
# ================================================================================
def ipa_to_glyphs(ipa_list):
    out=[]
    for ipa in ipa_list:
        hv=sum(ord(c) for c in ipa)
        out.append(GLYPH_ALPHABET_111[hv % ALEN])
    return "".join(out)

# ================================================================================
# PER-LANGUAGE SIGIL
# ================================================================================
def build_language_sigil(lang, word):
    dic = load_dictionary(lang)
    phn = load_phonemes(lang)
    if dic is None or phn is None:
        return None

    if word not in dic:
        return None

    definition = dic[word]

    ipa_tokens = G2P(lang, word)
    ipa_glyph = ipa_to_glyphs(ipa_tokens)
    word_glyph = text_to_glyphs(word)
    def_glyph = text_to_glyphs(definition)

    combined = f"{lang}::{word}::{definition}::" + "".join(ipa_tokens)
    digest = hashlib.sha3_256(combined.encode("utf-8")).hexdigest()

    sigil = f"⊏⚗${digest[:16]}:{ipa_glyph}:{word_glyph}$⊐"

    return {
        "language": lang,
        "word": word,
        "definition": definition,
        "ipa_tokens": ipa_tokens,
        "ipa_glyphs": ipa_glyph,
        "word_glyph": word_glyph,
        "definition_glyph": def_glyph,
        "sha3_256": digest,
        "sigil": sigil
    }

# ================================================================================
# GLOBAL SUPER-SIGIL (FUSION)
# ================================================================================
def build_global_sigil(word, languages=None):
    if languages is None:
        # auto-detect available languages
        languages = [
            f[:-5] for f in os.listdir(DICT_DIR)
            if f.endswith(".json")
        ]

    per_lang = {}
    merged_glyphs = []

    for lang in languages:
        s = build_language_sigil(lang, word)
        if s:
            per_lang[lang]=s
            merged_glyphs.append(s["ipa_glyphs"])
            merged_glyphs.append(s["word_glyph"])

    merged_stream = "".join(merged_glyphs)

    combined = word + "::" + "::".join(
        s["sha3_256"] for s in per_lang.values()
    )
    digest = hashlib.sha3_256(combined.encode("utf-8")).hexdigest()

    supersigil = {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "word": word,
        "languages": per_lang,
        "merged_glyph_stream": merged_stream,
        "sha3_256": digest,
        "supersigil": f"⊏⚗${digest[:24]}:{merged_stream}$⊐"
    }

    with open(ARCHIVE_FILE,"a",encoding="utf-8") as f:
        f.write(json.dumps(supersigil, ensure_ascii=False)+"\n")

    return supersigil

# ================================================================================
# CLI
# ================================================================================
if __name__=="__main__":
    import argparse, json
    p=argparse.ArgumentParser(description="Universal Multilingual Sigil Engine")
    p.add_argument("word")
    args=p.parse_args()

    out = build_global_sigil(args.word)
    print(json.dumps(out, ensure_ascii=False, indent=2))

EOF
