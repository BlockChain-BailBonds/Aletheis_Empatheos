cat > sigil_reverse_decoder.py << 'EOF'
#!/usr/bin/env python3
# ====================================================================================
# GlyphMatics • Reverse Decoder Engine
# SIGIL → GLYPHS → IPA → WORD (per language)
# Matthew Blake Ward (Nine1Eight) • SigilOS • GNC2
#
# Fully real inverse linguistic pipeline:
#   supersigil → merged glyphs → IPA glyphs → IPA tokens → reverse G2P → candidate words
# ====================================================================================

import os, re, json, unicodedata, hashlib
from collections import defaultdict
from g2p_engine import g2p as FORWARD_G2P

# ====================================================================================
# Load glyph alphabet
# ====================================================================================
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

# ====================================================================================
# Directories
# ====================================================================================
DICT_DIR = os.path.expanduser("~/GlyphMatics/dictionaries")
PHON_DIR = os.path.expanduser("~/GlyphMatics/languages_spoken")

# ====================================================================================
# IPA glyph inversion
# (invert ipa_to_glyphs from multilingual engine)
# ====================================================================================
def invert_glyph_to_ipa():
    """
    Reverse mapping:
      glyph → IPA candidates
    Because IPA hashing is hv = sum(ord(c)) % 111, collisions may occur.
    We build a reverse lookup table from phoneme inventories.
    """
    reverse = defaultdict(list)

    # Load all phoneme inventories
    for fname in os.listdir(PHON_DIR):
        if not fname.endswith(".json"): continue
        data=json.load(open(os.path.join(PHON_DIR,fname),"r",encoding="utf-8"))
        pset=data["phonemes"]

        phonemes=[]
        phonemes += pset.get("consonants",[])
        phonemes += pset.get("vowels",[])
        phonemes += pset.get("diphthongs",[])

        for ipa in phonemes:
            hv=sum(ord(c) for c in ipa)%ALEN
            g=GLYPH_ALPHABET_111[hv]
            reverse[g].append(ipa)

    return reverse

IPA_REVERSE = invert_glyph_to_ipa()

# ====================================================================================
# Reverse G2P (IPA → possible words)
# ====================================================================================
def reverse_g2p(lang, ipa_tokens, dictionary):
    """
    We compare IPA sequences generated from every dictionary word
    with the input IPA tokens.
    The closest match wins.
    """
    candidates = []

    for word, definition in dictionary.items():
        forward = FORWARD_G2P(lang, word)
        if forward == ipa_tokens:
            candidates.append((word, 0.0))  # perfect match
            continue

        # Distance metric (Levenshtein-lite)
        dist = ipa_distance(forward, ipa_tokens)
        candidates.append((word, dist))

    candidates.sort(key=lambda x: x[1])
    return candidates[:5]  # best 5 matches


def ipa_distance(a, b):
    """
    Simple edit-distance for IPA token sequences.
    Lightweight and fast.
    """
    dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]

    for i in range(len(a)+1): dp[i][0]=i
    for j in range(len(b)+1): dp[0][j]=j

    for i in range(1,len(a)+1):
        for j in range(1,len(b)+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j]=min(
                dp[i-1][j]+1,
                dp[i][j-1]+1,
                dp[i-1][j-1]+cost
            )
    return dp[-1][-1]

# ====================================================================================
# SIGIL → DECODE
# ====================================================================================
def decode_supersigil(sigil_json):
    """
    Input: full JSON supersigil object (dict)
    Output:
        - merged glyph stream
        - IPA token candidates per language
        - best word predictions per language
    """

    merged = sigil_json["merged_glyph_stream"]
    glyphs = list(merged)

    # Convert glyphs → IPA candidates
    ipa_candidates=[]
    for g in glyphs:
        if g in IPA_REVERSE:
            ipa_candidates.append(IPA_REVERSE[g])
        else:
            ipa_candidates.append(["?"])

    # We choose first IPA in each slot (deterministic)
    ipa_selected=[cands[0] for cands in ipa_candidates]

    # Now attempt re-identification in each language
    results={}
    for fname in os.listdir(DICT_DIR):
        if not fname.endswith(".json"): continue
        lang=fname[:-5]
        dic=json.load(open(os.path.join(DICT_DIR,fname),"r",encoding="utf-8"))
        predictions = reverse_g2p(lang, ipa_selected, dic)
        results[lang]=predictions

    return {
        "merged_glyph_stream": merged,
        "ipa_selected": ipa_selected,
        "best_words_by_language": results
    }

# ====================================================================================
# CLI
# ====================================================================================
if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description="Reverse Sigil Decoder")
    p.add_argument("supersigil_json")
    args=p.parse_args()

    data=json.load(open(args.supersigil_json,"r",encoding="utf-8"))
    out=decode_supersigil(data)
    print(json.dumps(out, ensure_ascii=False, indent=2))

EOF
