cat > parallel_wordlist_encoder.py << 'EOF'
#!/usr/bin/env python3
# ================================================================================
# GlyphMatics • Parallel Word List Encoder (1000+ words optimized)
# Matthew Blake Ward (Nine1Eight)
# Uses multiprocessing to encode large wordlists into multilingual supersigils.
# ================================================================================

import os, json, multiprocessing as mp, sys, time
from multilingual_sigil_engine import build_global_sigil

# ================================================================================
# Load word list (real format only)
# ================================================================================
def load_word_list(path):
    path = os.path.expanduser(path)

    if not os.path.exists(path):
        raise FileNotFoundError("Word list does not exist: " + path)

    if path.endswith(".json"):
        data = json.load(open(path, "r", encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON list must be a flat list of words.")
        return [str(x).strip() for x in data if str(x).strip()]

    # text file fallback
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if w:
                words.append(w)
    return words

# ================================================================================
# Worker function
# ================================================================================
def worker(word):
    try:
        return build_global_sigil(word)
    except Exception as e:
        return {"word": word, "error": str(e)}

# ================================================================================
# Parallel Processor
# ================================================================================
def process_in_parallel(words, workers=None):
    if workers is None:
        workers = max(2, mp.cpu_count() - 1)

    print(f"[+] Starting parallel encoding with {workers} workers...")
    t0 = time.time()

    with mp.Pool(processes=workers) as pool:
        results = []
        for idx, result in enumerate(pool.imap_unordered(worker, words), 1):
            results.append(result)
            if idx % 50 == 0:
                print(f"[+] {idx}/{len(words)} words processed...")

    t1 = time.time()
    print(f"[✓] Completed {len(words)} words in {t1 - t0:.2f}s")
    return results

# ================================================================================
# Write Results
# ================================================================================
def write_output(results, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[✓] Output written to {outpath}")

# ================================================================================
# Merge all supersigils into one
# ================================================================================
def merge_into_master(results):
    """
    Combine all merged_glyph_streams into a single master supersigil.
    """
    merged = "".join(r.get("merged_glyph_stream","") for r in results if "merged_glyph_stream" in r)
    digest = __import__("hashlib").sha3_256(merged.encode("utf-8")).hexdigest()
    return {
        "count": len(results),
        "sha3_256": digest,
        "master_supersigil": f"⊏⚗${digest[:32]}:{merged}$⊐"
    }

# ================================================================================
# CLI
# ================================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Parallel Multilingual Supersigil Encoder")
    p.add_argument("wordlist", help="Path to 1000-word list (txt or JSON list)")
    p.add_argument("--out", default="batch_supersigils.json", help="Output JSON file")
    p.add_argument("--workers", type=int, default=None, help="CPU workers (default = auto)")
    p.add_argument("--master", action="store_true", help="Also output a master supersigil merging all words")
    args = p.parse_args()

    words = load_word_list(args.wordlist)
    print(f"[+] Loaded {len(words)} words.")

    results = process_in_parallel(words, workers=args.workers)
    write_output(results, args.out)

    if args.master:
        master = merge_into_master(results)
        json.dump(master, open("master_supersigil.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
        print("[✓] Master supersigil saved to master_supersigil.json")
EOF
