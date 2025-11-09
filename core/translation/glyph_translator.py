#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
glyph_translator.py
Connects to local llama.cpp (GPT-OSS-20B) and encodes all output
as Codex glyph strings from Codex.glyphstore.txt.
"""

import json
import requests
from pathlib import Path


class GlyphTranslator:
    def __init__(
        self,
        glyphstore_path: str,
        host: str = "127.0.0.1",
        port: int = 8080,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
    ):
        self.glyphstore_path = Path(glyphstore_path)
        self.base_url = f"http://{host}:{port}"
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self._init_glyph_alphabet()

    # ------------------------------------------------------------
    # 1. glyph alphabet and codec
    # ------------------------------------------------------------
    def _init_glyph_alphabet(self):
        text = self.glyphstore_path.read_text(encoding="utf-8")
        seen = {}
        for ch in text:
            if not ch.isspace() and ch not in seen:
                seen[ch] = len(seen)
        if not seen:
            raise RuntimeError("No glyphs found in glyphstore file.")
        self.alphabet = list(seen.keys())
        self.base = len(self.alphabet)
        self.digit_to_glyph = {i: g for i, g in enumerate(self.alphabet)}
        self.glyph_to_digit = {g: i for i, g in enumerate(self.alphabet)}

    def text_to_glyphs(self, text: str) -> str:
        out = []
        for b in text.encode("utf-8"):
            hi = b // self.base
            lo = b % self.base
            out.append(self.digit_to_glyph[hi])
            out.append(self.digit_to_glyph[lo])
        return "".join(out)

    def glyphs_to_text(self, glyph_string: str) -> str:
        digits = [self.glyph_to_digit.get(g, 0) for g in glyph_string if g in self.glyph_to_digit]
        buf = bytearray()
        for i in range(0, len(digits) - 1, 2):
            val = digits[i] * self.base + digits[i + 1]
            buf.append(min(val, 255))
        return buf.decode("utf-8", errors="replace")

    # ------------------------------------------------------------
    # 2. local LLM completion
    # ------------------------------------------------------------
    def _llm_complete(self, prompt: str) -> str:
        url = f"{self.base_url}/completion"
        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": [],
            "stream": False,
        }
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data.get("content"), str):
            return data["content"]
        elif "completion" in data:
            return data["completion"]
        elif isinstance(data.get("content"), list):
            return "".join(part.get("text", "") for part in data["content"])
        return json.dumps(data, ensure_ascii=False)

    # ------------------------------------------------------------
    # 3. unified interface
    # ------------------------------------------------------------
    def translate_to_glyph(self, prompt: str) -> str:
        """
        Runs the local model and returns glyph-encoded output.
        """
        text_output = self._llm_complete(prompt)
        glyph_output = self.text_to_glyphs(text_output)
        return glyph_output


# ------------------------------------------------------------
# Demo
# ------------------------------------------------------------
if __name__ == "__main__":
    CODEx = "/storage/emulated/0/Download/unified_ledger.glyphstore"
    translator = GlyphTranslator(CODEx)

    prompt = "Translate 'The mind speaks through symbols.' into French."
    glyph_str = translator.translate_to_glyph(prompt)

    print("\nGlyph-encoded output:\n", glyph_str)
    print("\nDecoded back:\n", translator.glyphs_to_text(glyph_str))
