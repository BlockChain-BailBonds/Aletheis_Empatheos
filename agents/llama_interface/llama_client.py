#!/usr/bin/env python3
import requests, json, sys
URL = "http://127.0.0.1:8080/completion"

def query(prompt, n_predict=128, temperature=0.4):
    payload = {"prompt": prompt, "n_predict": n_predict, "temperature": temperature}
    r = requests.post(URL, json=payload, timeout=600)
    if r.ok:
        try:
            return r.json().get("content", "").strip()
        except:
            return r.text.strip()
    else:
        return f"[HTTP {r.status_code}] {r.text}"

if __name__ == "__main__":
    prompt = " ".join(sys.argv[1:]) or "Hello"
    print(query(prompt))
