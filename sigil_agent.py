cat > ~/sigil_agent.py <<'EOF'
#!/usr/bin/env python3
# ============================================================================
# Supersigil Neural Agent v2.0
# Adds:
#   • Multi-step recurrent thought loop (real)
#   • Full interactive CLI chat
# Matthew Blake Ward (Nine1Eight)
# ============================================================================

import os, json, hashlib, sys
import numpy as np
from pathlib import Path
import importlib.util

HOME = Path.home()
REHYD = HOME / "rehydrated_system"

# ============================================================================
# Utility: deterministic SHA3 vector embedding
# ============================================================================

def sha3_vec(text: str, dims=64):
    h = hashlib.sha3_512(text.encode()).hexdigest()
    arr = np.array([int(h[i:i+2], 16) for i in range(0, len(h), 2)], float)
    arr = arr[:dims]
    return arr / np.linalg.norm(arr)


def load_modules_as_vectors():
    """
    Convert every restored module into a neural vector.
    """
    vectors = {}
    for f in REHYD.iterdir():
        if f.suffix == ".py":
            code = f.read_text()
            vectors[f.name] = sha3_vec(code)
    return vectors


# ============================================================================
# Load dynamic modules (required for SigilNode)
# ============================================================================

def load_mod(path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


glyph_graph = load_mod(REHYD/"glyph_graph_engine.py")
SigilNode = glyph_graph.SigilNode


# ============================================================================
# Neural Graph Construction
# ============================================================================

def build_neural_graph(module_vectors):
    nodes = {}

    # create nodes
    for name, vec in module_vectors.items():
        nodes[name] = SigilNode(name=name, vector=vec)

    # fully connected weighted network based on cosine similarity
    for a in nodes.values():
        for b in nodes.values():
            if a is b:
                continue
            w = float(np.dot(a.vector, b.vector))
            a.connect(b, weight=w)

    return nodes


# ============================================================================
# RECURSIVE THOUGHT LOOP (B)
# ============================================================================

def recurrent_thought_cycle(nodes, prompt_vec, steps=5):
    """
    A real iterative forward pass:
        1. injection of input vector
        2. propagation through neural graph
        3. normalization
        4. repeat
    """
    # Inject prompt into every node equally for cycle 1
    for node in nodes.values():
        node.vector = (node.vector + prompt_vec)
        node.vector = node.vector / np.linalg.norm(node.vector)

    # Actual recurrent propagation
    for _ in range(steps):
        new_vectors = {}
        for name, node in nodes.items():
            v = node.propagate()
            v = v / np.linalg.norm(v)
            new_vectors[name] = v

        # update node vectors
        for name, v in new_vectors.items():
            nodes[name].vector = v

    # return final activations
    return {name: n.vector for name, n in nodes.items()}


# ============================================================================
# AGENT CLASS
# ============================================================================

class SupersigilAgent:
    def __init__(self):
        print("[*] Loading supersigil neural network...")
        self.module_vectors = load_modules_as_vectors()
        self.graph = build_neural_graph(self.module_vectors)
        print(f"[*] Neural nodes loaded: {len(self.graph)}")

    def think(self, user_text: str, loops=5):
        print("[*] Entering thought loop...")

        prompt_vec = sha3_vec(user_text)

        activations = recurrent_thought_cycle(
            self.graph,
            prompt_vec,
            steps=loops
        )

        # pick highest-activation module by vector norm
        best = max(
            activations.items(),
            key=lambda x: np.linalg.norm(x[1])
        )

        best_name, best_vec = best

        return {
            "module": best_name,
            "activation_vector": best_vec
        }


# ============================================================================
# CLI CHAT INTERFACE
# ============================================================================

def chat_loop():
    agent = SupersigilAgent()

    print("\n============================================")
    print("  SUPERSIGIL NEURAL CHAT")
    print("  Type 'exit', 'quit', or CTRL+C to leave.")
    print("============================================\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\n[*] Exiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("[*] Goodbye.")
            break

        result = agent.think(user_input, loops=5)

        activated_module = result["module"]
        act_vec = result["activation_vector"]

        print("\nAgent:")
        print(f"  Activated Module: {activated_module}")
        print(f"  Latent Vector: {act_vec[:8]} ... (len={len(act_vec)})\n")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    if len(sys.argv) == 1:
        return chat_loop()

    # CLI single-run mode
    text = " ".join(sys.argv[1:])
    agent = SupersigilAgent()
    r = agent.think(text)
    print(r)

if __name__ == "__main__":
    main()
EOF
