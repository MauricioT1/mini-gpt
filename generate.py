"""
Gera texto a partir de um checkpoint treinado.

Uso:
    python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.8 --top_k 50
"""
import argparse
import os
import torch

from model import MiniGPT
from tokenizer import CharTokenizer

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "out")


def load_model(device: str):
    ckpt = torch.load(os.path.join(OUT_DIR, "model.pt"), map_location=device)
    config = ckpt["config"]
    model = MiniGPT(**config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.load(os.path.join(OUT_DIR, "tokenizer.json"))
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    context = torch.tensor(
        [tokenizer.encode(args.prompt)], dtype=torch.long, device=device
    )
    out = model.generate(
        context,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    main()