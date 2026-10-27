"""
Loop de treinamento do mini-GPT no TinyShakespeare.

Uso:
    python data/download.py   # baixa o corpus (uma vez só)
    python train.py           # treina e salva checkpoint em out/model.pt
"""
import os
import time
import torch

from model import MiniGPT
from tokenizer import CharTokenizer

# ---- Hiperparâmetros (ajuste conforme sua GPU/CPU) ----
BATCH_SIZE = 64
BLOCK_SIZE = 256       # tamanho do contexto (quantos chars o modelo vê por vez)
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_ITERS = 5000
EVAL_INTERVAL = 250
EVAL_ITERS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "input.txt")
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "out")


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device):
    model.eval()
    out = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    print(f"Usando device: {DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Corpus não encontrado em {DATA_PATH}. Rode primeiro: python data/download.py"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    tokenizer.save(os.path.join(OUT_DIR, "tokenizer.json"))

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Retoma de um checkpoint existente, se houver (útil se o treino for
    # interrompido no meio, ex: Replit dormindo por inatividade)
    start_iter = 0
    ckpt_path = os.path.join(OUT_DIR, "model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        start_iter = ckpt.get("iter", 0) + 1
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Checkpoint encontrado, retomando da iteração {start_iter}")

    t0 = time.time()
    for it in range(start_iter, MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(
                model, train_data, val_data, BLOCK_SIZE, BATCH_SIZE, EVAL_ITERS, DEVICE
            )
            elapsed = time.time() - t0
            print(
                f"iter {it:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | {elapsed:.1f}s"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "iter": it,
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "block_size": BLOCK_SIZE,
                        "n_embd": N_EMBD,
                        "n_head": N_HEAD,
                        "n_layer": N_LAYER,
                        "dropout": DROPOUT,
                    },
                },
                os.path.join(OUT_DIR, "model.pt"),
            )

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Treino concluído. Checkpoint salvo em", os.path.join(OUT_DIR, "model.pt"))


if __name__ == "__main__":
    main()