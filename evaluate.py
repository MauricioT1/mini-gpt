"""Recalcula as losses de treino e validação a partir do checkpoint salvo."""
import torch
from generate import load_model
from train import estimate_loss, DATA_PATH, BATCH_SIZE, BLOCK_SIZE
 
model, tok = load_model("cpu")
text = open(DATA_PATH, encoding="utf-8").read()
data = torch.tensor(tok.encode(text), dtype=torch.long)
n = int(0.9 * len(data))
print(estimate_loss(model, data[:n], data[n:], BLOCK_SIZE, BATCH_SIZE, 50, "cpu"))