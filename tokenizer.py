"""
Tokenizer simples a nível de caractere.

É o ponto de partida mais fácil de entender: cada caractere único do corpus
vira um token. Não há sub-word merging (isso fica para uma evolução futura
com BPE), mas para um corpus pequeno como o TinyShakespeare funciona bem e
deixa o foco no transformer em si, não no tokenizer.
"""
import json
import os


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        obj = cls.__new__(cls)
        obj.stoi = {k: int(v) for k, v in data["stoi"].items()}
        obj.itos = {int(v): k for k, v in obj.stoi.items()}
        obj.vocab_size = len(obj.stoi)
        return obj


if __name__ == "__main__":
    # Teste rápido do tokenizer contra o corpus baixado
    data_path = os.path.join(os.path.dirname(__file__), "data", "input.txt")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer(text)
    print(f"Tamanho do vocabulário: {tok.vocab_size}")
    sample = text[:50]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded)
    print(f"Original: {sample!r}")
    print(f"Codificado: {encoded}")
    print(f"Decodificado bate com original: {decoded == sample}")