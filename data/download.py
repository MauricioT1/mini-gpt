"""
Baixa o dataset TinyShakespeare (~1MB) usado para treinar o mini-GPT.
Fonte: char-rnn dataset original de Andrej Karpathy.
"""
import os
import urllib.request

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
OUT_PATH = os.path.join(os.path.dirname(__file__), "input.txt")


def download():
    if os.path.exists(OUT_PATH):
        print(f"Já existe em {OUT_PATH}, pulando download.")
        return
    print(f"Baixando de {URL} ...")
    urllib.request.urlretrieve(URL, OUT_PATH)
    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Salvo em {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    download()