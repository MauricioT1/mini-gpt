# mini-gpt | Python, PyTorch

Implementação de um transformer decoder-only do zero (atenção multi-head causal, pre-norm, loop de treino e amostragem), com 10.8M de parâmetros, treinado no TinyShakespeare. Análise de overfitting (loss de treino 0.79 vs. validação 1.67) e documentação das limitações do modelo.
github.com/MauricioT1/mini-gpt

## Arquitetura

- Tokenizer a nível de caractere (vocabulário de ~65 símbolos)
- 6 camadas, 6 cabeças de atenção, embedding de 384, contexto de 256 caracteres
- 10.8M de parâmetros
- Otimizador AdamW, 5000 iterações, treinado em CPU

## Como rodar

```bash
pip install -r requirements.txt
python data/download.py
python train.py
python generate.py --prompt "ROMEO:" --tokens 500 --temperature 0.8
```

O treino salva checkpoints a cada 250 iterações e retoma sozinho se for interrompido.

## Resultados

| Split | Loss (nats/char) | Bits por caractere |
|---|---|---|
| Treino | 0.786 | 1.13 |
| Validação | 1.671 | 2.41 |

Exemplo de texto gerado com o prompt `ROMEO:`:

```
ROMEO:
Must I trust infection out
Comes your affairs?

PETER:
Alack, one but disposition.

MERCUTIO:
Nay, gentle Menenius, good Cominius,
I come with him to your voices: he has ever kind of
us.
```

## O que observei

- O modelo aprendeu o formato de roteiro de teatro (nomes em maiúsculas seguidos
  de `:`, falas em linhas separadas), a ortografia e o vocabulário elisabetano.
- O texto não faz sentido no nível da cena: ele mistura personagens de peças
  diferentes, porque o contexto de 256 caracteres é curto demais para "saber"
  em qual peça está.
- A diferença entre loss de treino (0.79) e de validação (1.67) indica
  **overfitting**: com 10.8M de parâmetros para ~1MB de texto, o modelo
  memoriza parte do corpus. Além disso, o split de validação (últimos 10% do
  arquivo) contém peças diferentes das de treino.
- Variar temperatura e top-k teve pouco efeito visível. Com um vocabulário de
  ~65 caracteres, top-k altos quase não filtram nada.

## Possíveis melhorias

- Salvar o checkpoint com a melhor loss de validação (hoje salva sempre o último)
- Mais regularização (dropout, weight decay) e early stopping
- Tokenizer BPE para aumentar o contexto efetivo

## Créditos

Arquitetura inspirada no [nanoGPT](https://github.com/karpathy/nanoGPT) de
Andrej Karpathy, reimplementada para fins de aprendizado.
