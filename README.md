# Projeto C (Ablação HSAMA: MLP Edge)

Este projeto concentra a **Ablação C** desenvolvida ao redor do escopo da arquitetura **HSAMA** (Hierarchical Symbolic Meta-Architecture).

Ao contrário do HSAMA autêntico com suas pontes baseadas no teorema de representação de Kolmogorov-Arnold (via as funções de Padé), o Projeto C reverte essa topologia para o clássico modelo fundacional de perceptrons multicamadas (MLP).

## Natureza da Ablação

A finalidade deste repositório isolado é promover a contra-validação dos benefícios arquiteturais do modelo superior.

- **MLP Edge with Dynamic Bias (`src/models/hsama.py`)**: A interconexão entre as malhas neurais do grafo abandona os _splines_ do KAN e os substitui por uma MLP oculta rudimentar de duas frentes (W1, W2) costuradas com uma ativação em `SiLU`.
- Neste arranjo, o DNA meta-gerado pelas hiper-redes não constrói a forma da ativação, mas atua meramente como um "**Viés Dinâmico**" acrescido a essas transformações lineares.
- Bibliotecas residuais (como `kan.py`) permanecem inativadas na hierarquia de arquivos e seu motor matemático é suprimido na roteirização do `forward`.

### Compatibilidade e Governança (`learning/` e `runtime/`)

Esta ablação foi arquitetada para encaixar suavemente em todo o restante do maquinário base: operando perfeitamente junto a função avaliadora multiescalar unificada do HSAMA e respondendo adequadamente as anomalias de transição de regime comutadas pelo motor de Replay do modelo ativo.
