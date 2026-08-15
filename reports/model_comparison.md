# Comparação entre os modelos ajustados

Relatório gerado por `poetry run qa-embrapa compare`. Os números vêm dos relatórios de avaliação de cada modelo, calculados sobre **as mesmas perguntas do mesmo conjunto de teste**.

- Conjunto de teste: **200 perguntas** não vistas no treino.
- Modelos comparados: PTT5 v2 base, Gemma 3 Gaia PT-BR 4b it.

O relatório traz **duas leituras**, que respondem a perguntas diferentes e podem apontar para modelos diferentes:

1. **Decodificação idêntica** (`beam_search` nos dois) — isola a diferença entre os modelos, mantendo a decodificação como variável controlada. É a base das seções *Qualidade*, *Originalidade* e *Custo*.
2. **Melhor configuração de cada um** — é o que um operador obteria na prática, já que a decodificação é escolhida junto com o modelo. É a base da **recomendação**.

## Recomendação

**Gemma 3 Gaia PT-BR 4b it** — maior índice de qualidade agregado na melhor configuração de cada um (19.98 em `greedy` contra 18.42 de PTT5 v2 base em `amostragem_criativa`, +1.56), ao custo de 25× mais tempo por resposta.

Critério: índice de qualidade agregado na melhor decodificação de cada modelo, com o custo de inferência como desempate.

> **A comparação depende da decodificação.** Os vencedores por decodificação não são unânimes (`amostragem_criativa` → gaia, `beam_search` → ptt5, `beam_search_curto` → gaia, `greedy` → gaia). É por isso que o critério compara cada modelo na sua melhor configuração, e não em uma decodificação eleita de antemão — que produziria a conclusão que se escolhesse.

## Os modelos

|  | PTT5 v2 base | Gemma 3 Gaia PT-BR 4b it |
| --- | --- | --- |
| Checkpoint | `unicamp-dl/ptt5-v2-base` | `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it` |
| Arquitetura | seq2seq | causal |
| Parâmetros | 223M | 4,3B |
| Estratégia de ajuste | ajuste-completo | lora-4bit |
| Parâmetros treinados | 222.903.552 | 29.802.496 |
| Épocas | 5 | 1 |
| Tempo de treino | 33 min | 261 min |

## Qualidade — decodificação idêntica (`beam_search`)

Aderência à resposta original da Embrapa. Todas as métricas vão de 0 a 100 e **quanto maior, melhor**.

> Estes números são da decodificação de referência, a mesma nos dois modelos. Não são a base da recomendação — para isso, ver *Qualidade por decodificação*, mais abaixo.

| Métrica | PTT5 v2 base | Gemma 3 Gaia PT-BR 4b it |
| --- | --- | --- |
| **Índice agregado** | **18.15** | **17.81** |
| rougeL | 18.73 | 17.68 |
| rouge1 | 25.68 | 25.09 |
| rouge2 | 8.36 | 7.12 |
| bleu | 1.43 | 1.54 |
| chrf | 17.05 | 18.24 |

Pesos do índice agregado: rougeL 35%, rouge1 25%, chrf 25%, rouge2 10%, bleu 5%.

## Originalidade — decodificação idêntica (`beam_search`)

O conteúdo é novo ou é cópia do material de treino?

| Métrica | PTT5 v2 base | Gemma 3 Gaia PT-BR 4b it |
| --- | --- | --- |
| 4-gramas inéditos (↑ mais original) | 72.6% | 74.6% |
| Maior trecho copiado (↓ melhor) | 5.3 palavras | 5.3 palavras |
| distinct-2 (↑ mais diverso) | 0.613 | 0.717 |
| Repetição interna (↓ melhor) | 0.003 | 0.000 |

## Custo e formato — decodificação idêntica (`beam_search`)

| Métrica | PTT5 v2 base | Gemma 3 Gaia PT-BR 4b it |
| --- | --- | --- |
| Segundos por resposta | 0.22 s | 5.32 s |
| Palavras por resposta | 30.9 | 32.3 |
| Respostas vazias | 0 | 0 |
| Perda de validação *(não comparável entre modelos)* | 2.705 | 1.884 |

> A perda de validação aparece apenas como registro interno de cada modelo. Ela **não** é comparável entre os dois: cada um a calcula sobre o próprio vocabulário e a própria segmentação do texto.

## Qualidade por decodificação

Índice agregado em cada estratégia, com a mesma decodificação aplicada aos dois modelos. A última coluna mostra onde cada um leva vantagem — é esta tabela que revela que o resultado não é uniforme.

| Decodificação | PTT5 v2 base | Gemma 3 Gaia PT-BR 4b it | Melhor |
| --- | --- | --- | --- |
| `amostragem_criativa` | 18.42 | 19.66 | Gemma 3 Gaia PT-BR 4b it (+1.24) |
| `beam_search` | 18.15 | 17.81 | PTT5 v2 base (+0.34) |
| `beam_search_curto` | 16.74 | 16.75 | Gemma 3 Gaia PT-BR 4b it (+0.01) |
| `greedy` | 17.06 | 19.98 | Gemma 3 Gaia PT-BR 4b it (+2.92) |

Melhor configuração de cada modelo — a base da recomendação:

| Modelo | Decodificação | Índice | Palavras | Segundos |
| --- | --- | --- | --- | --- |
| PTT5 v2 base | `amostragem_criativa` | **18.42** | 46.5 | 0.23 s |
| Gemma 3 Gaia PT-BR 4b it | `greedy` | **19.98** | 58.5 | 5.72 s |

## Respostas lado a lado

As mesmas perguntas do conjunto de teste, na decodificação de referência.

### 1. Arroz — Qual é a taxa de perda de grãos nas colheitas manual e mecanizada do arroz de terras altas?

**Resposta original da Embrapa**

> Levantamentos realizados pela Embrapa Arroz e Feijão demonstram que as perdas de grãos na colheita são elevadas e influenciadas por diversos fatores relacionados principalmente ao método e ao momento de realização da colheita, à qualidade da operação das colhedoras e à cultivar. Num estudo realizado em diversas lavouras de arroz, constatou-se perda média de 186 kg/ha na colheita manual e de 326 kg/ha -1 na mecanizada.

**PTT5 v2 base**

> A taxa de perda de grãos nas colheitas manual e mecanizada do arroz de terras altas varia de acordo com o tipo de solo, a época de plantio e a cultivar.

**Gemma 3 Gaia PT-BR 4b it**

> Nas colheitas manuais, as perdas de grãos variam de 10% a 15%. Nas colheitas mecanizadas, a perda é de 2% a 4%.

### 2. Banana — Como são confeccionadas as iscas tipo queijo?

**Resposta original da Embrapa**

> As iscas tipo queijo são preparadas cortando-se o pseudocaule a aproximadamente 30 cm do solo e, em seguida, efetuando-se um novo corte, parcial ou total, à metade dessa altura, de modo que a parte superior fique ligada à porção inferior por meio da bainha (corte parcial) ou simplesmente sobre a porção inferior (corte total).

**PTT5 v2 base**

> As iscas tipo queijo são confeccionadas a partir de uma mistura de farinha de mandioca com farinha de milho e farinha de mandioca. A mistura de farinha com farinha de mandioca e farinha de milho é feita a partir de um copo de água destilada em temperatura de 20 o C.

**Gemma 3 Gaia PT-BR 4b it**

> As iscas do tipo queijo são preparadas a partir de pedaços de casca de banana, que são cortados em pequenos cubos, colocados em bandejas e cobertos com uma mistura de terra e esterco de gado. As iscas são colocadas em canteiros e deixadas para secar ao sol. Depois de secas, as iscas são cobertas com uma camada de terra e adubo orgânico.

### 3. Sistema Plantio Direto — Qual a quantidade de sementes e qual a forma de semeadura de aveia e milheto mais adequada ao SPD?

**Resposta original da Embrapa**

> A quantidade de sementes a utilizar é variável em função do percentual de germinação das sementes, do método de semeadura, da finalidade do cultivo e das condições climáticas. De modo geral, a semeadura em linha é a mais adequada, mas em alguns casos específicos, a semeadura a lanço pode ser utilizada, além da sobressemeadura na fase de maturação da soja. Para a aveia-preta, recomenda-se utilizar em torno de 60 a 80 kg/ha e para a aveia-branca, de 70 a 90 kg/ha. Na semeadura do milheto, utilizam-se de 15 a 20 kg/ha. Na semeadura a lanço, deve-se fazer um acréscimo de 20% na quantidade de sementes.

**PTT5 v2 base**

> A aveia e o milheto devem ser semeados em sacos de plástico ou em sacos plásticos. A aveia pode ser semeada em sacolas plásticas ou em sacola plástica.

**Gemma 3 Gaia PT-BR 4b it**

> A aveia e o milheto podem ser semeados em sulcos ou em covas, com espaçamento de 15 cm entre fileiras e de 10 cm entre plantas dentro da fileira. As sementes devem ser distribuídas uniformemente no sulco ou na cova, a uma profundidade de 2 cm.

