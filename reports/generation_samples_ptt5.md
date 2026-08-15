# Cenário de execução — perguntas e respostas de exemplo

Modelo: **PTT5 v2 base** (`unicamp-dl/ptt5-v2-base`, 223M parâmetros) — pesos de `/home/m320118/apps/git/fiap/16-hugging-face-transformers/models/ptt5-qa-embrapa`

Cada pergunta aparece com duas decodificações: **busca em feixe** (`num_beams=4`, `length_penalty=1.3`, determinística e mais fiel ao material técnico) e **amostragem** (`temperature=0.8`, `top_p=0.9`, mais original).

> **Aviso** — as respostas são geradas automaticamente por um modelo de 223M parâmetros ajustado sobre a coleção e **podem conter erros técnicos**. Um dos exemplos usa um tema (Café) que não existe na coleção, para mostrar o comportamento fora do domínio de treino.

## 1. Soja — Qual é a melhor época de semeadura da soja?

**Busca em feixe**

> A melhor época de semeadura da soja é a que antecede o início da estação chuvosa.

**Amostragem criativa**

> A semeadura da soja deve ser feita de 8 a 10 dias antes do plantio.

## 2. Soja — Como identificar a ferrugem asiática na lavoura?

**Busca em feixe**

> A ferrugem asiática pode ser identificada por meio do exame físico da planta, que consiste na identificação de um ou mais microrganismos presentes na superfície da planta. O exame físico é feito com o auxílio de um microscópio eletrônico.

**Amostragem criativa**

> Os sintomas são visíveis na superfície da planta. O aspecto mais característico da ferrugem asiática é a presença de folhas vermelhas e manchas amarelas na superfície, em manchas que parecem ter uma coloração amarelada. A coloração branca da planta indica o aparecimento de um estriado, que poderá ser identificado pelo número de flores e do número de hastes. A forma mais comum da doença é o desfolhamento do fungo, o que pode ocasionar a morte da planta.

## 3. Milho — Qual o espaçamento recomendado para o plantio de milho?

**Busca em feixe**

> O espaçamento recomendado para o plantio de milho varia de acordo com o tipo de solo, a época de semeadura e a cultura.

**Amostragem criativa**

> O espaçamento recomendado para o plantio de milho depende de alguns fatores, tais como: Adubação adequada do solo, de cultivares com boa fertilidade, de áreas com boa fertilidade e que apresentem resistência a doenças, como a lagarta-do-milho e a cigarrinha-da-caule. Adubação de sementes, quando bem planejada, deve apresentar espaçados adequados para o desenvolvimento da cultura.

## 4. Gado de Leite — Como aumentar a produção de leite no período seco?

**Busca em feixe**

> A produção de leite no período seco pode ser aumentada por meio do uso de adubos nitrogenados e/ou adubos orgânicos. O adubo nitrogenado pode ser utilizado para aumentar a produção de leite durante o período seco.

**Amostragem criativa**

> Os efeitos do manejo do rebanho durante a seca são maiores quando se observam maiores temperaturas, principalmente durante o período seco. A produção de leite aumenta no período seco quando se percebe menor temperatura. A produção pode ser maior durante o período úmido se a temperatura for favorável (menos de 5 o C). No entanto, o manejo do rebanho, que pode variar de acordo com a condição do clima, também pode ser aumentado com o manejo do sistema de rotação de culturas e o uso de fertilizantes. O uso de fertilizante químico também ajuda a aumentar a produção de leite durante o período chuvoso.

## 5. Gado de Leite — O que é mastite e como preveni-la no rebanho?

**Busca em feixe**

> A mastite é uma doença causada por fungos, que causam a morte de animais. O tratamento da mastite pode ser feito com o uso de vacinas e/ou preventivos. A vacinação deve ser feita no início do período de lactação para evitar a ocorrência da doença.

**Amostragem criativa**

> A mastite é uma condição muito comum no rebanho bovino, causada pelo estresse térmico (acidez), o alongamento das lesões e a falta de apetite, o que pode ser prevenido com o uso de medicamentos.

## 6. Mandioca — Como escolher as manivas para o plantio da mandioca?

**Busca em feixe**

> A escolha das manivas para o plantio da mandioca deve ser feita de acordo com as características do solo, do clima e da época de semeadura.

**Amostragem criativa**

> As manivas devem ser produzidas em altura, peso e comprimento de 50 cm a 80 cm, de modo que possam ser usadas em qualquer época do ano.

## 7. Hortas — Como preparar o solo de uma horta doméstica?

**Busca em feixe**

> As hortas domésticas devem ser preparadas com adubos orgânicos e, preferencialmente, com fertilizantes orgânicos. O preparo do solo deve ser feito com o auxílio de máquinas e equipamentos adequados.

**Amostragem criativa**

> O preparo do solo é realizado com as seguintes medidas: Colheita de sementes e de raízes, com rotação das culturas. Retirada da terra das sementes. Uso de um pano úmido ou um lençol úmido, limpo e ventilado. Para a cobertura do solo, deve-se: preparar a terra com uma espátula de papel. Aplicar palhada e colher as folhas em seguida. Colocar o adubo de forma uniforme e espalhar por toda a superfície do solo. Lavar as sementes com um pano úmido, em local coberto com papel umedecido. Envolver o solo com o auxílio de uma faca. Em seguida, aparar as plantas, colocar a massa de papel, colocar o adubos e, posteriormente, realizar a secagem e a limpeza do solo. Após a compactação, o adubar e desfazimento do solo e lavar a parte superficial e a camada superficial, misturar os grãos e fazer o adubedor. Para que o solo fique arejado e limpo.

## 8. Café — Quando devo irrigar a lavoura em período de estiagem? *(tema ausente da coleção)*

**Busca em feixe**

> A irrigação deve ser feita sempre que houver necessidade de água para a lavoura. Em períodos de estiagem, a irrigação deve ser realizada quando a temperatura estiver mais amena.

**Amostragem criativa**

> A irrigação do café ocorre durante a época seca, quando o solo já se encontra em condições de estiagem, e a água, a partir de sua abertura, deve ser escoada para ser aplicada à lavoura.

## 9. Banana — Quais os sintomas do mal-do-panamá na bananeira?

**Busca em feixe**

> Os sintomas do mal-do-panamá na bananeira são os mesmos observados em outras espécies de bananeira.

**Amostragem criativa**

> O mal-do-panamá é uma doença de difícil controle, causada por fungos de diferentes partes do corpo, que atacam as folhas e a bananeira. As plantas infectadas apresentam uma estrutura desprezível, com folhas com manchas brancas e sem cor, de coloração azulada, que se espalham pelo tronco, com uma crosta marrom, de cor amarelada, e, no tronco, crescem de forma amarelada, formando uma camada de tecido que penetra na superfície da bananeira, formando um pequeno estigma, que não apresenta cor verde-clara ou marrom-escuro, o que dificulta a entrada do fungo.

## 10. Sementes — Como saber se uma semente tem dormência?

**Busca em feixe**

> A dormência de uma semente pode ser determinada por meio de um teste de germinação, que consiste na coleta de amostras de sementes para análise. O teste é feito a partir das amostras de sementes coletadas e analisadas.

**Amostragem criativa**

> As sementes germinam e crescem a uma dormência.
