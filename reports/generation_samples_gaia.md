# Cenário de execução — perguntas e respostas de exemplo

Modelo: **Gemma 3 Gaia PT-BR 4b it** (`CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it`, 4,3B parâmetros) — pesos de `/home/m320118/apps/git/fiap/16-hugging-face-transformers/models/gaia-qa-embrapa`

Cada pergunta aparece com duas decodificações: **busca em feixe** (`num_beams=2`, `length_penalty=1.0`, determinística e mais fiel ao material técnico) e **amostragem** (`temperature=0.8`, `top_p=0.9`, mais original).

> **Aviso** — as respostas são geradas automaticamente por um modelo de 4,3B parâmetros ajustado sobre a coleção e **podem conter erros técnicos**. Um dos exemplos usa um tema (Café) que não existe na coleção, para mostrar o comportamento fora do domínio de treino.

## 1. Soja — Qual é a melhor época de semeadura da soja?

**Busca em feixe**

> A semeadura deve ser feita quando a temperatura do solo estiver entre 18°C e 25°C.

**Amostragem criativa**

> A primeira semana após o plantio dos primeiros grãos no campo já pode ser considerada como adequada para a implantação das atividades relacionadas ao manejo integrado de pragas (MIP), sendo possível considerar essa data como o início do ciclo da cultura e também como ponto de referência para a condução dessas práticas na área. A segunda recomendação é que o uso das tecnologias agrícolas disponíveis deve começar tão cedo quanto possível, visando às culturas de verão ou milho-forrageira implantadas nos intervalos entre as safras principais e secundárias.

## 2. Soja — Como identificar a ferrugem asiática na lavoura?

**Busca em feixe**

> A identificação da doença pode ser feita por meio da observação da lavoura e da coleta de amostras de folhas para análise laboratorial. Na lavoura, deve-se procurar por folhas com manchas de coloração amarelada, que evoluem para amarelecimento intenso e, posteriormente, para a morte da folha. As folhas afetadas devem ser coletadas e colocadas em sacos de papel, para serem levadas ao laboratório.

**Amostragem criativa**

> A ferrugem-asiática (Pusa marginata) ocorre principalmente nos primeiros 20 dias após o plantio ou até mesmo no momento da semeadura se as sementes não estiverem tratadas. Os sintomas iniciais são pequenos pontos escuros que surgem entre os folíolos novos das folhas mais velhas. As lesões evoluem para manchas irregulares e podem ser confundidas com podridão radicular causada por Pythium spp., fungo que provoca uma doença comum aos cultivos da cultura. Entretanto, essa espécie normalmente ataca apenas os órgãos subterrâneos, enquanto o fungo causador da podridão radial afeta toda a planta. Em condições secas, ocorre deposição de esporos na face inferior das nervuras longitudinais dos folíolos superiores, facilitando sua dispersão pelo vento. Assim, a presença desses espóros deve ser considerada como indício da ocorrência dessa doença. Outro sintoma importante dessa doença está presente em plantas adultas, mas raramente ocorrendo durante a fase inicial do cultivo. Nas últimas semanas antes da floração, aparecem pequenas estruturas semelhantes a chifres, dispostas paralelamente às margens inferiores das folhas afetadas.

## 3. Milho — Qual o espaçamento recomendado para o plantio de milho?

**Busca em feixe**

> O espaçamento recomendado varia de 0,50 m a 0,80 m entre linhas, dependendo da cultivar e do sistema de cultivo.

**Amostragem criativa**

> O espaçamento varia conforme a cultivar (genótipo), as condições climáticas e os sistemas de cultivo utilizados. Para cultivares genômicas (com número reduzido de genes) utilizam-se plantas mais compactas. As variedades híbridas apresentam porte médio ou alto, sendo que os híbridos semialis podem apresentar características intermediárias entre esses dois grupos. Os tipos anões são geralmente utilizados na cultura irrigada, pois não exigem aplicação do sistema de fileiras duplas. Em geral, os espaçamentos recomendados situam-se entre 1 m e 30 cm de distância entre linhas, dependendo das práticas culturais adotadas. A maioria dos híbridos comerciais utiliza uma faixa de 25% de área foliar exposta ao sol, como norma padrão, mas existem situações específicas, sobretudo quando se planta em regiões muito quentes e úmidas durante todo o ciclo, onde essa porcentagem pode variar para até 60%, especialmente no caso do híbrido BRS Prata 789.

## 4. Gado de Leite — Como aumentar a produção de leite no período seco?

**Busca em feixe**

> A produção de leite durante o período seco pode ser aumentada pela utilização de suplementos alimentares, como farelo de soja, farelo de algodão, milho, soro de leite, etc.

**Amostragem criativa**

> É preciso usar os nutrientes mais ricos e de maior digestibilidade (proteínas, energia) na dieta dos animais para que possam fazer melhor uso desses recursos durante o período seco, além de utilizar as forrageiras disponíveis ao máximo, por meio do manejo adequado e bem feito dessas pastagens. Entretanto, não se deve esquecer de manter uma alimentação balanceada entre todos os requisitos nutricionais do animal, especialmente aminoácidos essenciais e vitaminas. Caso contrário, alguns componentes podem ser limitantes para essa espécie. Em condições adversas, como alta precipitação ou chuvas fortes, é muito importante ajustar adequadamente o manejo das pastagens, pois o excesso de água pode levar à perda de produtividade.

## 5. Gado de Leite — O que é mastite e como preveni-la no rebanho?

**Busca em feixe**

> A mastite é uma inflamação da glândula mamária causada por bactérias, fungos ou vírus. A prevenção da mastite é feita por meio da higiene das vacas, da limpeza e desinfecção das ordenhas, da utilização de ordenhadores adequados e da vacinação contra a mastite.

**Amostragem criativa**

> A mastite ocorre quando as glândulas mamárias dos animais são infectadas por bactérias provenientes das mamelas ou pelo trato respiratório superior (urina). A prevenção consiste na eliminação da fonte externa dessa infecção, pela higiene adequada do equipamento utilizado para o ordenhamento, a vacinação contra doenças infecciosas, a escolha criteriosa de bezerros sadios e a manutenção do bem-estar animal durante todo o período reprodutivo.

## 6. Mandioca — Como escolher as manivas para o plantio da mandioca?

**Busca em feixe**

> As manivas devem ser escolhidas com base na qualidade da planta-mãe, na época de colheita e no tipo de solo.

**Amostragem criativa**

> A escolha das manivas depende dos objetivos que se pretende alcançar na exploração. Se for a primeira vez que está cultivando mandiocas, é recomendável escolher variedades de porte baixo ou médio (entre 30 cm e 75 cm). Para cultivo convencional, são mais indicadas plantas de porte médio, com folhas sem espinhos, flores pequenas, pedúnculo verde-amarelado, peso seco entre 120 g e 180 g, raiz principal reta, comprimento total de 60 cm e tamanho final de 45 cm x 39 cm, rendimento comercial de raízes variando de 10 t/ha a 15 t/ha.

## 7. Hortas — Como preparar o solo de uma horta doméstica?

**Busca em feixe**

> O preparo do solo é uma das etapas mais importantes para o sucesso da horta doméstica. Deve ser feito com antecedência de pelo menos 30 dias antes do plantio. O preparo pode ser feito manualmente, com enxada, ou mecanicamente, com trator ou cultivador.

**Amostragem criativa**

> O preparo adequado dos solos para a formação das áreas vegetais requer medidas preventivas, não apenas corretivas. O objetivo dessa ação preventiva é evitar que os danos causados pela compactação, erosão, acidez ou salinidade sejam instalados na área. Os métodos adequados são: Abertura de vala (mais adequada ao cultivo orgânico), Aradoagem mínima ou descompactação manual. A utilização de rolo-faca para nivelamento e eliminação de camadas muito finas de terra solta ou argila, quando se faz o plantio direto. Recomenda-se também a escolha de sementes adaptadas às condições locais, pois essas plantas têm maior potencial de crescimento no campo em comparação aos híbridos comerciais disponíveis comercialmente, bem como sua capacidade de tolerar baixos teores de nutrientes nos primeiros anos após a semeadura. Além disso, devem ser consideradas as espécies de plantas companheiras ou de cobertura vegetal por suas funções específicas e seus benefícios às culturas principais.

## 8. Café — Quando devo irrigar a lavoura em período de estiagem? *(tema ausente da coleção)*

**Busca em feixe**

> A irrigação deve ser iniciada assim que a umidade do solo atingir 10% da capacidade de campo.

**Amostragem criativa**

> A recomendação para o plantio anual da cafeicultura no semiárido brasileiro sugere que os períodos chuvosos sejam suficientes para suprir as necessidades hídricas das plantas durante todo o ano agrícola. Entretanto, na região Centro-Oeste (MG, GO, MS), principalmente nos municípios localizados ao norte dos estados de Minas Gerais ou Goiás, podem ocorrer deficiências hídricas severas em determinadas épocas do ano. Assim, deve ser adotada uma estratégia eficiente de manejo das culturas agrícolas, visando maximizar sua produtividade sob condições climáticas adversas. Entre essas estratégias, destaca-se o controle adequado da erosão, pelo uso racional de água e adubos nitrogenados orgânicos, bem como pela utilização adequada de técnicas culturais como roçada contínua, rotação de culturas, cobertura permanente do solo por palha e correção química do solo, entre outras. Para evitar perdas totais e significativas dessas produções em decorrência de secas prolongadas, tem sido recomendado aos produtores locais aplicar irrigação suplementar quando necessário.

## 9. Banana — Quais os sintomas do mal-do-panamá na bananeira?

**Busca em feixe**

> Os sintomas são: amarelecimento das folhas mais velhas, que murcham e caem. Murcha das folhas mais novas. Morte das plantas. A doença pode ser controlada com fungicidas sistêmicos.

**Amostragem criativa**

> A presença dos esporos no solo ou sobre a superfície das raízes indica infecção pelo patógeno. Os primeiros sintomas ocorrem 2 semanas após o plantio, que consistem na formação de estruturas alongadas, ramificadas (pseudobulbos) ao redor da planta hospedeira. Em seguida ocorre amarelecimento intenso das folhas mais velhas até a morte completa. Se as lesões ocorrerem próximo aos frutos (fichas), estes podem ser muito afetados e perder qualidade comercial. Na fase final do desenvolvimento, a doença pode provocar desfolha total da planta. A frutificação deve ser realizada quando ainda houver folhagem normal presente, para evitar perda excessiva de produtividade. O controle é feito principalmente por meio de produtos químicos, sendo proibido usar fungicidas sistêmicos nos bananalis devido à possibilidade de contaminaçãolo de outros cultivos comerciais de interesse econômico.

## 10. Sementes — Como saber se uma semente tem dormência?

**Busca em feixe**

> A dormência pode ser detectada por meio de testes de germinação em diferentes substratos e em diferentes condições de temperatura e fotoperíodo.

**Amostragem criativa**

> Para determinar a presença ou não de dormência na semente, basta que o teste de germinação apresente resultados inferiores ao esperado para essa espécie no período observado. Por exemplo, se o ideal seria 80% de germinação nos primeiros cinco dias após a semeadura (DAS), mas apenas 30% foi registrado nesse mesmo tempo, então pode-se afirmar que existe dormência ativa pela região de armazenamento e distribuição da semente. Além disso, deve-se verificar se as sementes apresentam alguma reação fisiológica negativa quando expostas às condições desfavoráveis antes da semeadura, como baixos teores de umidade ou temperatura inadequada. As espécies que produzem frutos abertos são mais suscetíveis à ocorrência dessa reação. Se for constatado o problema de dormência, o agricultor deverá utilizar métodos adequados para superá-la e obter os melhores rendimentos possíveis de sua cultura.
