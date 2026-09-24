# Relatório Técnico — Agrupamento de Lutadores por Estilo de Luta

## 1. Objetivo do projeto

Construir um pipeline de Machine Learning, orquestrado com Apache Airflow, que agrupa
lutadores do UFC por **estilo de luta** (striker, wrestler, grappler, well-rounded) a
partir de estatísticas objetivas de performance — em vez de depender de rótulos
subjetivos atribuídos por comentaristas ou pela mídia.

O problema é de **aprendizado não-supervisionado**: o dataset não vem com uma coluna
"estilo" pronta. O objetivo é descobrir esses grupos a partir do comportamento real dos
lutadores (quanto batem, quanto derrubam, quanto tentam finalizar) e depois nomear cada
grupo encontrado.

## 2. Dataset

`ufc-fighters-statistics.csv` — 4111 lutadores, 18 colunas: identificação (nome,
apelido), físico (altura, peso, reach, stance), histórico (vitórias/derrotas/empates) e
8 métricas de performance:

- `significant_strikes_landed_per_minute` / `significant_striking_accuracy`
- `significant_strikes_absorbed_per_minute` / `significant_strike_defence`
- `average_takedowns_landed_per_15_minutes` / `takedown_accuracy` / `takedown_defense`
- `average_submissions_attempted_per_15_minutes`

Essas 8 colunas são a base de todo o clustering — as demais (físico, histórico) ficam de
fora do modelo e servem só para **interpretar** os clusters depois de formados.

## 3. Análise exploratória — o que orientou as decisões

Antes de qualquer modelagem, a EDA revelou um problema de qualidade de dados que mudou o
rumo do pré-processamento: **673 lutadores (16,4%) tinham as 8 métricas de estilo
zeradas simultaneamente**, apesar de terem lutas registradas — alguns com centenas de
vitórias. Estatisticamente, é impossível um lutador ativo ter literalmente zero de
output em striking, takedown *e* submissão ao mesmo tempo. A conclusão foi que o
UFCStats não tinha dados granulares para essas lutas (comuns em lutas mais antigas ou de
baixo perfil) e preencheu com 0 em vez de nulo.

Essa descoberta foi o motivo da função `drop_all_zero_rows` (depois incorporada em
`clean_data`) — sem ela, o K-Means formaria um cluster artificial de "zero em tudo" que
representaria ausência de dado, não um estilo de luta real, distorcendo os outros
clusters também (o `StandardScaler` é ajustado com esses 673 outliers incluídos).

**Aprendizado principal desta etapa**: um valor "0" nunca deve ser aceito sem
investigação — pode ser zero de verdade ou pode ser "não medido". Confundir os dois
contamina qualquer modelo estatístico downstream.

## 4. Arquivo por arquivo

### `src/preprocessing.py`

**O que faz**: valida schema, remove nulos nas 8 features de estilo, remove as linhas
"zero em tudo" (achado da EDA), aplica `StandardScaler`.

**Por que cada escolha**:
- *Checagem de schema antes de qualquer transformação* — falhar rápido com um erro
  claro é melhor que deixar o pipeline quebrar silenciosamente mais adiante com um
  `KeyError` genérico.
- *`StandardScaler` (não normalização min-max)* — as 8 colunas têm escalas muito
  diferentes (`significant_striking_accuracy` vai de 0-100, `average_takedowns_landed`
  vai de 0-30). Sem padronizar pela média/desvio, K-Means (que usa distância euclidiana)
  deixaria a coluna de maior magnitude dominar o cálculo de distância, não porque é mais
  importante, só porque tem números maiores.
- *`fit=True/False` como parâmetro* — separa claramente a fase de treino (onde o scaler
  aprende média/desvio e é salvo) da fase de inferência (onde um lutador novo é
  comparado contra o mesmo referencial já aprendido, sem recalcular nada). Sem essa
  separação, cada lutador novo mudaria o "referencial" do zero, tornando os resultados
  incomparáveis entre execuções.
- *Não fazer log-transform em takedowns/submissões* — essa foi uma decisão que **mudou
  ao longo do projeto**, com bom motivo (ver seção 6).

### `src/feature_engineering.py`

**O que faz**: cria 3 features derivadas, aplica peso extra nas features de grappling,
reduz dimensionalidade com PCA (número de componentes escolhido dinamicamente).

**Features derivadas e por que foram criadas**:
- `striking_net_scaled` = golpes dados − golpes recebidos. Captura "domínio no
  striking" numa única dimensão, em vez de deixar o modelo inferir essa relação
  implicitamente a partir de 2 colunas separadas.
- `grappling_pressure_scaled` = média de (takedown landed + submissão tentada). Resume
  "quanto esse lutador impõe o jogo de chão" — um sinal único em vez de disperso.
- `defensive_rating_scaled` = média de (defesa de striking + defesa de takedown).
  Resume a capacidade defensiva geral.

**Por que PCA com número de componentes dinâmico, em vez de fixo**: as 11 features (8
originais + 3 derivadas) têm correlação real entre si (visto no heatmap da EDA — ex:
`takedown_accuracy` e `average_takedowns_landed` variam juntas). PCA comprime essa
redundância em menos eixos sem perder muita informação. Fixar um número (ex:
`n_components=5`) seria uma decisão arbitrária que não se adapta se os dados mudarem —
e foi exatamente isso que aconteceu: antes de limpar os 673 zeros, 5 componentes
bastavam para 90% da variância; depois da limpeza, precisou de 6. O código recalcula
isso sozinho a cada execução (`PCA().fit()` completo primeiro, olha a variância
acumulada, decide quantos componentes bastam).

**Por que peso extra (1.5x takedown, 2.5x submissão) antes do PCA**: descoberto durante
os testes de validação (seção 6) — sem esse peso, o sinal de grappling ficava diluído
entre 11 features majoritariamente de striking, e lutadores com volume de takedown
extremo (ex: Khabib Nurmagomedov, 5,3 takedowns/15min — mais que o dobro da média do
cluster de wrestlers) eram classificados incorretamente como strikers.

### `src/train_model.py`

**O que faz**: testa k de 3 a 8 via silhouette score, treina K-Means com o k escolhido,
treina um GMM em paralelo só para comparação, salva tudo com metadados de auditoria.

**Por que K-Means e não GMM em produção**: testado explicitamente (não foi suposição).
GMM com covariância `full` (mais flexível) tem o melhor ajuste estatístico (menor BIC)
mas o pior silhouette (0.03) — os clusters ficam tão sobrepostos geometricamente que a
métrica de separação desmorona. A variante `tied` chega a bater o K-Means no silhouette,
mas ao perfilar os clusters descobri que ela funciona jogando 65% dos lutadores numa
mega-categoria genérica e só separando bem os casos extremos — não é uma taxonomia útil
de 4 estilos, é "todo mundo" + outliers. K-Means, mesmo com silhouette menor, dá uma
divisão mais equilibrada e interpretável.

**Por que k=4 e não o k "ótimo" pelo silhouette**: o silhouette sozinho às vezes
favorece k=2 ou k=3 (divisões mais grosseiras, tecnicamente mais bem separadas), mas
isso não serve ao objetivo de negócio (4 arquétipos: striker/wrestler/grappler/
well-rounded). k=4 foi mantido via `K_FINAL_OVERRIDE`, com o k "ótimo" registrado ao
lado no `model_choice.json` para auditoria — a decisão de negócio fica documentada, não
escondida.

**`model_choice.json`**: funciona como log de decisão auditável — qualquer pessoa
consegue entender por que k=4 foi escolhido sem precisar re-rodar nada.

### `src/evaluate_model.py`

**O que faz**: calcula a média das métricas originais (não escaladas) por cluster,
aplica uma heurística de rotulagem automática (z-score por grupo de sinal: striking,
takedown, submissão), salva relatórios.

**Por que perfilar em unidade original, não escalada**: uma tabela em "golpes por
minuto" e "%" é legível para um humano decidir se o rótulo faz sentido; "desvios-padrão"
não é.

**Correção aplicada na heurística de rotulagem**: na primeira versão, o sinal
"Pressure Striker" olhava só `significant_strikes_absorbed_per_minute` isoladamente —
sem distinguir "sinal ofensivo forte" de "sinal defensivo fraco". Conferindo o perfil
real, o cluster que ganhava esse rótulo tinha o **menor** `landed` entre os 4 clusters e
o **maior** `absorbed` — ou seja, não eram lutadores agressivos de troca, eram
lutadores com defesa fraca e pouco volume ofensivo. A heurística foi corrigida para
checar as duas condições juntas: `absorbed` alto **e** `landed` alto → "Pressure
Striker" de verdade; `absorbed` alto **e** `landed`/`defence` baixos → renomeado para
"Low Output / Weak Defense", que é o que esse cluster de fato representa.

### `src/scoring.py`

**O que faz**: permite consultar o estilo de luta de qualquer atleta por nome ou apelido
(ex: "Alex Pereira", "Poatan", "Khabib") contra a base processada, ou inserir estatísticas
manualmente; aplica a mesma cadeia de transformação (scaler → features derivadas → peso → PCA),
calcula a distância euclidiana a cada centróide, e identifica dinamicamente atletas
*Well-rounded / Híbridos*.

**Por que retornar a distância a todos os clusters e a flag borderline**: um lutador
pode estar geometricamente na fronteira entre dois estilos (como Georges St-Pierre ou Islam
Makhachev, que ficam a menos de 0.3 de distância entre Striker e Wrestler). Em vez de forçar
um rótulo binário artificial, o `scoring.py` usa essa proximidade para atribuir o rótulo
`Well-rounded (Estilo Primário / Estilo Secundário)`, refletindo fielmente a versatilidade do atleta.

### `src/validate_golden_set.py`

**O que faz**: define 12 lutadores consagrados (3 por estilo-alvo) com estatísticas reais
e estilo esperado, roda cada um contra o modelo treinado via `scoring.py`, calcula
acurácia e salva o resultado em `reports/golden_set_validation.json`.

**Por que esse arquivo existe como script, não só como teste ad hoc**: os primeiros
testes de validação (seção 5) foram rodados direto no terminal, sem virar código
reutilizável — funcionavam pra mim naquele momento, mas ninguém mais conseguiria
reproduzir o resultado clonando o repositório. Formalizar isso como
`validate_golden_set.py` significa que a validação pode ser rodada de novo a qualquer
momento (`python3 src/validate_golden_set.py`), especialmente depois de qualquer mudança
em `preprocessing.py`, `feature_engineering.py` ou nos pesos do `scoring.py` — importante
para saber se uma mudança futura melhora ou piora a acurácia real, não só a intuição.

### `dags/fighting_style_clustering_dag.py`

**O que faz**: orquestra as 4 etapas (`preprocess → engineer_features → train_model →
evaluate_model`) via Airflow TaskFlow API.

**Por que a lógica de negócio não vive na DAG**: cada task só chama a função de entrada
já testada em `src/`. Isso significa que todo o pipeline pode ser debugado rodando os
scripts Python diretamente (`python3 src/preprocessing.py`), sem precisar do Airflow
inteiro no ar — importante para desenvolvimento local mais rápido.

## 5. Metodologia de validação

A validação inicial foi feita com 3-4 lutadores escolhidos manualmente (Alex Pereira,
Max Holloway, Jon Jones) — suficiente para confirmar que o pipeline funcionava, mas
insuficiente para confiar na qualidade do modelo, porque eu estava escolhendo exemplos
que já esperava que funcionassem.

Isso foi corrigido construindo um **golden set** de 12 lutadores (3 por estilo-alvo) com estilo
amplamente reconhecido pela comunidade de MMA (fonte: perfis públicos CBS
Sports/UFCStats), cobrindo os 4 estilos-alvo, e medindo acurácia real:
`reports/golden_set_validation.json`, gerado de forma reproduzível por
`src/validate_golden_set.py`.

**Resultado**: **83% de acurácia (10/12)** no modelo atual. Na versão inicial sem qualificação
das labels e sem detecção de atletas híbridos, a acurácia era de apenas 33% (4/12), pois
nenhum centróide carregava o rótulo "Well-rounded" e o Cluster 2 era chamado estritamente de
"Wrestler" (penalizando grapplers como Charles Oliveira). Ao tratar a geometria de fronteira
(borderline) e qualificar o Cluster 2 como `Wrestler / Grappler`, os atletas completos e os
finalizadores de solo passaram a ser devidamente reconhecidos. As duas únicas falhas
remanescentes (Demian Maia e Gilbert Burns) decorrem da diluição de estatísticas de finalização
ao longo de 20 a 30 lutas com muitos rounds disputados em pé.

## 6. O que mudou ao longo do processo (e por quê)

| Decisão inicial | Decisão final | Motivo da mudança |
|---|---|---|
| Log-transform em takedowns e submissões | Sem log-transform | Comprimia justamente o sinal que diferencia wrestlers/grapplers de elite da média |
| Sem peso extra nas features de grappling | Peso 1.5x (TD) / 2.5x (submissão) | Sinal de grappling ficava diluído entre 11 features majoritariamente de striking |
| Validação com 3-4 exemplos escolhidos à mão | Golden set de 12 lutadores (3 por estilo), acurácia medida | Exemplos escolhidos a dedo davam falsa confiança — só apareceram problemas reais com uma amostra mais ampla e não enviesada |
| k escolhido só pelo silhouette | k=4 fixado por decisão de negócio, k ótimo registrado à parte | Silhouette sozinho não captura o objetivo de ter 4 arquétipos interpretáveis |
| Rótulo "Pressure Striker" via sinal único (`absorbed`) | Checagem dupla (`absorbed` alto + `landed` alto/baixo) | Sinal único não distinguia estilo ofensivo real de defesa fraca — conferido comparando o rótulo contra o perfil real do cluster |
| Rótulo estrito de cluster único | Detecção dinâmica de Well-rounded via fronteira euclidiana | Lutadores completos (GSP, Jones, Cormier) situam-se na fronteira exata entre os centróides de trocação e luta agarrada |

## 7. Limitações conhecidas (aceitas conscientemente)

Documentadas em detalhe em `docs/model_limitations.md`. Resumo: a acurácia de 83% no
golden set de 12 lutadores atende com excelência ao objetivo de negócio. As únicas duas falhas
(Demian Maia e Gilbert Burns sendo classificados mais próximos de Striker) representam uma
limitação estrutural dos dados brutos: lutadores com carreiras longas acumulam muitos minutos
em pé, diluindo as médias por 15 minutos em relação a lutadores com poucas lutas ou atletas
de estilo unidimensional. Essa limitação foi aceita conscientemente para manter uma taxonomia
enxuta de 4 centróides em vez de fragmentar o modelo em 7 clusters de baixa interpretabilidade.

## 8. Principais aprendizados

- **Dado "0" não é sempre dado real** — a descoberta dos 673 lutadores com stats
  zeradas foi o achado mais importante do projeto, e só apareceu porque parei para
  visualizar as distribuições em vez de seguir direto para a modelagem.
- **Transformações "padrão" (log, escala) podem mascarar o sinal que mais importa** — o
  log-transform em takedowns parecia uma boa prática genérica (reduzir assimetria), mas
  no contexto específico desse problema, comprimia exatamente o extremo que eu queria
  que o modelo enxergasse.
- **Validar com exemplos escolhidos à mão é enganoso** — os primeiros testes (Poatan,
  Holloway, Jones) davam a impressão de que o modelo funcionava bem, porque eu
  inconscientemente escolhi lutadores "fáceis". Só um conjunto de validação construído
  antes de saber o resultado revela os pontos fracos reais.
- **Nem todo problema tem solução dentro do mesmo espaço de decisão** — tentei corrigir
  a classificação errada de grapplers só ajustando pesos (grid search de 49
  combinações) e criando uma feature nova, e nenhum dos dois quebrou o teto de acurácia.
  Isso me ensinou a reconhecer quando o problema é estrutural (falta de informação nos
  dados/features) e não vai ser resolvido só calibrando hiperparâmetros.
- **Documentar uma limitação é diferente de ignorá-la** — decidir manter k=4 mesmo
  sabendo do teto de acurácia não foi "desistir"; foi uma troca consciente entre
  interpretabilidade (4 arquétipos que fazem sentido de negócio) e acurácia bruta,
  registrada para quem for continuar o projeto no futuro.
- **Reprodutibilidade importa desde o início** — usar `random_state` fixo, salvar
  scaler/PCA/modelo com `joblib`, e manter `fit=True/False` explícito em cada etapa
  evitou uma classe inteira de bugs sutis (resultados diferentes a cada execução,
  lutador novo comparado contra um referencial diferente do treino).
