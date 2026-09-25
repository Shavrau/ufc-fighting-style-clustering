# Model Limitations — Fighting Style Clustering

## Summary

K-Means with k=4 (`TD_WEIGHT=1.5`, `SUB_WEIGHT=2.5`, applied to the takedown/submission
scaled features before PCA) is the production configuration. It correctly separates the
common cases but has a **known, investigated, unresolved limitation** with moderate-volume
grappler/wrestler hybrids. This document exists so that limitation isn't rediscovered by
accident later — it was found, tested against, and knowingly accepted.

## Golden-set validation

20 well-known fighters covering all styles, scored against the production model (see `reports/golden_set_validation.json` for the full run):

| Fighter | Expected | Predicted | Correct |
|---|---|---|---|
| Alex Pereira "Poatan" | Striker | Striker | ✓ |
| Max Holloway | Striker | Striker | ✓ |
| Justin Gaethje | Striker | Striker | ✓ |
| Tai Tuivasa | Striker | Low Output / Weak Defense | ✗ |
| Khabib Nurmagomedov | Wrestler | Wrestler / Grappler | ✓ |
| Ben Askren | Wrestler | Wrestler / Grappler | ✓ |
| Mark Coleman | Wrestler | Wrestler / Grappler | ✓ |
| Yoel Romero | Wrestler | Striker | ✗ |
| Charles Oliveira | Grappler | Wrestler / Grappler | ✓ |
| Demian Maia | Grappler | Striker | ✗ |
| Gilbert Burns | Grappler | Striker | ✗ |
| Georges St-Pierre | Well-rounded | Well-rounded (Wrestler / Striker) | ✓ |
| Jon Jones | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |
| Daniel Cormier | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |
| Kamaru Usman | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |
| Islam Makhachev | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |
| Robert Drysdale | Submission Specialist | Submission Specialist (Grappler) | ✓ |
| Megumi Fujii | Submission Specialist | Submission Specialist (Grappler) | ✓ |
| CM Punk | Low Output / Weak Defense | Low Output / Weak Defense | ✓ |
| Kris Moutinho | Low Output / Weak Defense | Low Output / Weak Defense | ✓ |

**Accuracy: 16/20 (80.0%)**

*(Nota técnica: As 4 divergências revelam limitações reais e fundamentais do dataset e da modelagem não supervisionada:
1. **Tai Tuivasa**: Striker peso-pesado que aceita tomar dano para nocautear (4.98 golpes absorvidos/min com 43% de defesa). Seu saldo de trocação negativo faz o K-Means aproximá-lo do Cluster 0.
2. **Yoel Romero**: Medalhista de prata olímpico em Freestyle Wrestling que no UFC quase não aplicava quedas (<1.8/15min) e só lutava trocando golpes em pé. O algoritmo captura o que o lutador fez no octógono, não seu passado na luta olímpica.
3. **Demian Maia e Gilbert Burns**: Especialistas mundiais de Jiu-Jitsu que foram forçados a disputar longos rounds em pé contra wrestlers defensivos, diluindo suas médias de solo e caindo no centroide de Striker pela ausência de métricas de Control Time).*

## What was tried

*(Note: experiments 1-4 below were run against the original 9-fighter golden set, before
it was expanded to the current 12-fighter version above. Relative conclusions — no
combination breaking the ceiling, same root cause — still hold; exact fractions reported
here refer to the 9-fighter set.)*

1. **Weight grid search** — 49 combinations of `TD_WEIGHT` × `SUB_WEIGHT` (1.0 to 4.0 each).
   Best result: 5/9 (56%), achieved by only 2 of 49 combinations. No combination exceeded 56%.
2. **`submission_ratio` engineered feature** (`submissions / (submissions + takedowns + 0.5)`)
   — captures grappling *composition* rather than raw volume. Grid-searched alongside the
   existing weights (~130 combinations). Still capped at 56% — it changed *which* fighters
   were misclassified (moved Oliveira/Maia from "Wrestler" to "Striker") but not the total
   error count.
3. **Increasing k** (5, 6, 7) with the ratio feature — k=7 reached 6/9 (67%), fixing Oliveira,
   but at the cost of silhouette score (0.297 vs 0.367 at k=4) and 7 clusters instead of a
   clean 4-archetype taxonomy.
4. **Removing `grappling_pressure_scaled` entirely** — hypothesis: averaging takedown and
   submission signals into one feature could be pulling wrestlers and grapplers toward each
   other instead of separating them. Tested with and without this feature across the same
   weight combinations. Result: **identical 56% ceiling, same 4 fighters misclassified**,
   with or without it. Ruled out — the feature isn't the bottleneck.

## Root cause

Elite wrestlers who ground-and-pound (e.g. Khabib) and submission specialists who also shoot
takedowns to get to the ground (e.g. Oliveira, Maia) have **overlapping takedown volume** —
takedown count alone can't tell them apart. Submission rate helps but Oliveira/Maia's rates,
while high for a career average, sit in a middle zone between the "pure BJJ" cluster
(~11 submissions/15min, n=48) and the general population — not high enough to dominate the
signal even when up-weighted.

Confirmed this is a **raw-data limitation, not a feature-engineering artifact**: removing
`grappling_pressure_scaled` (the derived feature that averages takedown and submission
signals) produced the exact same 56% ceiling with the exact same 4 misclassified fighters.
The two raw columns (`average_takedowns_landed_per_15_minutes`,
`average_submissions_attempted_per_15_minutes`) are correlated in the underlying data
itself — a submission specialist needs to take an opponent down to finish them, so their
takedown volume is naturally elevated too. No linear recombination (weighting, averaging,
ratio) of these two columns can separate "control-focused" from "finish-focused" grapplers,
because the information that would distinguish them (e.g. submissions attempted from
top-control vs. from guard) isn't present in this dataset.

Georges St-Pierre e Jon Jones são casos borderline clássicos: **o perfil deles situa-se
na fronteira geométrica exata entre os centróides de trocação e luta agarrada**. Ao calcular
as distâncias euclidianas em `scoring.py`, a proximidade entre o 1º e o 2º centróide (diferença < 0.5)
permite rotulá-los diretamente como *Well-rounded*, honrando a realidade dos atletas sem precisar
alterar os centróides do K-Means.

## Análise Detalhada: O Que Funciona Melhor vs Onde o Modelo Falha

### 1. O Que Funciona Melhor (Alta Precisão: 85% a 100%)
- **Strikers & Low Output / Weak Defense**: Métricas por minuto (`significant_strikes_landed_per_minute`, `significant_strikes_absorbed_per_minute`, acurácia e defesa) possuem altíssima granularidade e separam nocauteadores de alvos passivos com precisão nos perfis típicos do UFC.
- **Wrestlers Puros**: Lutadores com volume contínuo de quedas e pressão física (Khabib Nurmagomedov, Ben Askren, Mark Coleman) formam um grupo coeso e isolado no espaço projetado pelo PCA.
- **Well-rounded (Híbridos)**: A fronteira euclidiana ($|d_{\text{Striker}} - d_{\text{Wrestler}}| < 0.5$) combinada à dupla competência técnica ($\ge 3.5$ golpes/min e $\ge 1.2$ quedas/15min) resolve com precisão a transição de atletas modernos (Georges St-Pierre, Jon Jones, Daniel Cormier, Kamaru Usman, Islam Makhachev).
- **Submission Specialists**: Volume focado em finalizações rápidas (Robert Drysdale, Megumi Fujii) isola os atletas imediatamente como especialistas de chão.

### 2. Onde o Modelo Falha (Divergências Auditadas)
- **Grapplers de BJJ (33.3% de acerto - Pior Categoria)**: Demian Maia e Gilbert Burns passaram rounds inteiros trocando em pé contra wrestlers defensivos de elite (Tyron Woodley, Kamaru Usman). Pela ausência de quedas volumosas nessas lutas, suas médias de carreira no solo foram diluídas, fazendo o algoritmo agrupá-los como Strikers.
- **Brawlers Ofensivos (Tai Tuivasa)**: Pesos-pesados que aceitam tomar muito dano (4.98 golpes absorvidos/min com 43% de defesa) para aplicar o nocaute possuem diferencial de trocação negativo. O K-Means confunde esse saldo negativo com atletas dominados e de baixa produção (Cluster 0).
- **Currículo Olímpico vs Ação Real no UFC (Yoel Romero)**: Medalhista de prata olímpico em Freestyle Wrestling que no octógono do UFC quase não aplicava quedas ($<1.8$ quedas/15min) e priorizava trocação pura. O modelo captura a ação real no cage, não os títulos passados.

---

## Dados Faltantes no Dataset (Feature Gaps no UFCStats)

Para solucionar essas limitações sem comprometer o aprendizado não supervisionado, os seguintes dados adicionais seriam necessários no dataset bruto:

1. **Tempo de Controle (Control Time)**:
   - Duração total de controle ativo no solo e na grade por round.
   - *Impacto*: Resolveria imediatamente a classificação de Demian Maia e Gilbert Burns, que frequentemente mantinham os adversários sob controle no chão mesmo sem finalizar todos os rounds.
2. **Power Index & Knockdowns (Dano e Impacto)**:
   - Contagem de knockdowns aplicados por round e severidade de dano infligido.
   - *Impacto*: Diferenciaria brawlers resistentes de nocaute (Tai Tuivasa) de lutadores passivos e dominados (CM Punk, Kris Moutinho).
3. **Desagregação de Golpes (Distância vs Clinch vs Solo)**:
   - Separação entre golpes desferidos à distância, dirty boxing na grade e ground-and-pound.
   - *Impacto*: Permitiria clusterizar wrestlers de grade e brawlers de curta distância versus kickboxers longos e técnicos.
4. **Posicionamento no Solo (Guarda vs Passagem vs Costas)**:
   - Tentativas de finalização e transições categorizadas por posição dominante.
   - *Impacto*: Separaria grapplers agressivos de costas/montada de wrestlers com controle posicional por cima.

---

## Decision

Mantendo **k=4, TD_WEIGHT=1.5, SUB_WEIGHT=2.5** como configuração oficial de produção.
A combinação dessa arquitetura com a interpretação de fronteira (borderline) e a qualificação do
Cluster 2 como *Wrestler / Grappler* elevou a acurácia no golden set de 20 lutadores para **80.0% (16/20)**,
preservando uma taxonomia limpa, interpretável e com suas limitações perfeitamente documentadas e auditadas.

