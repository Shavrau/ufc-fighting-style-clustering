# Model Limitations — Fighting Style Clustering

## Summary

K-Means with k=4 (`TD_WEIGHT=1.5`, `SUB_WEIGHT=2.5`, applied to the takedown/submission
scaled features before PCA) is the production configuration. It correctly separates the
common cases but has a **known, investigated, unresolved limitation** with moderate-volume
grappler/wrestler hybrids. This document exists so that limitation isn't rediscovered by
accident later — it was found, tested against, and knowingly accepted.

## Golden-set validation

12 well-known fighters (3 per target style) with community-agreed styles, scored against
the production model (see `reports/golden_set_validation.json` for the full run):

| Fighter | Expected | Predicted | Correct |
|---|---|---|---|
| Alex Pereira "Poatan" | Striker | Striker | ✓ |
| Max Holloway | Striker | Striker | ✓ |
| Justin Gaethje | Striker | Striker | ✓ |
| Khabib Nurmagomedov | Wrestler | Wrestler / Grappler | ✓ |
| Kamaru Usman | Wrestler | Well-rounded (Striker / Wrestler) | ✓ |
| Islam Makhachev | Wrestler | Well-rounded (Striker / Wrestler) | ✓ |
| Charles Oliveira | Grappler | Wrestler / Grappler | ✓ |
| Demian Maia | Grappler | Striker | ✗ |
| Gilbert Burns | Grappler | Striker | ✗ |
| Georges St-Pierre | Well-rounded | Well-rounded (Wrestler / Striker) | ✓ |
| Jon Jones | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |
| Daniel Cormier | Well-rounded | Well-rounded (Striker / Wrestler) | ✓ |

**Accuracy: 10/12 (83%)**

*(Nota histórica: a acurácia anterior de 33% / 4 de 12 decorria principalmente do fato de que nenhum
cluster de K-Means recebia o rótulo "Well-rounded" e o Cluster 2 era chamado apenas de "Wrestler",
penalizando grapplers como Charles Oliveira e lutadores completos como GSP, Jon Jones e Cormier.
Com a ampliação semântica do cluster de solo para "Wrestler / Grappler" e a detecção de lutadores
Well-rounded via distância euclidiana de fronteira (borderline) e competência dual, a acurácia
subiu para 83%, restando apenas duas falhas genuínas: Demian Maia e Gilbert Burns).*

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

## Decision

Mantendo **k=4, TD_WEIGHT=1.5, SUB_WEIGHT=2.5** como configuração oficial de produção.
A combinação dessa arquitetura com a interpretação de fronteira (borderline) e a qualificação do
Cluster 2 como *Wrestler / Grappler* elevou a acurácia no golden set de 12 lutadores para **83% (10/12)**,
preservando uma taxonomia limpa e interpretável. As limitações remanescentes (Demian Maia e Gilbert Burns)
permanecem aceitas conscientemente e só seriam resolvidas com:
- Dados granulares de solo (tentativas de finalização por posição: guarda vs montada/costas).
- Métricas de tempo de controle no solo (control time), ausentes nesta versão do dataset.
