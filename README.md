# ZUI — Základy umelej inteligencie

**Autor:** Tomáš Mucha · VŠB-TUO FEI · 2026

Repozitár obsahuje dve vypracované úlohy z predmetu Základy umelej inteligencie.

---

## Projekty

### 2048 AI Solvery

Implementácia a porovnanie troch AI solverov pre hru 2048. Každý solver reprezentuje iný prístup — od náhodného pohybu cez deterministickú heuristiku až po klasickú symbolickú AI.

| Solver | Avg skóre | Win rate | Max dlaždica |
|---|---|---|---|
| Random | 1 143 | 0 % | 256 |
| Greedy Heuristic | 7 225 | 0 % | 1 024 |
| **ExpectiMax** | **17 990** | **63 %** | **2 048** |

*30 hier per solver, seed=42*

**Kľúčové súbory:**
- [`2048_ai.ipynb`](2048_ai.ipynb) — hlavný notebook s kompletným riešením a výstupmi
- [`REPORT.md`](REPORT.md) — report s metodológiou, výsledkami a diskusiou
- [`images/`](images/) — grafy, dashboard, animácie GIF
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — vývojový log (4 iterácie ladenia)

---

### Bonus Task 2 — Sémantická klasifikácia textu (NLP)

Porovnanie TF-IDF vs. Sentence Embeddings vs. FAISS k-NN na datasete AG News (4 kategórie spravodajských článkov).

| Metóda | Accuracy | Poznámka |
|---|---|---|
| TF-IDF + Logistic Regression | 87.00 % | Lexikálny baseline, 2.6 s |
| Sentence Embeddings + LR | 87.70 % | all-MiniLM-L6-v2, 384-dim |
| **FAISS k-NN (k=5)** | **87.60 %** | Bez trénovania, 0.023 ms/query |

*seed=42, 5 000 train / 1 000 test z AG News*

**Kľúčové súbory:**
- [`NLP/nlp_classification.ipynb`](NLP/nlp_classification.ipynb) — notebook so všetkými 4 taskmi a výstupmi
- [`NLP/REPORT.md`](NLP/REPORT.md) — report s grafmi inline
- [`NLP/images/`](NLP/images/) — confusion matrix, t-SNE, TF-IDF features, F1 per class

---

## Webová prezentácia

**https://kames003.github.io/ZUI/**

---

## Štruktúra repozitára

```
ZUI/
├── 2048_ai.ipynb              # Hlavný notebook — 2048 AI
├── REPORT.md                  # Report — 2048
├── images/                    # Grafy a GIF animácie
├── create_notebook.py         # Generátor notebooku
├── run_experiment.py          # Spúšťač experimentu
├── test_2048.py               # Test suite (42 testov)
├── tune_expectimax.py         # Tuning ExpectiMax váh
├── 2048_results.xlsx          # Štatistiky v Exceli
├── NLP/
│   ├── nlp_classification.ipynb
│   ├── REPORT.md
│   ├── create_nlp_notebook.py
│   └── images/
└── docs/
    ├── EXPERIMENT_LOG.md      # Vývojový log 2048
    └── ZADANIE_KONTEXT.md     # Kontext zadania
```
