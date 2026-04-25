# NLP Bonus Task 2 — Sémantická klasifikácia textu

**Predmet:** Základy umelej inteligencie (ZUI) · Bonus Task 2  
**Autor:** Tomáš Mucha · VŠB-TUO FEI · 2026

---

## Téma

Porovnanie troch prístupov k textovej klasifikácii na datasete AG News (4 kategórie, 120k článkov):

| Metóda | Accuracy | Poznámka |
|---|---|---|
| TF-IDF + Logistic Regression | 87.00 % | Lexikálny baseline, 2.6 s |
| Sentence Embeddings + LR | 87.70 % | all-MiniLM-L6-v2, 384-dim |
| **FAISS k-NN (k=5)** | **87.60 %** | Bez trénovania, 0.023 ms/query |

*seed=42, 5 000 train / 1 000 test z AG News*

---

## Súbory

| Súbor | Popis |
|---|---|
| [`nlp_classification.ipynb`](nlp_classification.ipynb) | Hlavný notebook — 4 tasky s výstupmi a grafmi |
| [`REPORT.md`](REPORT.md) | Detailný report s metodológiou, grafmi a závermi |
| [`images/`](images/) | Confusion matrix, t-SNE, TF-IDF features, F1 per class, accuracy comparison |
| [`create_nlp_notebook.py`](create_nlp_notebook.py) | Generátor notebooku |

---

## Obsah notebooku

1. **Task 1 — TF-IDF Baseline** — `TfidfVectorizer(max_features=50k, ngram_range=(1,2))` + Logistic Regression
2. **Task 2 — Sentence Embeddings** — `all-MiniLM-L6-v2` (384-dim), porovnanie s TF-IDF
3. **Task 3 — Analýza chýb** — Confusion matrix, F1 per class, 2 reálne príklady chýb
4. **Task 4 — FAISS k-NN** — `IndexFlatIP`, k=5, majority vote, diskusia RAG a Vector DB

---

## Kľúčové závery

- TF-IDF je silný baseline pre lexikálne odlišné datasety — rozdiel vs. embeddings je iba +0.70 pp
- FAISS k-NN bez trénovania dosahuje porovnateľnú presnosť ako trénovaný klasifikátor
- Chyby sú systematické (Business↔Sci/Tech, World↔Business) — odrážajú reálnu ambiguitu kategórií
- Princíp FAISS k-NN je základom produkčných RAG systémov (Pinecone, Qdrant, pgvector)
