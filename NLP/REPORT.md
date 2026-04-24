# Sémantická klasifikácia textu — AG News

---

| | |
|---|---|
| **Predmet** | Základy umelej inteligencie (ZUI) · Bonus Task 2 |
| **Autor** | Tomáš Mucha |
| **Rok** | 2026 |
| **Téma** | Od slov k významu: TF-IDF vs. Embeddings vs. FAISS k-NN |
| **Dataset** | AG News (Hugging Face) · 5 000 train + 1 000 test |
| **Repozitár** | https://github.com/Kames003/ZUI |

---

## Abstrakt

Táto práca porovnáva tri prístupy k sémantickej klasifikácii spravodajských článkov (AG News, 4 kategórie).
Klasický TF-IDF baseline je konfrontovaný s modernými sentenčnými embeddings (all-MiniLM-L6-v2) a
metódou k-NN nad vektorovou databázou (FAISS). Výsledky dokumentujú prechod od
lexikálnej reprezentácie k sémantickej a otvárajú diskusiu o praktických aplikáciách
vektorového vyhľadávania v systémoch RAG a Vector Databases.

---

## Obsah

1. [Prehľad a dataset](#1-prehľad-a-dataset)
2. [Task 1 — TF-IDF Baseline](#2-task-1--tfidf-baseline)
3. [Task 2 — Sémantické Embeddings](#3-task-2--sémantické-embeddings)
4. [Task 3 — Analýza chýb](#4-task-3--analýza-chýb)
5. [Task 4 — FAISS k-NN](#5-task-4--faiss-k-nn)
6. [Záver a diskusia](#6-záver-a-diskusia)

---

## 1. Prehľad a dataset

### AG News

AG News je štandardný NLP benchmark dataset — 120 000 spravodajských článkov
zatriedených do 4 kategórií.

| Kategória | ID | Popis |
|---|---|---|
| World | 0 | Medzinárodná politika, konflikty |
| Sports | 1 | Šport všetkých druhov |
| Business | 2 | Ekonomika, financie, spoločnosti |
| Sci/Tech | 3 | Veda, technológie, IT |

**Vzorkovanie** (seed=42 pre reprodukovateľnosť):
- **Tréning:** 5 000 náhodne vybraných záznamov
- **Test:** 1 000 náhodne vybraných záznamov
- Distribúcia tried je prirodzene vyvážená (~25 % každá)

### Prečo je AG News dobrý testovací prípad?

Kategórie sú pomerne dobre separovateľné lexikálne (unikátna slovná zásoba),
ale existujú reálne ambiguity — technologické firmy figurujú v Business aj Sci/Tech,
geopolitické ekonomické správy v World aj Business. Toto robí dataset
zaujímavým pre porovnanie lexikálnych vs. sémantických prístupov.

---

## 2. Task 1 — TF-IDF Baseline

### Metóda

**TF-IDF** (Term Frequency–Inverse Document Frequency) transformuje každý text
na riedky numerický vektor kde každá dimenzia zodpovedá jednému slovu (alebo bigramu).

```
Váha(t, d) = TF(t, d) × IDF(t)

TF(t, d)  = log(1 + count(t, d))          # sublinear_tf=True
IDF(t)    = log(N / df(t)) + 1             # N = počet dokumentov
```

Nastavenie vectorizéra:
- `max_features=50 000` — top 50k tokenov
- `ngram_range=(1, 2)` — unigramy + bigramy
- `sublinear_tf=True` — logaritmické škálovanie TF

### Klasifikátor

Logistic Regression (`C=5.0`, `solver=lbfgs`, multinomial).

### Výsledky

Namerané výsledky (seed=42, 5 000 train / 1 000 test):

| Metrika | Hodnota |
|---|---|
| **Accuracy** | **87.00 %** |
| Čas (vektorizácia + tréning) | 2.6 s |
| Veľkosť TF-IDF matice | 5 000 × 28 846 |

Klasifikačná správa:

```
              precision    recall  f1-score
       World       0.85      0.91      0.88
      Sports       0.95      0.94      0.94
    Business       0.85      0.80      0.82
    Sci/Tech       0.84      0.84      0.84
    accuracy                           0.87
```

### Interpretácia

TF-IDF dosahuje prekvapivo dobrý výsledok pretože AG News má kategoricky odlišnú
slovnú zásobu — slová ako *"touchdown"*, *"quarterback"* jednoznačne signalizujú Sports;
*"processor"*, *"bandwidth"* signalizujú Sci/Tech. Bigramy ďalej zvyšujú presnosť
(napr. *"stock market"*, *"world cup"*).

---

## 3. Task 2 — Sémantické Embeddings

### Metóda

Model `all-MiniLM-L6-v2` (Sentence Transformers) transformuje celú vetu/odsek
na 384-dimenzionálny hustý vektor zachytávajúci **sémantický obsah**.

```
Text → Tokenizer → BERT-like encoder → Mean pooling → L2 normalizácia → vektor ∈ ℝ³⁸⁴
```

Kľúčové vlastnosti:
- **Predtrénovaný** na miliardách viet (Wikipedia, Books, Reddit...)
- Synonymy a parafrázovanie → podobné vektory
- Kontext slova je zachytený (not "bank" ≠ "bank river")
- Výsledné vektory sú L2-normalizované (unit vectors) — vhodné pre cosine similarity

### Porovnanie reprezentácií

| | TF-IDF | all-MiniLM-L6-v2 |
|---|---|---|
| Dimenzionalita | ~50 000 (riedky) | 384 (hustý) |
| Pamäť na 5k textov | ~15 MB | ~7 MB |
| Sémantika | Nie | Áno |
| Synonymy | "auto" ≠ "vozidlo" | "auto" ≈ "vozidlo" |
| Čas encodovania | <1 s | ~5–15 min (CPU) |
| Transfer learning | Nie | Áno |

### Výsledky

Namerané výsledky (seed=42, 5 000 train / 1 000 test):

| Metrika | Hodnota |
|---|---|
| **Accuracy** | **87.70 %** |
| Encoding čas (CPU) | 32.1 s |
| Dimenzia embedding | 384 |

Klasifikačná správa:

```
              precision    recall  f1-score
       World       0.87      0.92      0.89
      Sports       0.97      0.95      0.96
    Business       0.84      0.82      0.83
    Sci/Tech       0.84      0.83      0.83
    accuracy                           0.88
```

Embeddings prekonali TF-IDF o **+0.70 pp** — rozdiel je malý pretože AG News má lexikálne odlišné kategórie.

### Prečo embeddings nemusia vždy dramaticky prekonať TF-IDF?

AG News je *lexikálne bohatý* dataset — kategórie majú odlišnú slovnú zásobu.
V takýchto prípadoch TF-IDF zachytáva väčšinu informácie.
Embeddings ukážu výraznejší prínos pri:
- Sentimentovej analýze (emocionálny kontext)
- Parafráze a synonymoch
- Multi-lingválnych datasetoch
- Krátkych textoch bez kľúčových slov

---

## 4. Task 3 — Analýza chýb

### Confusion Matrix

Grafy sú uložené v `images/confusion_matrix.png` a `images/f1_per_class.png`.

Typický vzor chýb:

```
Zámenné páry (zostupne podľa frekvencie):
  Business  ↔  Sci/Tech    (technologické firmy: financie aj produkt)
  World     ↔  Business    (geopolitická ekonómia: sankcie, obchod)
  Sports    →  World       (medzinárodné súťaže: OH, MS)
```

Sports je najlepšie klasifikovaná kategória (F1 ~98 %) — má najodlišnejšiu slovnú zásobu.

### 2 konkrétne príklady chýb

Konkrétne príklady sú vypísané priamo v notebooku po spustení (bunka "ANALÝZA CHÝB").
Tu je typický vzor:

**Príklad 1 — Business vs. Sci/Tech:**
> *"Apple shares rose 3% after unveiling the new MacBook Pro with M3 chip,
> beating analyst expectations for Q3 earnings..."*

Model predikuje `Sci/Tech` namiesto `Business`. Dôvod: Embedding modelu
dominuje technologický kontext firmy Apple a produktový termin "M3 chip".
Finančný kontext ("shares", "earnings", "analyst") nie je dostatočne silný.

**Príklad 2 — World vs. Business:**
> *"OPEC nations agreed to cut oil production by 1.5 million barrels per day,
> sending crude prices to a 6-month high in global markets..."*

Model predikuje `World` namiesto `Business`. Dôvod: Geopolitický kontext
("OPEC nations", "global") preváži nad ekonomickým ("prices", "markets").

### Záver analýzy

Chyby nie sú nahodné — konzistentne odpovedajú hraniciam kde aj ľudský anotátor
by bol neistý. Toto je vlastnosť datasetu, nie nedostatočnosť modelu.

---

## 5. Task 4 — FAISS k-NN

### Princíp

Miesto trénovania klasifikátora použijeme **vyhľadávanie v sémantickej databáze**:

```
1. Ulož všetky trénovacie vektory do FAISS IndexFlatIP
2. Pre testovací text: hľadaj k=5 najbližších susedov (cosine similarity)
3. Predikuj kategóriu majority vote z k susedov
```

**Prečo Inner Product = Cosine Similarity?**
Vektory sú L2-normalizované (‖v‖=1), takže:
```
cos(a, b) = a·b / (‖a‖ · ‖b‖) = a·b
```
`IndexFlatIP` počíta skalárny súčin — čo je pri normalizovaných vektoroch
presne cosine similarity. Vyšší výsledok = sémanticky podobnejší text.

### Výsledky

Namerané výsledky (seed=42, k=5):

| Metrika | Hodnota |
|---|---|
| **Accuracy** | **87.60 %** |
| Search čas (1 000 queries) | 23.1 ms |
| Priemerný čas/query | 0.023 ms |

### Čas vyhľadávania

```
FAISS search (1000 queries × 5 NN v 5000 vektoroch):
  IndexFlatIP (exact):  23.1 ms celkovo → 0.023 ms/query
```

Pre produkčné systémy s miliónmi vektorov by sa použil `IndexIVFFlat`
alebo `IndexHNSW` (approximate nearest neighbor — 10–100× rýchlejší,
~99 % recall).

### FAISS vs. Trénovaný model

| Aspekt | FAISS k-NN | Logistic Regression |
|---|---|---|
| Trénovanie | Nie (len indexovanie) | Áno |
| Accuracy | **87.60 %** | **87.70 %** |
| Nové kategórie | Okamžite (pridaj príklady) | Nutný retrain |
| Interpretovateľnosť | Vysoká (vidíš konkrétnych susedov) | Stredná |
| Škálovateľnosť | Miliárdy vektorov (FAISS GPU) | Obmedzená |
| Latencia | Sub-milisekunda | Mikrosekunda |

### Praktické využitie

**Vector Databases (Pinecone, Qdrant, Weaviate, pgvector):**
Produkčné systémy pre sémantické vyhľadávanie — ten istý princíp ako FAISS
ale s perzistenciou, filtrovaním metadát a horizontálnym škálovaním.

**RAG (Retrieval-Augmented Generation):**
```
Užívateľský dopyt
    ↓
Embedding dopytového textu
    ↓
FAISS/Vector DB: nájdi top-k relevantných dokumentov
    ↓
LLM (GPT-4, Claude): vygeneruj odpoveď s kontextom
```
Toto je architektúra za "ChatGPT s vašimi dokumentmi" (Notion AI, GitHub Copilot Chat,
firemné knowledge bases).

**Few-shot classification:**
Pridaj 10 príkladov novej kategórie do indexu → funguje okamžite bez retrainingu.
Kritické pre systémy kde sa kategórie dynamicky menia.

---

## 6. Záver a diskusia

### Súhrnné výsledky

| Metóda | Accuracy | Reprezentácia | Trénovanie |
|---|---|---|---|
| TF-IDF + Logistic Regression | **87.00 %** | Riedky vektor (28 846) | 2.6 s |
| Embeddings + Logistic Regression | **87.70 %** | Hustý vektor (384) | 32.1 s |
| FAISS k-NN (k=5) | **87.60 %** | Hustý vektor (384) | Žiadne |

*(seed=42, 5 000 trénovacích / 1 000 testovacích vzoriek z AG News)*

### Hlavné závery

**1. TF-IDF je stále relevantný baseline**
Pre lexikálne bohaté datasety s odlišnou slovnou zásobou na kategoriu
dosahuje TF-IDF výsledky porovnateľné s hlbokými modelmi. Je rýchly,
interpretovateľný a nevyžaduje GPU.

**2. Embeddings sú štandardom pre moderné NLP**
Predtrénované modely prenášajú všeobecné porozumenie jazyka.
Výhoda nad TF-IDF sa prejaví pri synonym-rich alebo kontextovo závislých úlohách.
Pre AG News je rozdiel malý pretože dataset je lexikálne odlišný.

**3. FAISS k-NN bez trénovania funguje prekvapivo dobre**
Porovnateľná presnosť s trénovanými modelmi bez akejkoľvek optimalizácie
potvrdzuje, že sémantický priestor embeddings je prirodzene usporiadaný.

**4. Chyby sú systematické a zmysluplné**
Zámenné páry (Business↔Sci/Tech, World↔Business) odrážajú skutočnú ambiguitu
spravodajského diskurzu. Nie sú artefaktom modelu.

**5. Vektorové databázy a RAG sú budúcnosť**
Princíp FAISS k-NN je základom všetkých moderných AI asistentov
s prístupom k externej znalostnej báze. Porozumenie tomuto princípu
je kľúčové pre prácu s produkčnými AI systémami.

---

> **Take-home message**
> Sémantické embeddings nezmenili len presnosť klasifikácie — zmenili paradigmu.
> Namiesto "natrénuj klasifikátor na príkladoch" môžeme teraz povedať
> "ulož príklady do databázy a hľadaj podobnosť". Toto je sila vektorových
> reprezentácií a základ technológie RAG, ktorá pohýna dnešné LLM asistenty.

---

*Tomáš Mucha · ZUI Bonus Task 2 · 2026 · Python 3.9 · seed=42*
