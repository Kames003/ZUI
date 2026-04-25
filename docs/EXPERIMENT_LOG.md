# Vývojový log — 2048 AI Solvery
*Iteratívne ladenie a zdôvodnenie rozhodnutí*

---

## Beh č. 1 — Základná implementácia (pôvodný stav)

**Konfigurácia:**
- Random: náhodný platný ťah
- Heuristic: pevná priorita smerov `down > left > up > right`
- ExpectiMax: depth=3, MONO=1.0, SMOOTH=0.1, EMPTY=2.7, CORNER=1.0, corner_mode='any'
- Všetky 3 solvery zdieľali jeden spoločný seed (`np.random.seed(42)` raz na začiatku)

**Výsledky (30 hier):**

| Solver | Priem. skóre | Max dlaždica | Výhry/30 |
|--------|-------------|--------------|---------|
| Random | ~1 100 | 256 | 0/30 |
| Heuristic | ~2 193 | 1024 | 0/30 |
| ExpectiMax | ~14 000 | 2048 | 24/30 (80%) |

**Pozorovaný problém:**
Heuristika s pevnou prioritou `down > left` dosiahla len priem. 2 193 — iba 2× lepšie ako náhoda. Potenciál bol výrazne vyšší.

---

## Iterácia 1 — Vylepšenie heuristiky: greedy 1-ply eval

**Diagnóza:** Pevná priorita smerov ignoruje aktuálny stav plochy. Ťah `down` môže byť v konkrétnej situácii horší ako `right`.

**Zmena:** Heuristic solver prešiel na **greedy 1-ply evaluáciu** — pre každý platný ťah sa vypočíta skóre plochy pomocou eval funkcie (monotonicity + smoothness + empty + corner bonus) a vyberie sa najlepší ťah.

```python
# Pred zmenou:
PRIORITY = {'down': 3, 'left': 2, 'up': 1, 'right': 0}
return max(valid_moves, key=lambda m: PRIORITY[m])

# Po zmene:
for move in MOVES:
    g_new, _ = apply_move(grid, score, move)
    val = evaluate(g_new)   # monotonicity + smoothness + empty + corner
    if val > best_val: best_move = move
```

**Výsledky po zmene (Beh č. 2):**

| Solver | Priem. skóre | Max dlaždica | Výhry/30 |
|--------|-------------|--------------|---------|
| Random | ~1 100 | 256 | 0/30 |
| Heuristic | **~6 905** | 2048 | 1/30 |
| ExpectiMax | ~16 412 | 2048 | **17/30 (57%)** |

**Nový problém — kontaminácia seedu:**
Heuristika teraz hrá ~469 ťahov/hru (predtým ~199) → spotrebuje ~8 000 extra náhodných čísel → ExpectiMax dostáva úplne iné rozloženia dlaždíc → výhry klesli z 24 na 17. Nejde o zhoršenie algoritmu, ide o **nefér porovnanie**.

---

## Iterácia 2 — Oprava seedu + adaptive depth pre ExpectiMax

**Diagnóza č. 1 (seed):** Zdieľaný globálny seed spôsobuje, že každý solver hrá iné hry. Solver ktorý hrá viac ťahov "spotrebuje" viac náhody a zmení sekvenciu hier pre ďalší solver. Riešenie: **resetovať seed pred každým solverom**.

```python
# Pred každým run_solver():
np.random.seed(42); random.seed(42)
```

→ Všetky 3 solvery hrajú **rovnakých 30 hier** = férové porovnanie.

**Diagnóza č. 2 (adaptive depth):** Analýzou prehier ExpectiMax ukázala, že väčšina strát nastáva v záverečnej fáze hry (≤2 prázdne políčka) kde depth=3 nestačí na správne rozhodnutie pri kritickom merge 1024→2048.

**Zmena:** Adaptive depth — pri ≤2 prázdnych políčkach sa depth zvýši z 3 na 4.

```python
BASE_DEPTH = 3
CRITICAL_DEPTH = 4
CRITICAL_THRESHOLD = 2  # trigger keď ≤2 prázdne políčka

depth = CRITICAL_DEPTH if empty_count <= CRITICAL_THRESHOLD else BASE_DEPTH
```

*Poznámka: Threshold=4 bol testovaný ale zamietnutý — spomaľuje hru 6× bez výrazného zlepšenia.*

**Očakávané výsledky po oprave (Beh č. 3 — prebieha):**

| Solver | Očakávané priem. skóre | Očakávané výhry/30 |
|--------|------------------------|-------------------|
| Random | ~1 100 | 0/30 |
| Heuristic | ~6 000–7 000 | 0–1/30 |
| ExpectiMax | ~16 000–20 000 | **≥24/30 (≥80%)** |

---

## Zdôvodnenie výberu solverov

### Prečo práve tieto 3 typy?

**Random** — zámerná baseline. Ukazuje minimálnu úroveň bez akejkoľvek inteligencie (~1 100 skóre).

**Greedy heuristic** — rule-based AI: ručne navrhnutá eval funkcia, žiadne učenie, žiadny strom. Stredná úroveň (~6 000–7 000 skóre, 3–4× lepšia ako náhoda).

**ExpectiMax** — pokročilá AI: správne modeluje stochastickú povahu hry (náhodné spawnovanie dlaždíc) cez chance-nody. Kanonická metóda pre 2048 v AI literatúre.

### Prečo nie MCTS alebo neurónová sieť?

| Alternatíva | Problém |
|-------------|---------|
| MCTS | Vyžaduje stovky simulácií na ťah — podobná rýchlosť, ťažšia implementácia |
| Neurónová sieť | Potrebuje predtrénovanie (hodiny dát) — mimo rozsahu zadania |
| Alpha-Beta | Neplatí pre stochastické hry (spawnovanie nie je protihráč) |

**ExpectiMax je jedinou správnou voľbou** pre stochastickú hru s úplnou informáciou ako 2048.

---

## Zhrnutie iterácií

```
Beh č.1: Heuristic 2 193  │  ExpectiMax 24/30
         ↓ problém: heuristika ignoruje stav plochy
Beh č.2: Heuristic 6 905  │  ExpectiMax 17/30 ← seed contamination
         ↓ problém: zdieľaný seed = nefér hry, depth=3 nestačí v endgame
Beh č.3: Heuristic ~7 000 │  ExpectiMax ≥24/30 ← prebieha
```

---

*Posledná aktualizácia: 2026-04-20*
