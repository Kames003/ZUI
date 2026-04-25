# ZUI Cvičenie — 2048 AI Solvery: Kompletný kontext zadania

> Tento súbor slúži ako referenčný kontext pre AI agentov a pre prípravu reportu/prezentácie.
> Obsahuje: text zadania, vysvetlenie pedagóga, obsah tabule, a stav nášho riešenia.

---

## 1. Text zadania (originál, česky)

Tuto (asi už historickou) oblíbenou mobilní hříčku snad znáte. Vaším úkolem v tomto cvičení je zkusit ji zreprodukovat a přijít s vlastní (více či méně inteligentní) strategií pro řešení této hry.

Otestujete, jak se Vámi vytvořené "umělé inteligenci" daří ve smyslu výher/proher, dosažené skóre ve 30 hrách (základní statistika - nejlepší skóre, nejhorší skóre, průměrné skóre, počet výher/počet proher, průměrná dosažená hodnota maximální buňky ve hře, průměrný počet tahů na hru v jednotlivých směrech/celkově, atp.).

### Požiadavky na odovzdanie (zip obsahujúci):
- Kód
- Report
- Video s animáciami
- Štatistika v Exceli
- Prípadne iné formáty

### 3 vlastné solvery (príklady zo zadania):
- Čistý random
- Jednoduchá pravidla pre posun
- Monte Carlo metoda
- Minimax algoritmus
- Iná heuristika
- **ExpectiMax** ← toto sme zvolili

### Milníky cvičení:
1. Funkčný deterministický solver + random
2. Heuristika + úvaha nad AI readiness a požiadavky na AI
3. Implementácia AI (konzultácia prístupov) — môže byť 3. solver aj nad rámec (RL, agenti...)
4. Ladenie a finetuning AI až do konca kurzu → **funkčná AI = výborná za skúšku**

---

## 2. Take-home message (kľúčové myšlienky pedagóga)

### Čo je vlastne AI?
- **Symbolická AI (GOFAI):** prohľadávanie stavového priestoru, expertné systémy, pravidlá → sem patrí náš ExpectiMax
- **Štatistická AI / Machine Learning:** vyžaduje DATA, tréning → neuronové siete, RL
- Mylná predstava: "ak to nemá miliardu parametrov a netrénuje sa na GPU, nie je to AI" — **NESPRÁVNE**
- ExpectiMax s dobrou heuristikou **JE umelá inteligencia** (klasická/symbolická)

### "Kladivo na komára" — Je AI všade potrebná?
Kľúčová otázka zadania:
> "Kto dosiahol najlepšie skóre a akým solverom? Koľko času zabralo vymyslieť heuristiku a koľko by zabralo natrénovať neurónovú sieť?"

**Overengineering:** Nasadzovať Deep Learning na problém, ktorý sa dá deterministicky prohľadať na pár ťahov dopredu, je zbytočné plýtvanie výpočtovým výkonom. AI má zmysel tam, kde heuristiky zlyhávajú alebo je stavový priestor príliš veľký/neznámy (rozpoznávanie obrazu, LLM...). Pre 4×4 doskovú hru? — ExpectiMax stačí.

### AI Readiness — Ako by sa dal problém riešiť cez ML?
Keby sme **MUSELI** použiť neurónovú sieť alebo RL, definícia problému:

**Reinforcement Learning — 3 piliere:**
- **State (stav):** Matica 4×4 čísel. Lepšie: one-hot encoding mocnín dvojky (aby sieť chápala že 1024 a 2048 sú si blízko, nie 2 a 1024)
- **Action (akcia):** 4 diskrétne smery (left, right, up, down)
- **Reward (odmena):** Najčastejšia chyba! Možnosti:
  - Celkové skóre? (reward hacking riziko)
  - Vytvorenie prázdneho políčka?
  - Spojenie dvoch najvyšších dlaždíc?
  - → Pozor na **Reward Hacking**

### Zlatá stredná cesta — Synergia klasiky a ML
**AlphaGo princíp:**
- Klasický stromový prechod (MCTS / ExpectiMax)
- + Neurónová sieť ako **evaluátor pozície** (namiesto ručne písanej heuristiky)
- Value network predikuje "hodnotu" stavu na konci stromu
- Policy network navrhuje sľubné ťahy (orezávanie stromu)

---

## 3. Obsah tabule (rekonštrukcia)

```
┌─────────────────────────────────────────────────────────────────┐
│                         AlphaGo                                 │
│  Random → náhodný pohyb                                         │
│                                  AI ──→ symbolická → GOFAI      │
│  Deterministický → Rule-based        ──→ štatistická/ML         │
│                  → All in ← MonteCarlo      │                   │
│                                             ├→ DATA?            │
│  Ensemble → Hierarchicky                    ├→ formát dát?      │
│                                             └→ tréning!         │
│                                                                  │
│              ┌──────────────────────────────────┐               │
│              │  Minimax   Expectimax    Value    │               │
│              └──────────────────────────────────┘               │
│                                    RL (circled)                  │
│                                    ├→ State                      │
│                                    ├→ Policy → ?                 │
│                                    └→ Reward → ? → ? → ?        │
└─────────────────────────────────────────────────────────────────┘
```

**Interpretácia tabule:**
- Ľavá strana: spektrum solverov od Random → Deterministický (rule-based) → All-in (Monte Carlo) → Ensemble (hierarchické kombinácie)
- Pravá strana: dichotómia AI — Symbolická (GOFAI: Minimax, ExpectiMax) vs. Štatistická (ML/RL)
- RL vetva: otvorené otázky okolo definície State/Policy/Reward — pedagóg ukazuje že toto **nie je triviálne**
- AlphaGo ako príklad spojenia oboch svetov (navrchu tabule = vrchol hierarchie)

---

## 4. Naše riešenie — súlad so zadaním

| Požiadavka | Stav | Detail |
|-----------|------|--------|
| 3 solvery | ✅ | Random, Greedy Heuristic, ExpectiMax |
| Štatistika 30 hier | ✅ | avg/best/worst score, wins, max tile, ťahy per smer |
| Excel výstup | ✅ | `2048_results.xlsx` — 5 listov |
| Animácie / GIFy | ✅ | `replay_random.gif`, `replay_heuristic.gif`, `replay_expectimax.gif` |
| Grafy / PNG | ✅ | dashboard, score dist., win/loss, max tile dist., move directions |
| Report | 🔄 | v príprave |
| Zip pre odovzdanie | 🔄 | po dokončení experimentu |

### Naše 3 solvery — zdôvodnenie výberu:

**1. Random** — zámerná dolná hranica, ukazuje výsledky bez akejkoľvek inteligencie

**2. Greedy Heuristic (Rule-based)** — 1-ply evaluácia všetkých ťahov pomocou ručne navrhnutej eval funkcie:
- Monotonicity (usporiadanosť riadkov/stĺpcov)
- Smoothness (podobnosť susedných dlaždíc)
- Empty cells (počet prázdnych políčok)
- Corner bonus (max dlaždica v rohu)
- Deterministická, žiadne učenie → klasická symbolická AI

**3. ExpectiMax** — pokročilá symbolická AI:
- Expectiminimax strom (max-nody + chance-nody)
- Správne modeluje stochastické spawnovanie (90% dlaždica 2, 10% dlaždica 4)
- depth=3 bežne, depth=4 pri ≤2 prázdnych políčkach (adaptive depth)
- Eval funkcia: MONO=1.0, SMOOTH=0.1, EMPTY=2.7, CORNER=1.0

### Prečo nie MCTS ani neurónová sieť?
- **MCTS:** Rýchlejší na ťah ale vyžaduje stovky simulácií → podobná rýchlosť, ťažšia implementácia, žiadny jasný prínos
- **RL / neurónová sieť:** Potrebuje predtrénovanie (hodiny/dni dát), mimo rozsahu zadania, "kladivo na komára"
- **Záver:** ExpectiMax je kanonická a najefektívnejšia metóda pre 2048 — potvrdzuje take-home message zadania

---

## 5. Výsledky experimentov (iteratívny vývoj)

Viď: `EXPERIMENT_LOG.md` — obsahuje kompletný vývojový log s číslami.

**Skrátene:**
- Beh č.1: Heuristic avg=2 193, ExpectiMax 24/30 výhier (80%)
- Beh č.2: Heuristic avg=6 905 (greedy eval), ExpectiMax 17/30 (seed contamination bug)
- Beh č.3: Heuristic ~7 000, ExpectiMax ≥24/30 (oprava seedu + adaptive depth) — *prebieha*

---

## 6. Kľúčové súbory projektu

| Súbor | Účel |
|-------|------|
| `2048_ai.ipynb` | Hlavný notebook — kompletné riešenie |
| `create_notebook.py` | Generátor notebooku (zdrojový kód) |
| `test_2048.py` | Test suite — 42 testov |
| `tune_expectimax.py` | Tunovací skript pre eval váhy |
| `EXPERIMENT_LOG.md` | Vývojový log s iteráciami a číslami |
| `2048_results.xlsx` | Výsledky experimentu (Excel) |
| `archive/run_beh2_seed_contamination/` | Grafy z Behu č.2 (pred opravou) |

---

*Posledná aktualizácia: 2026-04-20*
