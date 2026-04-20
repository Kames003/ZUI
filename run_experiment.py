#!/usr/bin/env python3
"""
Spúšťa každý solver samostatne – výsledky viditeľné ihneď po dokončení.
Ukladá výsledky do pickle súborov pre generate_visuals.py.

Použitie:
  python3 run_experiment.py              # všetky tri solvery postupne
  python3 run_experiment.py random       # len Random
  python3 run_experiment.py heuristic    # len Heuristic
  python3 run_experiment.py expectimax   # len ExpectiMax
"""
import sys, copy, time, random, pickle, os
import numpy as np

RESULTS_DIR = '/Users/admin/Desktop/ZUI/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Herná logika (verbatim z 2048_base.ipynb)
# ─────────────────────────────────────────────────────────────────────────────

def add_score(sc, val): return sc + val

def move_left(grid, score):
    for i in range(4):
        nz = [x for x in grid[i,:] if x != 0]
        grid[i,:] = np.array(nz + [0]*(4-len(nz)))
        for j in range(3):
            if grid[i,j] == grid[i,j+1] != 0:
                grid[i,j] *= 2; score = add_score(score, grid[i,j]); grid[i,j+1] = 0
        nz = [x for x in grid[i,:] if x != 0]
        grid[i,:] = np.array(nz + [0]*(4-len(nz)))
    return grid, score

def move_right(grid, score):
    for i in range(4):
        nz = [x for x in grid[i,:] if x != 0]
        grid[i,:] = np.array([0]*(4-len(nz)) + nz[::-1])
        for j in range(3, 0, -1):
            if grid[i,j] == grid[i,j-1] != 0:
                grid[i,j] *= 2; score = add_score(score, grid[i,j]); grid[i,j-1] = 0
        nz = [x for x in grid[i,:] if x != 0]
        grid[i,:] = np.array([0]*(4-len(nz)) + nz[::-1])
    return grid, score

def move_up(grid, score):
    for i in range(4):
        nz = [x for x in grid[:,i] if x != 0]
        grid[:,i] = np.array(nz + [0]*(4-len(nz)))
        for j in range(3):
            if grid[j,i] == grid[j+1,i] != 0:
                grid[j,i] *= 2; score = add_score(score, grid[j,i]); grid[j+1,i] = 0
        nz = [x for x in grid[:,i] if x != 0]
        grid[:,i] = np.array(nz + [0]*(4-len(nz)))
    return grid, score

def move_down(grid, score):
    for i in range(4):
        nz = [x for x in grid[:,i] if x != 0]
        grid[:,i] = np.array([0]*(4-len(nz)) + nz[::-1])
        for j in range(3, 0, -1):
            if grid[j,i] == grid[j-1,i] != 0:
                grid[j,i] *= 2; score = add_score(score, grid[j,i]); grid[j-1,i] = 0
        nz = [x for x in grid[:,i] if x != 0]
        grid[:,i] = np.array([0]*(4-len(nz)) + nz[::-1])
    return grid, score

def add_new_number(grid):
    zi = np.where(grid == 0)
    if len(zi[0]) == 0: return False
    idx = np.random.choice(len(zi[0]))
    grid[zi[0][idx], zi[1][idx]] = 2 if np.random.random() < 0.9 else 4
    return True

def check_game_over(grid):
    if not np.all(grid): return False
    for r in range(4):
        for c in range(4):
            if r < 3 and grid[r,c] == grid[r+1,c]: return False
            if c < 3 and grid[r,c] == grid[r,c+1]: return False
    return True

def check_win(grid): return 2048 in grid

def play_2048(grid, move, score):
    orig = copy.deepcopy(grid)
    if check_game_over(grid): raise RuntimeError("GO")
    fns = {'left': move_left, 'right': move_right, 'up': move_up, 'down': move_down}
    grid, score = fns[move](grid, score)
    if check_win(grid): raise RuntimeError("WIN")
    if not np.array_equal(grid, orig): add_new_number(grid)
    return grid, score

def new_game():
    grid = np.zeros((4,4), dtype=int)
    add_new_number(grid); add_new_number(grid)
    return grid, 0

# ─────────────────────────────────────────────────────────────────────────────
# Infraštruktúra
# ─────────────────────────────────────────────────────────────────────────────
MOVES = ['left', 'right', 'up', 'down']
_FNS  = {'left': move_left, 'right': move_right, 'up': move_up, 'down': move_down}

def apply_copy(grid, score, move):
    g = copy.deepcopy(grid)
    return _FNS[move](g, score)

def is_valid(grid, score, move):
    g2, _ = apply_copy(grid, score, move)
    return not np.array_equal(g2, grid)

def run_solver(solver_fn, n_games=30, record_history=True):
    records = []
    for gi in range(n_games):
        grid, score = new_game()
        rec = {
            'score': 0, 'max_tile': 0, 'move_count': 0,
            'moves_left': 0, 'moves_right': 0, 'moves_up': 0, 'moves_down': 0,
            'outcome': '',
            'grid_history': [(copy.deepcopy(grid), score)] if record_history else [],
        }
        while True:
            move = solver_fn(grid, score)
            if move is None:
                rec['outcome'] = 'loss'; break
            try:
                grid, score = play_2048(grid, move, score)
                rec[f'moves_{move}'] += 1; rec['move_count'] += 1
                if record_history:
                    rec['grid_history'].append((copy.deepcopy(grid), score))
            except RuntimeError as e:
                if str(e) == 'WIN':
                    rec['outcome'] = 'win'
                    rec[f'moves_{move}'] += 1; rec['move_count'] += 1
                    if record_history:
                        rec['grid_history'].append((copy.deepcopy(grid), score))
                else:
                    rec['outcome'] = 'loss'
                break
        rec['score']    = score
        rec['max_tile'] = int(grid.max())
        records.append(rec)
        if (gi + 1) % 5 == 0:
            wins = sum(r['outcome'] == 'win' for r in records)
            avg  = np.mean([r['score'] for r in records])
            print(f"  hra {gi+1:2d}/30 | skóre={score:6d} | max={rec['max_tile']:4d} | "
                  f"výsledok={rec['outcome']:4s} | výhry={wins} | avg={avg:.0f}")
    return records

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 1 – Random baseline
# ─────────────────────────────────────────────────────────────────────────────
def random_solver(grid, score):
    valid = [m for m in MOVES if is_valid(grid, score, m)]
    return random.choice(valid) if valid else None

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 2 – Greedy Evaluation Heuristic
# ─────────────────────────────────────────────────────────────────────────────
# Namiesto pevnej priority vyhodnotí každý platný ťah okamžitou skórovacou
# funkciou a vyberie najlepší. Žiadny lookahead = stále deterministická heuristika.
#
# Skóre ťahu = merge_zisk + prázdne_polia × W_empty
#            + corner_bonus (max dlaždica v ľavom dolnom rohu / v ľubovoľnom rohu)
#            + priority_bias (preferencie smeru bez lookahead)

_H_W_EMPTY   = 40.0   # prázdne políčko je veľmi cenné
_H_W_CORNER  = 500.0  # max v ľavom dolnom rohu
_H_W_ANYCORNER = 200.0
_H_PRIORITY  = {'down': 30, 'left': 20, 'up': 5, 'right': 0}

def heuristic_solver(grid, score):
    """
    Greedy 1-step heuristic (bez lookahead).
    Pre každý platný ťah: skóre = zisk_z_merge + bonus_prázdnych + bonus_rohu + priorita_smeru.
    Vyberie ťah s najvyšším skóre – deterministická viackriteriálna heuristika.
    """
    best_move, best_val = None, -float('inf')
    for move in MOVES:
        g_new, s_new = apply_copy(grid, score, move)
        if np.array_equal(g_new, grid):
            continue
        merge_gain = s_new - score
        n_empty    = int(np.sum(g_new == 0))
        max_val    = int(g_new.max())
        bl_corner  = int(g_new[3, 0])   # bottom-left
        corners    = [int(g_new[0,0]), int(g_new[0,3]), int(g_new[3,0]), int(g_new[3,3])]

        corner_bonus = (_H_W_CORNER if max_val == bl_corner
                        else _H_W_ANYCORNER if max_val in corners
                        else 0.0)

        val = (merge_gain * 1.0
               + n_empty  * _H_W_EMPTY
               + corner_bonus
               + _H_PRIORITY[move])

        if val > best_val:
            best_val  = val
            best_move = move
    return best_move

# ─────────────────────────────────────────────────────────────────────────────
# SOLVER 3 – ExpectiMax (vylepšená eval + adaptívna hĺbka)
# ─────────────────────────────────────────────────────────────────────────────

def _log2(x): return float(np.log2(x)) if x > 0 else 0.0

# Snake weight matica: maximálna dlaždica patrí do ľavého dolného rohu (grid[3,0]).
# Hodnoty klesajú hadovite: (3,0)→(3,1)→(3,2)→(3,3)→(2,3)→(2,2)→...→(0,0)
# Každá pozícia má váhu 2^k kde k ∈ {1..16}.
_WEIGHTS_RAW = np.array([
    [2,      4,      8,    16],
    [256,   128,    64,    32],
    [512,  1024,  2048,  4096],
    [65536, 32768, 16384, 8192]
], dtype=float)
_WEIGHTS_NORM = _WEIGHTS_RAW / _WEIGHTS_RAW.max()   # normalizácia do [0,1]

def _snake(grid):
    """Boduje dlaždice podľa ich polohy v snake vzorke (weighted dot product v log2)."""
    return float(np.sum(np.vectorize(_log2)(grid) * _WEIGHTS_NORM))

def _mono(grid):
    """Penalizuje nemonotónne riadky/stĺpce (v log2 škále)."""
    lg = np.vectorize(_log2)(grid); t = 0.0
    for r in range(4):
        row = lg[r,:]
        t -= min(sum(max(row[i]-row[i+1],0) for i in range(3)),
                 sum(max(row[i+1]-row[i],0) for i in range(3)))
    for c in range(4):
        col = lg[:,c]
        t -= min(sum(max(col[i]-col[i+1],0) for i in range(3)),
                 sum(max(col[i+1]-col[i],0) for i in range(3)))
    return t

def _smooth(grid):
    """Penalizuje susedné dlaždice s veľkým rozdielom (ťažké budúce merge)."""
    lg = np.vectorize(_log2)(grid); p = 0.0
    for r in range(4):
        for c in range(4):
            if grid[r,c] == 0: continue
            if c < 3 and grid[r,c+1]: p -= abs(lg[r,c]-lg[r,c+1])
            if r < 3 and grid[r+1,c]: p -= abs(lg[r,c]-lg[r+1,c])
    return p

def _empty(grid):
    """log2(počet prázdnych políčok) – survival metric."""
    return float(np.log2(max(int(np.sum(grid==0)), 1)))

def _corner(grid):
    """Bonus ak max dlaždica je v niektorom rohu."""
    mv = int(grid.max())
    if mv == 0: return 0.0
    return _log2(mv) if mv in [int(grid[0,0]),int(grid[0,3]),int(grid[3,0]),int(grid[3,3])] else 0.0

# Váhy komponentov – vyladené empiricky
_W_SNAKE  = 3.0
_W_MONO   = 1.0
_W_SMOOTH = 0.1
_W_EMPTY  = 2.7
_W_CORNER = 1.0

def evaluate(grid):
    """
    Kombinovaná heuristická funkcia:
    - Snake weight matica: odmeňuje tiles v správnej hadovitej vzorke (nový!)
    - Monotonicity: korekcia pre monotónne sekvencie
    - Smoothness: susedné dlaždice s podobnými hodnotami
    - Empty cells: log2(prázdne políčka) – kľúčové pre prežitie
    - Corner bonus: max dlaždica v rohu
    """
    return (_W_SNAKE  * _snake(grid)
          + _W_MONO   * _mono(grid)
          + _W_SMOOTH * _smooth(grid)
          + _W_EMPTY  * _empty(grid)
          + _W_CORNER * _corner(grid))

def expectimax(grid, score, depth, is_max):
    if depth == 0: return evaluate(grid)
    if is_max:
        best = -float('inf'); any_v = False
        for m in MOVES:
            g2, s2 = apply_copy(grid, score, m)
            if np.array_equal(g2, grid): continue
            any_v = True
            v = expectimax(g2, s2, depth-1, False)
            if v > best: best = v
        return best if any_v else evaluate(grid)
    else:
        ep = list(zip(*np.where(grid == 0)))
        if not ep: return evaluate(grid)
        tot = 0.0
        for (r,c) in ep:
            for tv, prob in ((2, 0.9), (4, 0.1)):
                gc = copy.deepcopy(grid); gc[r,c] = tv
                tot += prob * expectimax(gc, score, depth-1, True)
        return tot / len(ep)

# Adaptívna hĺbka: keď je málo prázdnych políčok (kritické momenty hry),
# použijeme väčšiu hĺbku – viac výpočtu tam kde to naozaj záleží.
_DEPTH_DEFAULT  = 3   # bežná hra (>6 prázdnych)
_DEPTH_CRITICAL = 4   # kritická situácia (≤6 prázdnych)

def expectimax_solver(grid, score):
    """
    ExpectiMax s adaptívnou hĺbkou:
    - depth=3 keď board je relatívne voľný (≥6 prázdnych políčok)
    - depth=4 v kritických momentoch (<6 prázdnych) – väčšia presnosť kde záleží
    """
    n_empty = int(np.sum(grid == 0))
    depth   = _DEPTH_CRITICAL if n_empty < 6 else _DEPTH_DEFAULT

    best_move, best_val = None, -float('inf')
    for m in MOVES:
        g2, s2 = apply_copy(grid, score, m)
        if np.array_equal(g2, grid): continue
        val = expectimax(g2, s2, depth - 1, False)
        if val > best_val:
            best_val, best_move = val, m
    return best_move

# ─────────────────────────────────────────────────────────────────────────────
# Štatistiky
# ─────────────────────────────────────────────────────────────────────────────
def print_stats(records, name):
    scores = [r['score']      for r in records]
    tiles  = [r['max_tile']   for r in records]
    moves  = [r['move_count'] for r in records]
    wins   = sum(r['outcome'] == 'win' for r in records)
    n      = len(records)
    bar    = "═" * 54

    # Distribúcia max dlaždíc
    tile_counts = {}
    for t in tiles:
        tile_counts[t] = tile_counts.get(t, 0) + 1

    print(f"\n╔{bar}╗")
    print(f"║  {name:^52s}║")
    print(f"╠{bar}╣")
    print(f"║  Hry:               {n:<33d}║")
    print(f"║  Výhry / Prehry:    {wins}/{n-wins:<32d}║")
    print(f"║  Miera výhier:      {100*wins/n:.1f}%{'':<30s}║")
    print(f"║  {'─'*50}  ║")
    print(f"║  Najlepšie skóre:   {max(scores):<33d}║")
    print(f"║  Priemerné skóre:   {np.mean(scores):<33.1f}║")
    print(f"║  Mediánové skóre:   {np.median(scores):<33.1f}║")
    print(f"║  Najhoršie skóre:   {min(scores):<33d}║")
    print(f"║  Std. odchýlka:     {np.std(scores):<33.1f}║")
    print(f"║  {'─'*50}  ║")
    print(f"║  Avg. max dlaždica: {np.mean(tiles):<33.1f}║")
    print(f"║  Max. dlaždica:     {max(tiles):<33d}║")
    dist_str = '  '.join(f'{t}×{c}' for t,c in sorted(tile_counts.items()))
    # wrap if too long
    print(f"║  Distribúcia:       {dist_str[:33]:<33s}║")
    print(f"║  {'─'*50}  ║")
    print(f"║  Avg. ťahov/hru:    {np.mean(moves):<33.1f}║")
    print(f"║  Avg. vľavo:        {np.mean([r['moves_left']  for r in records]):<33.1f}║")
    print(f"║  Avg. vpravo:       {np.mean([r['moves_right'] for r in records]):<33.1f}║")
    print(f"║  Avg. hore:         {np.mean([r['moves_up']    for r in records]):<33.1f}║")
    print(f"║  Avg. dole:         {np.mean([r['moves_down']  for r in records]):<33.1f}║")
    print(f"╚{bar}╝\n")

# ─────────────────────────────────────────────────────────────────────────────
# Hlavný beh
# ─────────────────────────────────────────────────────────────────────────────
SOLVER_MAP = {
    'random':     ('Random',     random_solver),
    'heuristic':  ('Heuristic',  heuristic_solver),
    'expectimax': ('ExpectiMax', expectimax_solver),
}

def run_one(key):
    np.random.seed(42); random.seed(42)
    name, fn = SOLVER_MAP[key]
    print(f"\n{'▶'*3}  Spúšťam {name} solver – 30 hier  {'◀'*3}")
    t0 = time.time()
    records  = run_solver(fn, n_games=30, record_history=True)
    elapsed  = time.time() - t0
    print(f"\n  ⏱  Čas: {elapsed:.1f}s ({elapsed/30:.1f}s/hra)")
    print_stats(records, name)
    path = os.path.join(RESULTS_DIR, f'{key}_records.pkl')
    with open(path, 'wb') as f:
        pickle.dump(records, f)
    print(f"  💾  Uložené: {path}")
    return records

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['random', 'heuristic', 'expectimax']
    invalid = [t for t in targets if t not in SOLVER_MAP]
    if invalid:
        print(f"Neznáme solvery: {invalid}. Možnosti: random, heuristic, expectimax")
        sys.exit(1)
    print("=" * 60)
    print("  2048 AI Experiment")
    print("=" * 60)
    for key in targets:
        run_one(key)
    print("\n✅  Hotovo! Spusti generate_visuals.py pre grafy a Excel.")
