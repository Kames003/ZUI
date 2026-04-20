#!/usr/bin/env python3
"""
Rýchly tuning skript pre ExpectiMax eval váhy a depth parametre.
Spúšťa N_GAMES hier pre každú konfiguráciu a reportuje výsledky.

Použitie:
  python3 tune_expectimax.py          # testuje všetky configs nižšie
  python3 tune_expectimax.py --quick  # len 3 hry na konfiguráciu
"""
import sys, copy, random, time
import numpy as np

N_GAMES = 5     # hier na konfiguráciu (rýchle: 5 ≈ 3-4 min)
SEED    = 42    # pevný seed = porovnateľné výsledky

# ─── Herná logika (verbatim z 2048_base.ipynb) ───────────────────────────────

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

def play_2048(grid, move, score):
    orig = copy.deepcopy(grid)
    if check_game_over(grid): raise RuntimeError("GO")
    fns = {'left': move_left, 'right': move_right, 'up': move_up, 'down': move_down}
    grid, score = fns[move](grid, score)
    if 2048 in grid: raise RuntimeError("WIN")
    if not np.array_equal(grid, orig): add_new_number(grid)
    return grid, score

def new_game():
    grid = np.zeros((4,4), dtype=int)
    add_new_number(grid); add_new_number(grid)
    return grid, 0

MOVES = ['left', 'right', 'up', 'down']
_FNS  = {'left': move_left, 'right': move_right, 'up': move_up, 'down': move_down}

def apply_copy(grid, score, move):
    g = copy.deepcopy(grid)
    return _FNS[move](g, score)

def is_valid(grid, score, move):
    g2, _ = apply_copy(grid, score, move)
    return not np.array_equal(g2, grid)

# ─── Eval funkcia (konfigurovateľná) ─────────────────────────────────────────

def _log2(x): return float(np.log2(x)) if x > 0 else 0.0

def _monotonicity(grid):
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

def _smoothness(grid):
    lg = np.vectorize(_log2)(grid); p = 0.0
    for r in range(4):
        for c in range(4):
            if grid[r,c] == 0: continue
            if c < 3 and grid[r,c+1]: p -= abs(lg[r,c]-lg[r,c+1])
            if r < 3 and grid[r+1,c]: p -= abs(lg[r,c]-lg[r+1,c])
    return p

def _empty(grid):
    return float(np.log2(max(int(np.sum(grid==0)), 1)))

def _corner(grid, mode='any'):
    """
    mode='any'  : bonus ak max je v ľubovoľnom rohu
    mode='bl'   : plný bonus pre bottom-left, čiastočný pre ostatné rohy
    mode='snake': bonus proporcionálny k snake-weight pozícii max dlaždice
    """
    mv = int(grid.max())
    if mv == 0: return 0.0
    lm = _log2(mv)
    if mode == 'any':
        corners = [int(grid[0,0]),int(grid[0,3]),int(grid[3,0]),int(grid[3,3])]
        return lm if mv in corners else 0.0
    elif mode == 'bl':
        if int(grid[3,0]) == mv: return lm * 1.5
        corners = [int(grid[0,0]),int(grid[0,3]),int(grid[3,3])]
        return lm * 0.5 if mv in corners else 0.0
    elif mode == 'snake':
        # Snake weight matrix – bottom-left corner highest
        SNAKE = np.array([
            [2,    4,    8,   16],
            [256, 128,  64,   32],
            [512,1024,2048, 4096],
            [65536,32768,16384,8192]], dtype=float)
        SNAKE /= SNAKE.max()
        r_idx, c_idx = np.unravel_index(np.argmax(grid), grid.shape)
        return lm * SNAKE[r_idx, c_idx]

def make_evaluate(w_mono, w_smooth, w_empty, w_corner, corner_mode='any'):
    def evaluate(grid):
        return (w_mono   * _monotonicity(grid)
              + w_smooth * _smoothness(grid)
              + w_empty  * _empty(grid)
              + w_corner * _corner(grid, corner_mode))
    return evaluate

# ─── ExpectiMax (s konfigurovateľnou eval + depth) ───────────────────────────

def make_expectimax_solver(evaluate, base_depth=3, critical_depth=3, critical_threshold=0):
    def expectimax(grid, score, depth, is_max):
        if depth == 0: return evaluate(grid)
        if is_max:
            best = -float('inf'); any_v = False
            for m in MOVES:
                g2,s2 = apply_copy(grid,score,m)
                if np.array_equal(g2,grid): continue
                any_v = True
                v = expectimax(g2,s2,depth-1,False)
                if v > best: best = v
            return best if any_v else evaluate(grid)
        else:
            ep = list(zip(*np.where(grid==0)))
            if not ep: return evaluate(grid)
            tot = 0.0
            for (r,c) in ep:
                for tv,prob in ((2,0.9),(4,0.1)):
                    gc = copy.deepcopy(grid); gc[r,c]=tv
                    tot += prob * expectimax(gc,score,depth-1,True)
            return tot / len(ep)

    def solver(grid, score):
        n_empty = int(np.sum(grid==0))
        depth = critical_depth if (critical_threshold > 0 and n_empty <= critical_threshold) else base_depth
        best_m, best_v = None, -float('inf')
        for m in MOVES:
            g2,s2 = apply_copy(grid,score,m)
            if np.array_equal(g2,grid): continue
            v = expectimax(g2,s2,depth-1,False)
            if v > best_v: best_v,best_m = v,m
        return best_m
    return solver

# ─── Runner ──────────────────────────────────────────────────────────────────

def run_games(solver, n=N_GAMES, seed=SEED):
    np.random.seed(seed); random.seed(seed)
    scores, tiles, outcomes = [], [], []
    for gi in range(n):
        grid, score = new_game()
        outcome = 'loss'
        while True:
            move = solver(grid, score)
            if move is None: break
            try:
                grid, score = play_2048(grid, move, score)
            except RuntimeError as e:
                if str(e) == 'WIN': outcome = 'win'
                break
        scores.append(score)
        tiles.append(int(grid.max()))
        outcomes.append(outcome)
    wins = outcomes.count('win')
    return {
        'avg':    int(np.mean(scores)),
        'med':    int(np.median(scores)),
        'best':   max(scores),
        'worst':  min(scores),
        'wins':   wins,
        'win_pct': round(100*wins/n, 1),
        'avg_tile': int(np.mean(tiles)),
        'tiles':  tiles,
        'scores': scores,
    }

def print_result(name, r, elapsed):
    tile_dist = {}
    for t in r['tiles']: tile_dist[t] = tile_dist.get(t,0)+1
    dist_str = '  '.join(f"{k}×{v}" for k,v in sorted(tile_dist.items()))
    print(f"  {name:<40s} | avg={r['avg']:6d}  wins={r['wins']}/{N_GAMES} ({r['win_pct']:5.1f}%)  "
          f"best={r['best']:6d}  worst={r['worst']:6d}  tiles=[{dist_str}]  ({elapsed:.0f}s)")

# ─── Konfigurácie na testovanie ───────────────────────────────────────────────

CONFIGS = [
    # Názov,  w_mono, w_smooth, w_empty, w_corner, corner_mode, base_d, crit_d, crit_thresh
    ("baseline  mono=1.0 sm=0.1 em=2.7 co=1.0 any  d=3",    1.0, 0.1, 2.7, 1.0, 'any',   3, 3, 0),
    ("corner_bl mono=1.0 sm=0.1 em=2.7 co=1.0 bl   d=3",    1.0, 0.1, 2.7, 1.0, 'bl',    3, 3, 0),
    ("snake     mono=1.0 sm=0.1 em=2.7 co=1.5 snake d=3",   1.0, 0.1, 2.7, 1.5, 'snake', 3, 3, 0),
    ("depth4@4  mono=1.0 sm=0.1 em=2.7 co=1.0 any  d=3/4",  1.0, 0.1, 2.7, 1.0, 'any',   3, 4, 4),
    ("depth4@2  mono=1.0 sm=0.1 em=2.7 co=1.0 any  d=3/4",  1.0, 0.1, 2.7, 1.0, 'any',   3, 4, 2),
    ("mono+     mono=1.5 sm=0.1 em=2.7 co=1.0 any  d=3",    1.5, 0.1, 2.7, 1.0, 'any',   3, 3, 0),
    ("empty+    mono=1.0 sm=0.1 em=3.5 co=1.0 any  d=3",    1.0, 0.1, 3.5, 1.0, 'any',   3, 3, 0),
    ("bl+depth4 mono=1.0 sm=0.1 em=2.7 co=1.0 bl   d=3/4",  1.0, 0.1, 2.7, 1.0, 'bl',    3, 4, 4),
]

if __name__ == '__main__':
    quick = '--quick' in sys.argv
    N_GAMES = 3 if quick else N_GAMES
    if quick:
        print(f"QUICK mode: {N_GAMES} games per config\n")
    else:
        print(f"Tuning: {N_GAMES} games per config, seed={SEED}\n")

    print(f"  {'Config':<40s} | {'avg':>6}  {'wins':>8}  {'best':>6}  {'worst':>6}  tiles  (time)")
    print("  " + "-"*100)

    results = []
    for cfg in CONFIGS:
        name, w_m, w_s, w_e, w_c, c_mode, bd, cd, ct = cfg
        ev = make_evaluate(w_m, w_s, w_e, w_c, c_mode)
        solver = make_expectimax_solver(ev, bd, cd, ct)
        t0 = time.time()
        r = run_games(solver)
        elapsed = time.time() - t0
        print_result(name, r, elapsed)
        results.append((r['avg'], r['win_pct'], name))

    print("\n  === Ranking podľa avg score ===")
    for avg, wp, name in sorted(results, reverse=True):
        print(f"  avg={avg:6d}  wins={wp:5.1f}%  {name}")
