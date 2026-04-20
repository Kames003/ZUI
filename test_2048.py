#!/usr/bin/env python3
"""
Comprehensive test suite for 2048 AI project.
Run: python test_2048.py -v

Tests: 37 total
  - 16 unit tests: game mechanics
  -  5 unit tests: evaluation functions
  -  5 unit tests: solvers
  -  7 integration tests
  -  4 statistical / smoke tests
"""
import unittest
import numpy as np
import copy
import random
import os
import tempfile

# ─────────────────────────────────────────────────────────────────────────────
# Inline definitions (zero dependency on notebook)
# ─────────────────────────────────────────────────────────────────────────────

def add_score(sc, val):
    sc += val
    return sc

def move_left(grid, score):
    for i in range(4):
        non_zero = [x for x in grid[i, :] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[i, :] = np.array(non_zero + zero)
        for j in range(3):
            if grid[i, j] == grid[i, j + 1]:
                grid[i, j] *= 2
                score = add_score(score, grid[i, j])
                grid[i, j + 1] = 0
        non_zero = [x for x in grid[i, :] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[i, :] = np.array(non_zero + zero)
    return (grid, score)

def move_right(grid, score):
    for i in range(4):
        non_zero = [x for x in grid[i, :] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[i, :] = np.array(zero + non_zero[::-1])
        for j in range(3, 0, -1):
            if grid[i, j] == grid[i, j - 1]:
                grid[i, j] *= 2
                score = add_score(score, grid[i, j])
                grid[i, j - 1] = 0
        non_zero = [x for x in grid[i, :] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[i, :] = np.array(zero + non_zero[::-1])
    return (grid, score)

def move_up(grid, score):
    for i in range(4):
        non_zero = [x for x in grid[:, i] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[:, i] = np.array(non_zero + zero)
        for j in range(3):
            if grid[j, i] == grid[j + 1, i]:
                grid[j, i] *= 2
                score = add_score(score, grid[j, i])
                grid[j + 1, i] = 0
        non_zero = [x for x in grid[:, i] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[:, i] = np.array(non_zero + zero)
    return (grid, score)

def move_down(grid, score):
    for i in range(4):
        non_zero = [x for x in grid[:, i] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[:, i] = np.array(zero + non_zero[::-1])
        for j in range(3, 0, -1):
            if grid[j, i] == grid[j - 1, i]:
                grid[j, i] *= 2
                score = add_score(score, grid[j, i])
                grid[j - 1, i] = 0
        non_zero = [x for x in grid[:, i] if x != 0]
        zero = [0] * (4 - len(non_zero))
        grid[:, i] = np.array(zero + non_zero[::-1])
    return (grid, score)

def add_new_number(grid):
    zero_indices = np.where(grid == 0)
    if len(zero_indices[0]) == 0:
        return False
    index = np.random.choice(len(zero_indices[0]))
    i, j = zero_indices[0][index], zero_indices[1][index]
    grid[i, j] = 2 if np.random.random() < 0.9 else 4
    return True

def check_game_over(grid):
    if np.all(grid) == False:
        return False
    for row in range(4):
        for col in range(4):
            if row != 3:
                if grid[row, col] == grid[row + 1, col]:
                    return False
            if col != 3:
                if grid[row, col] == grid[row, col + 1]:
                    return False
    return True

def check_win(grid):
    return 2048 in grid

def play_2048(grid, move, score):
    orig_grid = copy.deepcopy(grid)
    if check_game_over(grid):
        raise RuntimeError("GO")
    if move == 'left':
        grid, score = move_left(grid, score)
    elif move == 'right':
        grid, score = move_right(grid, score)
    elif move == 'up':
        grid, score = move_up(grid, score)
    elif move == 'down':
        grid, score = move_down(grid, score)
    else:
        raise ValueError("Invalid move")
    if check_win(grid):
        raise RuntimeError("WIN")
    if not np.array_equal(grid, orig_grid):
        add_new_number(grid)
    return (grid, score)

def new_game():
    score = 0
    grid = np.zeros((4, 4), dtype=int)
    add_new_number(grid)
    add_new_number(grid)
    return (grid, score)

# ─────────────────────────────────────────────────────────────────────────────
# Solver infrastructure
# ─────────────────────────────────────────────────────────────────────────────
MOVES = ['left', 'right', 'up', 'down']
_MOVE_FNS = {'left': move_left, 'right': move_right, 'up': move_up, 'down': move_down}

def apply_move_to_copy(grid, score, move):
    g = copy.deepcopy(grid)
    return _MOVE_FNS[move](g, score)

def is_valid_move(grid, score, move):
    g_new, _ = apply_move_to_copy(grid, score, move)
    return not np.array_equal(g_new, grid)

def empty_game_record():
    return {
        'solver': '', 'score': 0, 'max_tile': 0,
        'move_count': 0, 'moves_left': 0, 'moves_right': 0,
        'moves_up': 0, 'moves_down': 0, 'outcome': '', 'grid_history': [],
    }

def run_solver(solver_fn, n_games=30, record_history=True, verbose=False):
    records = []
    for _ in range(n_games):
        rec = empty_game_record()
        grid, score = new_game()
        if record_history:
            rec['grid_history'].append((copy.deepcopy(grid), score))
        while True:
            move = solver_fn(grid, score)
            if move is None:
                rec['outcome'] = 'loss'
                break
            try:
                grid, score = play_2048(grid, move, score)
                rec[f'moves_{move}'] += 1
                rec['move_count'] += 1
                if record_history:
                    rec['grid_history'].append((copy.deepcopy(grid), score))
            except RuntimeError as e:
                if str(e) == 'WIN':
                    rec['outcome'] = 'win'
                    rec[f'moves_{move}'] += 1
                    rec['move_count'] += 1
                else:
                    rec['outcome'] = 'loss'
                break
        rec['score'] = score
        rec['max_tile'] = int(grid.max())
        records.append(rec)
    return records

# ─────────────────────────────────────────────────────────────────────────────
# Solvers
# ─────────────────────────────────────────────────────────────────────────────
def random_solver(grid, score):
    valid = [m for m in MOVES if is_valid_move(grid, score, m)]
    return random.choice(valid) if valid else None

HEURISTIC_PRIORITY = ['down', 'left', 'up', 'right']

def heuristic_solver(grid, score):
    for move in HEURISTIC_PRIORITY:
        if is_valid_move(grid, score, move):
            return move
    return None

def _log2(x):
    return float(np.log2(x)) if x > 0 else 0.0

def eval_monotonicity(grid):
    log_g = np.vectorize(_log2)(grid)
    total = 0.0
    for r in range(4):
        row = log_g[r, :]
        fwd = sum(max(row[i] - row[i+1], 0) for i in range(3))
        bwd = sum(max(row[i+1] - row[i], 0) for i in range(3))
        total -= min(fwd, bwd)
    for c in range(4):
        col = log_g[:, c]
        fwd = sum(max(col[i] - col[i+1], 0) for i in range(3))
        bwd = sum(max(col[i+1] - col[i], 0) for i in range(3))
        total -= min(fwd, bwd)
    return total

def eval_smoothness(grid):
    log_g = np.vectorize(_log2)(grid)
    penalty = 0.0
    for r in range(4):
        for c in range(4):
            if grid[r, c] == 0:
                continue
            if c < 3 and grid[r, c+1] != 0:
                penalty -= abs(log_g[r, c] - log_g[r, c+1])
            if r < 3 and grid[r+1, c] != 0:
                penalty -= abs(log_g[r, c] - log_g[r+1, c])
    return penalty

def eval_empty(grid):
    return float(np.log2(max(int(np.sum(grid == 0)), 1)))

def eval_corner(grid):
    max_val = int(grid.max())
    if max_val == 0:
        return 0.0
    corners = [int(grid[0,0]), int(grid[0,3]), int(grid[3,0]), int(grid[3,3])]
    return _log2(max_val) if max_val in corners else 0.0

def evaluate(grid):
    return (1.0 * eval_monotonicity(grid) +
            0.1 * eval_smoothness(grid) +
            2.7 * eval_empty(grid) +
            1.0 * eval_corner(grid))

def expectimax(grid, score, depth, is_max_node):
    if depth == 0:
        return evaluate(grid)
    if is_max_node:
        best = -float('inf')
        any_valid = False
        for move in MOVES:
            g_new, s_new = apply_move_to_copy(grid, score, move)
            if np.array_equal(g_new, grid):
                continue
            any_valid = True
            val = expectimax(g_new, s_new, depth - 1, False)
            if val > best:
                best = val
        return best if any_valid else evaluate(grid)
    else:
        empty_pos = list(zip(*np.where(grid == 0)))
        if not empty_pos:
            return evaluate(grid)
        total = 0.0
        for (r, c) in empty_pos:
            for tile_val, prob in ((2, 0.9), (4, 0.1)):
                g_copy = copy.deepcopy(grid)
                g_copy[r, c] = tile_val
                total += prob * expectimax(g_copy, score, depth - 1, True)
        return total / len(empty_pos)

EXPECTIMAX_DEPTH = 3

def expectimax_solver(grid, score):
    best_move, best_val = None, -float('inf')
    for move in MOVES:
        g_new, s_new = apply_move_to_copy(grid, score, move)
        if np.array_equal(g_new, grid):
            continue
        val = expectimax(g_new, s_new, EXPECTIMAX_DEPTH - 1, False)
        if val > best_val:
            best_val, best_move = val, move
    return best_move

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build specific grid states
# ─────────────────────────────────────────────────────────────────────────────
def make_grid(rows):
    return np.array(rows, dtype=int)

def stuck_grid():
    """A grid where no move is possible (game over)."""
    return make_grid([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [4, 2, 4, 2],
    ])


# ═════════════════════════════════════════════════════════════════════════════
# TEST CLASSES
# ═════════════════════════════════════════════════════════════════════════════

class TestMoveMechanics(unittest.TestCase):
    """16 unit tests for core game mechanics."""

    def setUp(self):
        np.random.seed(0)

    # ── move_left ──────────────────────────────────────────────
    def test_move_left_basic(self):
        grid = make_grid([[0, 0, 2, 4],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, _ = move_left(grid, 0)
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[0, 1], 4)

    def test_move_left_merge(self):
        grid = make_grid([[2, 2, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, score = move_left(grid, 0)
        self.assertEqual(result[0, 0], 4)
        self.assertEqual(result[0, 1], 0)
        self.assertEqual(score, 4)

    def test_move_left_no_double_merge(self):
        """[2,2,2,2] → [4,4,0,0], NOT [8,0,0,0]."""
        grid = make_grid([[2, 2, 2, 2],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, score = move_left(grid, 0)
        self.assertEqual(list(result[0, :]), [4, 4, 0, 0])
        self.assertEqual(score, 8)

    def test_move_left_no_change(self):
        """Already left-packed, no merges → unchanged."""
        grid = make_grid([[2, 4, 8, 16],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        original = grid.copy()
        result, score = move_left(grid, 0)
        np.testing.assert_array_equal(result[0, :], original[0, :])
        self.assertEqual(score, 0)

    # ── move_right ─────────────────────────────────────────────
    def test_move_right_basic(self):
        grid = make_grid([[2, 4, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, _ = move_right(grid, 0)
        self.assertEqual(result[0, 3], 4)
        self.assertEqual(result[0, 2], 2)

    def test_move_right_merge(self):
        grid = make_grid([[0, 0, 2, 2],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, score = move_right(grid, 0)
        self.assertEqual(result[0, 3], 4)
        self.assertEqual(score, 4)

    # ── move_up ────────────────────────────────────────────────
    def test_move_up_basic(self):
        grid = make_grid([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [2, 0, 0, 0],
                          [4, 0, 0, 0]])
        result, _ = move_up(grid, 0)
        self.assertEqual(result[0, 0], 2)
        self.assertEqual(result[1, 0], 4)

    def test_move_up_merge(self):
        grid = make_grid([[2, 0, 0, 0],
                          [2, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, score = move_up(grid, 0)
        self.assertEqual(result[0, 0], 4)
        self.assertEqual(result[1, 0], 0)
        self.assertEqual(score, 4)

    # ── move_down ──────────────────────────────────────────────
    def test_move_down_basic(self):
        grid = make_grid([[2, 0, 0, 0],
                          [4, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        result, _ = move_down(grid, 0)
        self.assertEqual(result[3, 0], 4)
        self.assertEqual(result[2, 0], 2)

    def test_move_down_merge(self):
        grid = make_grid([[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [2, 0, 0, 0],
                          [2, 0, 0, 0]])
        result, score = move_down(grid, 0)
        self.assertEqual(result[3, 0], 4)
        self.assertEqual(score, 4)

    # ── score ──────────────────────────────────────────────────
    def test_score_increments_on_merge(self):
        grid = make_grid([[4, 4, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        _, score = move_left(grid, 100)
        self.assertEqual(score, 108)  # 100 + 8

    # ── check_game_over ────────────────────────────────────────
    def test_check_game_over_full_no_merge(self):
        self.assertTrue(check_game_over(stuck_grid()))

    def test_check_game_over_has_empty(self):
        grid = make_grid([[2, 4, 2, 4],
                          [4, 2, 4, 2],
                          [2, 4, 2, 4],
                          [4, 2, 4, 0]])  # one zero
        self.assertFalse(check_game_over(grid))

    def test_check_game_over_merge_possible(self):
        grid = make_grid([[2, 4, 2, 4],
                          [4, 2, 4, 2],
                          [2, 4, 2, 4],
                          [4, 2, 4, 4]])  # last two cols mergeable
        self.assertFalse(check_game_over(grid))

    # ── check_win ──────────────────────────────────────────────
    def test_check_win_with_2048(self):
        grid = make_grid([[2048, 4, 2, 4],
                          [4, 2, 4, 2],
                          [2, 4, 2, 4],
                          [4, 2, 4, 2]])
        self.assertTrue(check_win(grid))

    def test_check_win_without_2048(self):
        grid = make_grid([[1024, 4, 2, 4],
                          [4, 2, 4, 2],
                          [2, 4, 2, 4],
                          [4, 2, 4, 2]])
        self.assertFalse(check_win(grid))

    # ── add_new_number ─────────────────────────────────────────
    def test_add_new_number_places_tile(self):
        grid = np.zeros((4, 4), dtype=int)
        result = add_new_number(grid)
        self.assertTrue(result)
        self.assertEqual(int(np.sum(grid != 0)), 1)
        tile = int(grid[grid != 0][0])
        self.assertIn(tile, (2, 4))

    def test_add_new_number_full_board(self):
        grid = make_grid([[2, 4, 8, 16],
                          [32, 64, 128, 256],
                          [512, 1024, 2, 4],
                          [8, 16, 32, 64]])
        result = add_new_number(grid)
        self.assertFalse(result)

    def test_new_game_exactly_two_tiles(self):
        grid, score = new_game()
        self.assertEqual(score, 0)
        self.assertEqual(int(np.sum(grid != 0)), 2)
        self.assertEqual(grid.shape, (4, 4))

    # ── play_2048 ──────────────────────────────────────────────
    def test_play_2048_raises_go(self):
        with self.assertRaises(RuntimeError) as ctx:
            play_2048(stuck_grid(), 'left', 0)
        self.assertEqual(str(ctx.exception), 'GO')

    def test_play_2048_raises_win(self):
        """A move that creates 2048 should raise WIN."""
        grid = make_grid([[1024, 1024, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        with self.assertRaises(RuntimeError) as ctx:
            play_2048(grid, 'left', 0)
        self.assertEqual(str(ctx.exception), 'WIN')


class TestEvaluationFunctions(unittest.TestCase):
    """5 unit tests for ExpectiMax evaluation components."""

    def test_eval_empty_correct_count(self):
        grid = make_grid([[2, 0, 4, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 2],
                          [4, 0, 0, 0]])
        # zeros: row0=(1,3), row1=all4, row2=(0,1,2), row3=(1,2,3) → 12 zeros
        result = eval_empty(grid)
        self.assertAlmostEqual(result, np.log2(12), places=5)

    def test_eval_corner_bonus_max_in_corner(self):
        grid = make_grid([[1024, 4, 2, 4],
                          [4, 2, 4, 2],
                          [2, 4, 2, 4],
                          [4, 2, 4, 2]])
        bonus = eval_corner(grid)
        self.assertGreater(bonus, 0)
        self.assertAlmostEqual(bonus, np.log2(1024), places=5)

    def test_eval_corner_bonus_max_not_in_corner(self):
        grid = make_grid([[2, 4, 2, 4],
                          [4, 1024, 4, 2],  # max in center
                          [2, 4, 2, 4],
                          [4, 2, 4, 2]])
        self.assertEqual(eval_corner(grid), 0.0)

    def test_eval_monotonicity_perfect_descending_row(self):
        """A perfectly descending row should have 0 (or very small) penalty."""
        grid = make_grid([[16, 8, 4, 2],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
        mono = eval_monotonicity(grid)
        # Only row 0 is non-zero and perfectly monotone (descending)
        # Other rows contribute 0 penalty since all zeros
        # mono should be 0 for row 0 (no penalty in best direction)
        self.assertGreaterEqual(mono, -0.01)

    def test_eval_monotonicity_chaotic_lower_than_ordered(self):
        """Chaotic board should score lower than ordered board."""
        ordered = make_grid([[16, 8, 4, 2],
                             [8, 4, 2, 1],
                             [4, 2, 1, 0],
                             [2, 1, 0, 0]])
        chaotic = make_grid([[2, 16, 4, 8],
                             [8, 1, 2, 4],
                             [4, 16, 8, 2],
                             [2, 4, 1, 8]])
        self.assertGreater(eval_monotonicity(ordered), eval_monotonicity(chaotic))


class TestSolvers(unittest.TestCase):
    """5 unit tests for solver behavior."""

    def setUp(self):
        np.random.seed(1)
        random.seed(1)

    def test_random_solver_returns_valid_move(self):
        grid, score = new_game()
        move = random_solver(grid, score)
        self.assertIn(move, MOVES)
        # Move must actually change the board
        g_new, _ = apply_move_to_copy(grid, score, move)
        self.assertFalse(np.array_equal(g_new, grid))

    def test_random_solver_returns_none_on_stuck_board(self):
        result = random_solver(stuck_grid(), 0)
        self.assertIsNone(result)

    def test_heuristic_solver_picks_best_eval_move(self):
        """Greedy 1-ply: heuristic must pick the move with the highest evaluate() score."""
        grid, score = new_game()
        move = heuristic_solver(grid, score)
        # Verify chosen move achieves the maximum eval among all valid moves
        best_val = -float('inf')
        for m in MOVES:
            if not is_valid_move(grid, score, m):
                continue
            g_new, _ = apply_move_to_copy(grid, score, m)
            val = evaluate(g_new)
            if val > best_val:
                best_val = val
        chosen_grid, _ = apply_move_to_copy(grid, score, move)
        self.assertAlmostEqual(evaluate(chosen_grid), best_val, places=5)

    def test_expectimax_solver_returns_move_on_normal_board(self):
        grid, score = new_game()
        move = expectimax_solver(grid, score)
        self.assertIn(move, MOVES)

    def test_expectimax_solver_returns_none_on_stuck_board(self):
        result = expectimax_solver(stuck_grid(), 0)
        self.assertIsNone(result)


class TestIntegration(unittest.TestCase):
    """7 integration tests for run_solver and data pipeline."""

    def setUp(self):
        np.random.seed(2)
        random.seed(2)
        self.records = run_solver(random_solver, n_games=5,
                                  record_history=True, verbose=False)

    def test_run_solver_returns_n_records(self):
        records = run_solver(random_solver, n_games=10,
                             record_history=False, verbose=False)
        self.assertEqual(len(records), 10)

    def test_all_records_have_valid_outcome(self):
        for rec in self.records:
            self.assertIn(rec['outcome'], ('win', 'loss'))

    def test_move_counts_sum_to_total(self):
        for rec in self.records:
            total_dir = (rec['moves_left'] + rec['moves_right'] +
                         rec['moves_up'] + rec['moves_down'])
            self.assertEqual(total_dir, rec['move_count'])

    def test_history_length_is_move_count_plus_one(self):
        """History starts with initial state, then one entry per move."""
        for rec in self.records:
            if rec['outcome'] == 'win':
                # On win the move IS counted but history also includes the win state
                expected = rec['move_count'] + 1
            else:
                expected = rec['move_count'] + 1
            self.assertEqual(len(rec['grid_history']), expected)

    def test_max_tile_matches_actual_final_grid(self):
        """rec['max_tile'] must equal the max in the final recorded grid."""
        for rec in self.records:
            if rec['grid_history']:
                final_grid, _ = rec['grid_history'][-1]
                self.assertEqual(rec['max_tile'], int(final_grid.max()))

    def test_excel_export_creates_file(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest("pandas not installed")
        records = run_solver(random_solver, n_games=3,
                             record_history=False, verbose=False)

        def to_df(recs, name):
            return pd.DataFrame([{
                'game': i+1, 'solver': name,
                'score': r['score'], 'max_tile': r['max_tile'],
                'move_count': r['move_count'], 'outcome': r['outcome'],
            } for i, r in enumerate(recs)])

        df = to_df(records, 'Random')
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            tmp_path = f.name
        try:
            with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Random', index=False)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 0)
        finally:
            os.unlink(tmp_path)

    def test_excel_has_correct_sheet_names(self):
        try:
            import pandas as pd
            import openpyxl
        except ImportError:
            self.skipTest("pandas/openpyxl not installed")
        records = run_solver(random_solver, n_games=2,
                             record_history=False, verbose=False)

        def to_df(recs, name):
            return pd.DataFrame([{'game': i+1, 'solver': name,
                                   'score': r['score'], 'outcome': r['outcome']}
                                  for i, r in enumerate(recs)])

        df = to_df(records, 'Random')
        summary = pd.DataFrame([{'solver': 'Random', 'wins': 0}])

        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            tmp_path = f.name
        try:
            with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
                summary.to_excel(writer, sheet_name='Summary', index=False)
                df.to_excel(writer, sheet_name='Random', index=False)
                df.to_excel(writer, sheet_name='Heuristic', index=False)
                df.to_excel(writer, sheet_name='ExpectiMax', index=False)
                df.to_excel(writer, sheet_name='All_Games', index=False)
            wb = openpyxl.load_workbook(tmp_path)
            self.assertEqual(set(wb.sheetnames),
                             {'Summary', 'Random', 'Heuristic', 'ExpectiMax', 'All_Games'})
        finally:
            os.unlink(tmp_path)


class TestSmoke(unittest.TestCase):
    """4 statistical / smoke tests."""

    def test_random_30_games_no_crash(self):
        np.random.seed(10)
        random.seed(10)
        records = run_solver(random_solver, n_games=30,
                             record_history=False, verbose=False)
        self.assertEqual(len(records), 30)
        for r in records:
            self.assertGreater(r['score'], 0)
            self.assertGreater(r['max_tile'], 0)

    def test_heuristic_30_games_no_crash(self):
        np.random.seed(11)
        records = run_solver(heuristic_solver, n_games=30,
                             record_history=False, verbose=False)
        self.assertEqual(len(records), 30)
        for r in records:
            self.assertIn(r['outcome'], ('win', 'loss'))

    def test_expectimax_5_games_no_crash(self):
        np.random.seed(12)
        records = run_solver(expectimax_solver, n_games=5,
                             record_history=False, verbose=False)
        self.assertEqual(len(records), 5)
        for r in records:
            self.assertGreater(r['score'], 0)

    def test_expectimax_beats_random_on_average(self):
        """ExpectiMax should score higher than Random on average over 10 games."""
        np.random.seed(99)
        random.seed(99)
        rand_records = run_solver(random_solver,     n_games=10,
                                  record_history=False, verbose=False)
        np.random.seed(99)
        random.seed(99)
        em_records   = run_solver(expectimax_solver, n_games=10,
                                  record_history=False, verbose=False)
        avg_rand = np.mean([r['score'] for r in rand_records])
        avg_em   = np.mean([r['score'] for r in em_records])
        self.assertGreater(avg_em, avg_rand,
                           msg=f"ExpectiMax avg {avg_em:.0f} should beat Random avg {avg_rand:.0f}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    loader  = unittest.TestLoader()
    suite   = unittest.TestSuite()
    for cls in [TestMoveMechanics, TestEvaluationFunctions,
                TestSolvers, TestIntegration, TestSmoke]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    total  = result.testsRun
    failed = len(result.failures) + len(result.errors)
    print(f"\n{'='*60}")
    print(f"TOTAL: {total} tests | PASSED: {total - failed} | FAILED: {failed}")
    sys.exit(0 if failed == 0 else 1)
