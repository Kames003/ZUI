#!/usr/bin/env python3
"""
Vytvorí 2048_submission.zip so správnou adresárovou štruktúrou.
Spustenie: python3 create_zip.py
"""
import zipfile, os, shutil

ROOT    = '/Users/admin/Desktop/ZUI'
OUT_ZIP = os.path.join(ROOT, '2048_submission.zip')

files = [
    # (zdrojová cesta, cesta v ZIP)
    # ── Report ──────────────────────────────────────────────────────
    ('REPORT.md',                              'REPORT.md'),

    # ── Grafy a animácie ─────────────────────────────────────────────
    ('dashboard.png',                          'images/dashboard.png'),
    ('score_distribution.png',                 'images/score_distribution.png'),
    ('win_loss.png',                           'images/win_loss.png'),
    ('max_tile_distribution.png',              'images/max_tile_distribution.png'),
    ('score_progression.png',                  'images/score_progression.png'),
    ('move_directions.png',                    'images/move_directions.png'),
    ('replay_random.gif',                      'images/replay_random.gif'),
    ('replay_heuristic.gif',                   'images/replay_heuristic.gif'),
    ('replay_expectimax.gif',                  'images/replay_expectimax.gif'),

    # ── Kód ──────────────────────────────────────────────────────────
    ('2048_ai.ipynb',                          'code/2048_ai.ipynb'),
    ('create_notebook.py',                     'code/create_notebook.py'),
    ('tune_expectimax.py',                     'code/tune_expectimax.py'),
    ('test_2048.py',                           'code/test_2048.py'),

    # ── Dáta ─────────────────────────────────────────────────────────
    ('2048_results.xlsx',                      'data/2048_results.xlsx'),

    # ── Dokumentácia ─────────────────────────────────────────────────
    ('EXPERIMENT_LOG.md',                      'docs/EXPERIMENT_LOG.md'),
    ('ZADANIE_KONTEXT.md',                     'docs/ZADANIE_KONTEXT.md'),

    # ── Archív behov ─────────────────────────────────────────────────
    ('archive/run_beh2_seed_contamination/win_loss.png',
                                               'archive/beh2_seed_contamination/win_loss.png'),
    ('archive/run_beh2_seed_contamination/dashboard.png',
                                               'archive/beh2_seed_contamination/dashboard.png'),
    ('archive/run_beh3_seed_fix_threshold2/win_loss.png',
                                               'archive/beh3_final/win_loss.png'),
    ('archive/run_beh3_seed_fix_threshold2/dashboard.png',
                                               'archive/beh3_final/dashboard.png'),
    ('archive/run_beh4_bl_depth4/win_loss.png',
                                               'archive/beh4_bl_depth4/win_loss.png'),
    ('archive/run_beh4_bl_depth4/dashboard.png',
                                               'archive/beh4_bl_depth4/dashboard.png'),
    ('archive/run_beh5_split_eval_empty35/win_loss.png',
                                               'archive/beh5_split_eval/win_loss.png'),
    ('archive/run_beh5_split_eval_empty35/dashboard.png',
                                               'archive/beh5_split_eval/dashboard.png'),
]

with zipfile.ZipFile(OUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:
    for src, dst in files:
        full_src = os.path.join(ROOT, src)
        if os.path.exists(full_src):
            zf.write(full_src, dst)
            print(f"  ✓  {dst}")
        else:
            print(f"  ✗  CHÝBA: {src}")

size_mb = os.path.getsize(OUT_ZIP) / 1024 / 1024
print(f"\nHotovo: {OUT_ZIP}")
print(f"Veľkosť: {size_mb:.1f} MB")
print(f"Súborov: {len(zf.namelist())}")
