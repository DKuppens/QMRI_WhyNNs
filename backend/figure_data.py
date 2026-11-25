"""
backend/figure_data.py
======================
Persist the exact numbers behind every figure, so a figure can be re-drawn
later (different font size, legend position, colours, labels, ...) WITHOUT
re-running the experiment and without changing a single plotted value.

Layout
------
For a figure saved at ``<out>/<name>.png`` the data lives in

    <out>/data/<name>/
        *.csv        tidy tables (one row per plotted value / sample)
        arrays.npz   bulky raw arrays (only for some figures)
        meta.json    settings, parameter names, SNR, seeds, column notes

Floats are written with ``repr`` (shortest string that round-trips), so
reloading gives bit-identical float64 values.

Re-drawing example
------------------
    from backend import figure_data as fd
    d   = 'figures/IVIM/data/H1_local_minima_rmse'
    t   = fd.load_table(d, 'rmse_summary')       # dict: column -> np.ndarray
    meta = fd.load_meta(d)
    sel = t['parameter'] == meta['param_names'][0]
    plt.semilogx(t['n_restarts'][sel], t['line_rep0'][sel])
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

DATA_SUBDIR = 'data'


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def data_dir_for(fig_path: str) -> str:
    """
    Folder holding the data of the figure saved at ``fig_path``
    (``<dir>/<stem>.png`` -> ``<dir>/data/<stem>/``). Created if missing.
    The extension is optional, so a bare stem also works.
    """
    folder, fname = os.path.split(str(fig_path))
    stem = os.path.splitext(fname)[0]
    d = os.path.join(folder, DATA_SUBDIR, stem)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _cell(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v))          # 'nan' / 'inf' round-trip via float()
    return str(v)


def _is_scalar(v) -> bool:
    return isinstance(v, (str, bytes)) or np.ndim(v) == 0


def save_table(d: str, name: str, columns: dict) -> str:
    """
    Write ``columns`` (name -> sequence) to ``<d>/<name>.csv``.
    All sequences must have equal length; scalars are broadcast.
    """
    lengths = {len(v) for v in columns.values() if not _is_scalar(v)}
    if len(lengths) > 1:
        raise ValueError(f"save_table({name}): unequal column lengths "
                         f"{ {k: (None if _is_scalar(v) else len(v)) for k, v in columns.items()} }")
    n = lengths.pop() if lengths else 1
    cols = {k: ([v] * n if _is_scalar(v) else list(v)) for k, v in columns.items()}

    path = os.path.join(d, f'{name}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(list(cols.keys()))
        for row in zip(*cols.values()):
            w.writerow([_cell(x) for x in row])
    return path


def save_arrays(d: str, name: str = 'arrays', **arrays) -> str:
    """Write bulky raw arrays to ``<d>/<name>.npz`` (lossless)."""
    path = os.path.join(d, f'{name}.npz')
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    return path


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def save_meta(d: str, **meta) -> str:
    """Write settings / labels / notes to ``<d>/meta.json``."""
    path = os.path.join(d, 'meta.json')
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    return path


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _convert(vals):
    try:
        return np.array([int(v) for v in vals])
    except ValueError:
        pass
    try:
        return np.array([float(v) for v in vals])
    except ValueError:
        return np.array(vals)


def load_table(d: str, name: str) -> dict:
    """Read ``<d>/<name>.csv`` -> dict of column name -> numpy array
    (int, float or str, whichever the column parses as)."""
    with open(os.path.join(d, f'{name}.csv'), newline='') as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    return {h: _convert([r[i] for r in body]) for i, h in enumerate(header)}


def load_arrays(d: str, name: str = 'arrays') -> dict:
    with np.load(os.path.join(d, f'{name}.npz'), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def load_meta(d: str) -> dict:
    with open(os.path.join(d, 'meta.json')) as f:
        return json.load(f)


def grid2d_from_table(t: dict, param_x: str, param_y: str):
    """
    Rebuild one 2D loss slice from a ``loss_2d`` style table (columns param_x,
    param_y, ix, iy, x, y, loss; rows iy-outer / ix-inner).
    Returns (grid_x, grid_y, loss) with loss.shape == (len(grid_y), len(grid_x)).
    """
    m = (t['param_x'] == param_x) & (t['param_y'] == param_y)
    nx = int(t['ix'][m].max()) + 1
    ny = int(t['iy'][m].max()) + 1
    return t['x'][m][:nx], t['y'][m][::nx], t['loss'][m].reshape(ny, nx)