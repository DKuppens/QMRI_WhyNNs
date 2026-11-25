import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from backend.QMRI.model_registry import get_model_config
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.experiment_utils import signals_to_array
from backend.plot_style import (
    apply_style, COLORS, LEGEND_SMALL, param_label, param_unit, with_unit,
    outdir, save_fig, snr_colormap,
)
from backend import figure_data as fd

# --- tunables -------------------------------------------------------------
SNR_LEVELS      = [5, 10, 20, 50]   # ascending -> increasing SNR (decreasing
                                     # noise), matching the ordering used in
                                     # H2_noise_dominated_minimum.py
N_REALIZATIONS  = 50                # noisy repeats of the same GT per SNR
N_RESTARTS      = 15                # LSQ restarts per realization
N_TRAIN_NN      = 100000              # NN training samples (one network per SNR,
                                     # trained once and shared by all cases)
GRID_RES        = 60                # background (near-noiseless) landscape
                                     # grid resolution
GRID_RES_SNR    = 35                # grid resolution for the per-SNR 1D
                                     # diagonal-panel curves (coarser than
                                     # GRID_RES since only a single curve is
                                     # drawn from each, not a filled surface)
N_COLOR_LEVELS  = 20                # number of distinguishable filled-
                                     # contour color bands for the
                                     # background, log-spaced to match the
                                     # LogNorm color scale. NOTE:
                                     # contourf/contour's `levels=N`
                                     # shorthand spaces levels *linearly* in
                                     # data value even under a LogNorm color
                                     # scale, crowding almost all levels
                                     # into the high-loss end and leaving
                                     # only 2-3 visually different colors
                                     # over the low-loss region that
                                     # actually matters. Levels are
                                     # generated explicitly with
                                     # np.logspace below to fix this.
N_ISO_LINES     = 15                 # number of labeled isolines on the
                                     # background, also log-spaced
N_CASES_TO_PLOT = 20                # number of ground-truth cases to
                                     # auto-select and render (each saved
                                     # to its own file), so you can pick
                                     # the clearest one yourself, as in
                                     # H1_loss_landscape.py. The NNs are
                                     # trained once per SNR (not per case),
                                     # so the per-case cost is dominated by
                                     # the LSQ restart fits and the
                                     # landscape grids.
# ---------------------------------------------------------------------------

# drift-path line style per NN variant (LSQ's is dashed)
PATH_LS = {'sup': ':', 'self': '-.'}


# ------------------------------------------------------------------ #
# Core numerics
# ------------------------------------------------------------------ #

def residual(params, signal, xvalues, cfg):
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def best_of_restarts(signal, xvalues, bounds, n_restarts, cfg, rng):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    best_params, best_res = None, np.inf
    for _ in range(n_restarts):
        init = rng.uniform(lo, hi)
        p = cfg.fit_lsq_1d(xvalues, signal, init)
        r = residual(p, signal, xvalues, cfg)
        if r < best_res:
            best_res, best_params = r, p.copy()
    return best_params


def compute_loss_grid(fixed_params, i, j, grid_i, grid_j, xvalues, signal, cfg):
    xvalues_t = torch.tensor(xvalues, dtype=torch.float64)
    signal_t  = torch.tensor(signal,  dtype=torch.float64)
    loss_grid = np.empty((len(grid_j), len(grid_i)))
    base = list(fixed_params)
    for gi, vi in enumerate(grid_i):
        base[i] = vi
        for gj, vj in enumerate(grid_j):
            base[j] = vj
            signal_pred = cfg.parameters_to_signal(
                *[torch.tensor(float(v)) for v in base], xvalues_t
            )
            loss_grid[gj, gi] = float(torch.sum((signal_t - signal_pred) ** 2))
    return loss_grid


def compute_loss_1d(fixed_params, p, grid_p, xvalues, signal, cfg):
    xvalues_t = torch.tensor(xvalues, dtype=torch.float64)
    signal_t  = torch.tensor(signal,  dtype=torch.float64)
    base = list(fixed_params)
    loss_1d = np.empty(len(grid_p))
    for gi, v in enumerate(grid_p):
        base[p] = v
        signal_pred = cfg.parameters_to_signal(
            *[torch.tensor(float(x)) for x in base], xvalues_t)
        loss_1d[gi] = float(torch.sum((signal_t - signal_pred) ** 2))
    return loss_1d


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run_experiment(model_name='DCE', seed=7, case_idx=None, n_cases=N_CASES_TO_PLOT):
    apply_style()
    cfg = get_model_config(model_name)
    rng = np.random.default_rng(seed=seed)

    print(f"\n=== H2 anecdote — Noise-induced shift | model={model_name} ===")

    case_idx_list = list(case_idx) if case_idx is not None else list(range(n_cases))
    n_pool = max(max(case_idx_list) + 1, n_cases)
    params_cand, _ = cfg.sample_parameters(n_pool, cfg.param_distribution)

    out_dir = outdir(model_name)

    lo = np.array([b[0] for b in cfg.bounds])
    hi = np.array([b[1] for b in cfg.bounds])
    worst_snr = min(SNR_LEVELS)

    # --- NN: train ONE network per SNR level, once, before the case loop ---
    # (as in H2_noise_dominated_minimum.py: trained on data simulated at the
    # SNR it is evaluated at). The training distribution does not depend on
    # the ground truth of any case, so all cases at a given SNR share the
    # same network. torch is seeded per SNR so the training is reproducible
    # given --seed (parameter sampling / noise simulation below are not
    # seeded, as before). Each SNR gets one supervised and one
    # self-supervised NN (train_variants).
    models = {}
    for snr in SNR_LEVELS:
        print(f"\nTraining NN for SNR={snr} ({N_TRAIN_NN} samples)...")
        params_train, coeffs_train = cfg.sample_parameters(N_TRAIN_NN, cfg.param_distribution)
        signals_train = cfg.simulate_signals(params_train, cfg.xvalues, snr)
        models[snr] = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg,
                                     seed=seed + int(snr))

    case_results = []
    for rank, case in enumerate(case_idx_list):
        print(f"\n--- Case {rank + 1}/{len(case_idx_list)}: candidate #{case} ---")

        true_params = params_cand[case]
        true_params_batch = true_params[None, :]
        print(f"Ground truth: "
              f"{dict(zip(cfg.param_names, np.round(true_params, 4)))}")

        # near-noiseless reference signal, for the background landscape
        noiseless_signal_obj = cfg.simulate_signals(true_params_batch, cfg.xvalues, np.inf)
        noiseless_signal = signals_to_array(noiseless_signal_obj)[0]

        lsq_by_snr = {}
        nn_by_snr  = {v['key']: {} for v in NN_VARIANTS}
        rep_signal_by_snr = {}   # one representative real noisy signal per
                                  # SNR, used to draw that SNR's own 1D dip

        for snr in SNR_LEVELS:
            print(f"\nSNR={snr}...")
            params_rep = np.tile(true_params, (N_REALIZATIONS, 1))

            # --- LSQ: independent noise realizations, best-of-restarts each ---
            signals_rep_obj = cfg.simulate_signals(params_rep, cfg.xvalues, snr)
            signals_rep_arr = signals_to_array(signals_rep_obj)
            lsq_estimates = np.array([
                best_of_restarts(signals_rep_arr[r], cfg.xvalues, cfg.bounds,
                                  N_RESTARTS, cfg, rng)
                for r in range(N_REALIZATIONS)
            ])
            lsq_by_snr[snr] = lsq_estimates
            print(f"  LSQ mean: {np.round(lsq_estimates.mean(axis=0), 4)}  "
                  f"(true: {np.round(true_params, 4)})")

            # Pick the realization whose LSQ fit landed closest to this SNR's
            # mean estimate as the "representative" one for the landscape
            # overlay below -- a typical draw rather than a cherry-picked or
            # unusually extreme one.
            centroid_lsq = lsq_estimates.mean(axis=0)
            rep_idx = int(np.argmin(np.linalg.norm(lsq_estimates - centroid_lsq, axis=1)))
            rep_signal_by_snr[snr] = signals_rep_arr[rep_idx]

            # --- NN: the network trained once for this SNR (see above) ---
            nn_estimates = predict_variants(models[snr], signals_rep_obj,
                                            cfg.xvalues, cfg)
            for v in NN_VARIANTS:
                nn_by_snr[v['key']][snr] = nn_estimates[v['key']]
                print(f"  {v['label']} mean: "
                      f"{np.round(nn_estimates[v['key']].mean(axis=0), 4)}")

        # normalized drift of the LSQ centroid at the lowest SNR, relative
        # to ground truth -- just a printed summary so you can tell which
        # rendered case shows the clearest shift, not used to pick cases
        drift_vec   = (lsq_by_snr[worst_snr].mean(axis=0) - true_params) / (hi - lo)
        drift_score = float(np.linalg.norm(drift_vec))

        title = (f'H2 anecdote — {model_name}: the global minimum shifts away from '
                f'ground truth as noise increases (case #{case})\n'
                f'(diamonds=LSQ, squares=NN sup., triangles=NN self-sup. centroids; '
                f'see legend for SNR colors)')
        out_path = os.path.join(out_dir, f'H2_shift_landscape_case{case}.png')
        plot_corner_drift(cfg, true_params, noiseless_signal, lsq_by_snr,
                          nn_by_snr, rep_signal_by_snr, GRID_RES, title, out_path)

        _plot_per_parameter_drift(cfg, true_params, lsq_by_snr, nn_by_snr,
                                  model_name, out_dir, case)

        case_results.append(dict(case_idx=case, true_params=true_params,
                                 lsq_by_snr=lsq_by_snr, nn_by_snr=nn_by_snr,
                                 drift_score=drift_score))

    print(f"\n=== Rendered {len(case_results)} case(s) — pick whichever shows "
          f"the clearest drift ===")
    for r in sorted(case_results, key=lambda x: -x['drift_score']):
        print(f"  case #{r['case_idx']}: drift score={r['drift_score']:.4f} "
              f"(normalized distance from truth to lowest-SNR LSQ centroid)")

    return case_results


def _save_estimates(d, cfg, true_params, lsq_by_snr, nn_by_snr):
    """estimates.csv: every LSQ / NN estimate (one row per noise realization);
    ground_truth.csv: the true parameters of the case."""
    names = list(cfg.param_names)
    cols = {'estimator': [], 'snr': [], 'realization': []}
    cols.update({nm: [] for nm in names})
    estimators = [('msLSQ', lsq_by_snr)] + [
        (f"NN_{v['key']}", nn_by_snr[v['key']]) for v in NN_VARIANTS]
    for est, by_snr in estimators:
        for snr in sorted(by_snr):
            arr = np.asarray(by_snr[snr])
            for i in range(arr.shape[0]):
                cols['estimator'].append(est)
                cols['snr'].append(snr)
                cols['realization'].append(i)
                for k, nm in enumerate(names):
                    cols[nm].append(arr[i, k])
    fd.save_table(d, 'estimates', cols)
    fd.save_table(d, 'ground_truth', {nm: [true_params[k]] for k, nm in enumerate(names)})


def _save_corner_data(out_path, cfg, title, true_params, noiseless_signal,
                      rep_signal_by_snr, lsq_by_snr, nn_by_snr, lsq_centroids,
                      nn_centroids, grids_1d, loss_1d, grids_2d, loss_2d,
                      grids_1d_o, loss_1d_by_snr, vmin, vmax, snr_sorted):
    """
    Everything the corner-drift plot draws, under ``<dir>/data/<figure name>/``
      loss_1d_noiseless.csv   thin black reference curves (diagonal)
      loss_1d_by_snr.csv      one coloured curve per SNR (diagonal)
      loss_2d_noiseless.csv   filled-contour background (lower triangle;
                              rows iy-outer / ix-inner, loss[iy, ix])
      centroids.csv           LSQ / NN centroid per SNR (markers + drift paths)
      estimates.csv           all individual estimates behind the centroids
      signals.csv             noiseless + representative noisy signal per SNR
    """
    d = fd.data_dir_for(out_path)
    names = list(cfg.param_names)
    n = len(names)

    cols = dict(parameter=[], value=[], loss=[])
    for p in range(n):
        cols['parameter'] += [names[p]] * len(grids_1d[p])
        cols['value'] += list(grids_1d[p]); cols['loss'] += list(loss_1d[p])
    fd.save_table(d, 'loss_1d_noiseless', cols)

    cols = dict(snr=[], parameter=[], value=[], loss=[])
    for snr in snr_sorted:
        for p in range(n):
            g = grids_1d_o[p]
            cols['snr'] += [snr] * len(g); cols['parameter'] += [names[p]] * len(g)
            cols['value'] += list(g); cols['loss'] += list(loss_1d_by_snr[snr][p])
    fd.save_table(d, 'loss_1d_by_snr', cols)

    cols = dict(param_x=[], param_y=[], ix=[], iy=[], x=[], y=[], loss=[])
    for (r, c), lg in loss_2d.items():
        gc, gr = grids_2d[(r, c)]
        X, Y = np.meshgrid(gc, gr)
        IX, IY = np.meshgrid(np.arange(len(gc)), np.arange(len(gr)))
        m = lg.size
        cols['param_x'] += [names[c]] * m; cols['param_y'] += [names[r]] * m
        cols['ix'] += IX.ravel().tolist(); cols['iy'] += IY.ravel().tolist()
        cols['x'] += X.ravel().tolist(); cols['y'] += Y.ravel().tolist()
        cols['loss'] += lg.ravel().tolist()
    fd.save_table(d, 'loss_2d_noiseless', cols)

    cols = {'estimator': [], 'snr': []}
    cols.update({nm: [] for nm in names})
    centroid_sets = [('msLSQ', lsq_centroids)] + [
        (f"NN_{v['key']}", nn_centroids[v['key']]) for v in NN_VARIANTS]
    for est, cen in centroid_sets:
        for snr in snr_sorted:
            cols['estimator'].append(est); cols['snr'].append(snr)
            for k, nm in enumerate(names):
                cols[nm].append(cen[snr][k])
    fd.save_table(d, 'centroids', cols)

    _save_estimates(d, cfg, true_params, lsq_by_snr, nn_by_snr)

    cols = {'x': np.asarray(cfg.xvalues).ravel(),
            'noiseless': np.asarray(noiseless_signal).ravel()}
    for snr in snr_sorted:
        cols[f'snr_{snr}'] = np.asarray(rep_signal_by_snr[snr]).ravel()
    fd.save_table(d, 'signals', cols)

    fd.save_meta(d, title=title, param_names=names, snr_levels=list(snr_sorted),
                 bounds=[list(b) for b in cfg.bounds], vmin=vmin, vmax=vmax,
                 n_realizations=N_REALIZATIONS, n_restarts=N_RESTARTS,
                 grid_res=len(grids_1d[0]), grid_res_snr=len(grids_1d_o[0]),
                 n_color_levels=N_COLOR_LEVELS, n_iso_lines=N_ISO_LINES,
                 note='Noise / parameter sampling in this script is not seeded; '
                      'these files are the only exact record of the plotted numbers.')


def plot_corner_drift(cfg, true_params, noiseless_signal, lsq_by_snr,
                      nn_by_snr, rep_signal_by_snr, grid_res, title, out_path):
    """
    Corner-plot visualization of the global minimum shifting away from
    ground truth as noise increases:
      * diagonal panels -> the near-noiseless 1D loss profile (thin black
        reference), overlaid with one colored curve per SNR computed from
        a real representative noisy signal at that SNR -- the dip visibly
        moves as color goes light (high SNR) -> dark (low SNR). Each
        curve's own minimum is marked with a small triangle.
      * lower-triangle  -> the near-noiseless 2D loss slice as a filled
        contour (background/reference), with LSQ/NN centroids
        (diamond/square, each estimator in its own shared colour, tinted
        per-SNR via marker edge) and a drift path connecting them across
        SNR showing precise before/after positions.
      * upper-triangle  -> left empty (as in a standard corner plot);
        used to host one shared colorbar + legend
    """
    n = cfg.n_params
    snr_sorted = sorted(SNR_LEVELS)   # low SNR -> high SNR (ascending), so
                                       # the plasma scale actually comes out
                                       # light=high SNR / dark=low SNR as
                                       # documented above (a descending order
                                       # here would put it backwards)
    snr_colors = snr_colormap(snr_sorted)

    lsq_centroids = {snr: lsq_by_snr[snr].mean(axis=0) for snr in snr_sorted}
    nn_centroids  = {k: {snr: by_snr[snr].mean(axis=0) for snr in snr_sorted}
                     for k, by_snr in nn_by_snr.items()}

    # --- near-noiseless background grids (reference landscape) ---
    grids_1d, loss_1d = {}, {}
    for p in range(n):
        grids_1d[p] = np.linspace(*cfg.bounds[p], grid_res)
        loss_1d[p]  = compute_loss_1d(true_params, p, grids_1d[p],
                                      cfg.xvalues, noiseless_signal, cfg)

    grids_2d, loss_2d = {}, {}
    for r in range(1, n):
        for c in range(r):
            grid_c = np.linspace(*cfg.bounds[c], grid_res)
            grid_r = np.linspace(*cfg.bounds[r], grid_res)
            loss_2d[(r, c)] = compute_loss_grid(true_params, c, r, grid_c,
                                                grid_r, cfg.xvalues,
                                                noiseless_signal, cfg)
            grids_2d[(r, c)] = (grid_c, grid_r)

    all_loss = np.concatenate(
        [v.ravel() for v in loss_1d.values()] +
        [v.ravel() for v in loss_2d.values()]
    )
    vmin, vmax = max(all_loss.min(), 1e-12), all_loss.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fill_levels = np.logspace(np.log10(vmin), np.log10(vmax), N_COLOR_LEVELS + 1)
    iso_levels  = np.logspace(np.log10(vmin), np.log10(vmax), N_ISO_LINES + 2)[1:-1]

    # --- per-SNR overlay grids for the diagonal panels, at a coarser
    # resolution (only one curve each is drawn from these, not a filled
    # surface, so full grid_res isn't needed) ---
    overlay_res = min(grid_res, GRID_RES_SNR)
    grids_1d_o = {p: np.linspace(*cfg.bounds[p], overlay_res) for p in range(n)}
    loss_1d_by_snr = {snr: {} for snr in snr_sorted}
    for snr in snr_sorted:
        for p in range(n):
            loss_1d_by_snr[snr][p] = compute_loss_1d(
                true_params, p, grids_1d_o[p], cfg.xvalues,
                rep_signal_by_snr[snr], cfg)

    # Save the raw numbers behind this figure (before plotting)
    _save_corner_data(out_path, cfg, title, true_params, noiseless_signal,
                      rep_signal_by_snr, lsq_by_snr, nn_by_snr, lsq_centroids,
                      nn_centroids, grids_1d, loss_1d, grids_2d, loss_2d,
                      grids_1d_o, loss_1d_by_snr, vmin, vmax, snr_sorted)

    draw_corner_drift(cfg, true_params, lsq_by_snr, nn_by_snr, grids_1d,
                      loss_1d, grids_2d, loss_2d, grids_1d_o, loss_1d_by_snr,
                      vmin, vmax, title, out_path)


def draw_corner_drift(cfg, true_params, lsq_by_snr, nn_by_snr, grids_1d,
                      loss_1d, grids_2d, loss_2d, grids_1d_o, loss_1d_by_snr,
                      vmin, vmax, title, out_path):
    """Draw the corner-drift plot from precomputed loss curves/grids and
    estimates (used by the experiment and by ``--replot``)."""
    n = cfg.n_params
    snr_sorted = sorted(SNR_LEVELS)
    snr_colors = snr_colormap(snr_sorted)
    lsq_centroids = {snr: lsq_by_snr[snr].mean(axis=0) for snr in snr_sorted}
    nn_centroids  = {k: {snr: by_snr[snr].mean(axis=0) for snr in snr_sorted}
                     for k, by_snr in nn_by_snr.items()}
    norm = LogNorm(vmin=vmin, vmax=vmax)
    fill_levels = np.logspace(np.log10(vmin), np.log10(vmax), N_COLOR_LEVELS + 1)
    iso_levels  = np.logspace(np.log10(vmin), np.log10(vmax), N_ISO_LINES + 2)[1:-1]

    fig, axes = plt.subplots(n, n, figsize=(2.9 * n + 1.6, 2.9 * n), squeeze=False)
    last_cf = None

    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.axis('off')
                continue

            if r == c:
                # near-noiseless reference curve
                ax.plot(grids_1d[r], loss_1d[r], color='black', lw=1.1,
                       alpha=0.6, zorder=2, label='near-noiseless')
                ax.set_yscale('log')
                ax.set_ylim(vmin, vmax)
                ax.set_xlim(cfg.bounds[r])
                ax.axvline(true_params[r], color=COLORS['GT'], lw=2.2, ls='--', zorder=5)

                # one colored curve per SNR, from a real noisy signal at
                # that SNR -- the dip moving is the minimum shifting
                for snr in snr_sorted:
                    col = snr_colors[snr]
                    curve = loss_1d_by_snr[snr][r]
                    ax.plot(grids_1d_o[r], curve, color=col, lw=1.7,
                           alpha=0.9, zorder=3)
                    min_idx = int(np.argmin(curve))
                    ax.scatter(grids_1d_o[r][min_idx], curve[min_idx],
                              marker='v', s=42, color=col, edgecolors='k',
                              linewidths=0.5, zorder=6)
                if r != n - 1:
                    ax.set_xticklabels([])
                ax.set_yticklabels([])
            else:
                grid_c, grid_r = grids_2d[(r, c)]
                lg = loss_2d[(r, c)]
                last_cf = ax.contourf(grid_c, grid_r, lg, levels=fill_levels,
                                      norm=norm, cmap='viridis')
                cs = ax.contour(grid_c, grid_r, lg, levels=iso_levels,
                                colors='white', linewidths=0.5, alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

                lsq_path = np.array([lsq_centroids[snr] for snr in snr_sorted])
                ax.plot(lsq_path[:, c], lsq_path[:, r], '--', color=COLORS['msLSQ'],
                       lw=1.6, alpha=0.85, zorder=5)
                for v in NN_VARIANTS:
                    nn_path = np.array([nn_centroids[v['key']][snr] for snr in snr_sorted])
                    ax.plot(nn_path[:, c], nn_path[:, r], PATH_LS[v['key']],
                           color=v['color'], lw=1.6, alpha=0.85, zorder=5)
                for snr in snr_sorted:
                    # marker fill = estimator identity (LSQ / each NN), marker
                    # edge = SNR, so both axes stay legible at once
                    col = snr_colors[snr]
                    ax.scatter(lsq_centroids[snr][c], lsq_centroids[snr][r],
                              marker='D', s=75, color=COLORS['msLSQ'],
                              edgecolors=col, linewidths=2.0, zorder=6)
                    for v in NN_VARIANTS:
                        cen = nn_centroids[v['key']][snr]
                        ax.scatter(cen[c], cen[r], marker=v['marker'], s=75,
                                  color=v['color'], edgecolors=col,
                                  linewidths=2.0, zorder=6)

                ax.scatter(true_params[c], true_params[r], marker='*', s=280,
                          color=COLORS['GT'], edgecolors='k', linewidths=0.8, zorder=7)
                ax.set_xlim(cfg.bounds[c])
                ax.set_ylim(cfg.bounds[r])
                if r != n - 1:
                    ax.set_xticklabels([])
                if c != 0:
                    ax.set_yticklabels([])

            if r == n - 1:
                ax.set_xlabel(param_label(cfg.param_names[c]))
            if c == 0 and r != 0:
                pname_r = cfg.param_names[r]
                ax.set_ylabel(with_unit(param_label(pname_r), pname_r))
            if r == 0 and c == 0:
                ax.set_ylabel('loss (log, a.u.)')

    cbar_ax = fig.add_axes([0.64, 0.60, 0.02, 0.30])
    fig.colorbar(last_cf, cax=cbar_ax,
                label='Near-noiseless residual (log scale, a.u.)')

    # One explicit legend entry per SNR, in its actual plotted color --
    # rather than relying on a "light->dark" description, which is hard to
    # map back onto a specific line/marker at a glance.
    snr_legend_elems = [
        Line2D([0], [0], color=snr_colors[snr], lw=3.0, label=f'SNR = {snr}')
        for snr in snr_sorted
    ]
    # one entry per estimator: centroid marker joined by its drift-path line
    # (marker edge = SNR is explained in the SNR legend's title)
    marker_legend_elems = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['GT'],
              markeredgecolor='k', markersize=15, label='Ground truth', linestyle='None'),
        Line2D([0], [0], marker='D', color=COLORS['msLSQ'], ls='--', lw=1.6,
              markerfacecolor=COLORS['msLSQ'], markeredgecolor='gray',
              markersize=8, label='LSQ centroid, drift'),
    ]
    for v in NN_VARIANTS:
        marker_legend_elems.append(
            Line2D([0], [0], marker=v['marker'], color=v['color'],
                  ls=PATH_LS[v['key']], lw=1.6, markerfacecolor=v['color'],
                  markeredgecolor='gray', markersize=8,
                  label=f"{v['label']} centroid, drift"))
    marker_legend_elems.append(
        Line2D([0], [0], color='black', lw=1.1, alpha=0.6, label='Near-noiseless reference'))
    fig.legend(handles=snr_legend_elems, loc='upper left',
              bbox_to_anchor=(0.64, 0.58), fontsize=LEGEND_SMALL, framealpha=0.9,
              title='SNR colors (also used for\n1D dips / marker edges)')
    fig.legend(handles=marker_legend_elems, loc='upper left',
              bbox_to_anchor=(0.64, 0.40), fontsize=LEGEND_SMALL, framealpha=0.9)

    fig.subplots_adjust(left=0.1, right=0.98, top=0.96, bottom=0.08,
                        wspace=0.08, hspace=0.08)
    save_fig(fig, out_path, tight=False)
    plt.show()
    print(f"Saved: {out_path}")
    print(f"  {title}")


def _plot_per_parameter_drift(cfg, true_params, lsq_by_snr, nn_by_snr,
                              model_name, out_dir, case_idx):
    n_params = cfg.n_params
    snr_sorted = sorted(SNR_LEVELS)   # ascending, matching
                                       # H2_noise_dominated_minimum.py and
                                       # the corner plot above
    # Save the raw numbers behind this figure (before plotting)
    d = fd.data_dir_for(os.path.join(out_dir, f'H2_shift_per_parameter_case{case_idx}.png'))
    _save_estimates(d, cfg, true_params, lsq_by_snr, nn_by_snr)
    fd.save_meta(d, model=model_name, case_idx=case_idx,
                 param_names=list(cfg.param_names), snr_levels=list(snr_sorted),
                 n_realizations=N_REALIZATIONS, n_restarts=N_RESTARTS,
                 note='Boxplots are computed from estimates.csv; the dashed line is ground_truth.csv.')

    _draw_per_parameter_drift(cfg, true_params, lsq_by_snr, nn_by_snr,
                              model_name, out_dir, case_idx)


def _draw_per_parameter_drift(cfg, true_params, lsq_by_snr, nn_by_snr,
                              model_name, out_dir, case_idx):
    """Draw the per-parameter boxplot figure from the stored estimates."""
    n_params = cfg.n_params
    snr_sorted = sorted(SNR_LEVELS)
    fig, axes = plt.subplots(1, n_params, figsize=(5.5 * n_params, 4.5),
                             squeeze=False)
    axes = axes[0]

    # one group of boxes per SNR: msLSQ, then one box per NN variant
    n_boxes = 1 + len(NN_VARIANTS)
    group_stride = 0.9 * n_boxes + 0.7

    for k, (ax, pname) in enumerate(zip(axes, cfg.param_names)):
        positions_lsq = np.arange(len(snr_sorted)) * group_stride

        bp_lsq = ax.boxplot([lsq_by_snr[snr][:, k] for snr in snr_sorted],
                            positions=positions_lsq, widths=0.7,
                            patch_artist=True, showfliers=False)
        for box in bp_lsq['boxes']:
            box.set_facecolor(COLORS['msLSQ']); box.set_alpha(0.75)
        for j, v in enumerate(NN_VARIANTS, start=1):
            bp_nn = ax.boxplot([nn_by_snr[v['key']][snr][:, k] for snr in snr_sorted],
                               positions=positions_lsq + 0.9 * j, widths=0.7,
                               patch_artist=True, showfliers=False)
            for box in bp_nn['boxes']:
                box.set_facecolor(v['color']); box.set_alpha(0.75)

        ax.axhline(true_params[k], color=COLORS['GT'], linestyle='--',
                  linewidth=2, zorder=5)

        ax.set_xticks(positions_lsq + 0.9 * (n_boxes - 1) / 2)
        ax.set_xticklabels([f'{s}' for s in snr_sorted])
        ax.set_title(param_label(pname))
        # Every panel shows its own unit as its normal y-axis label (right
        # next to the axis, as usual) -- a dimensionless parameter (f,
        # v_e, v_p) gets no unit label at all rather than an empty "()".
        # The leftmost panel additionally carries a second, outer label
        # ("Parameter value") further from the axis, applying to the row
        # as a whole -- two stacked levels: the generic category label
        # further out, that panel's own specific unit closer in, exactly
        # like the two-line row labels used elsewhere in this project
        # (e.g. H6's all-parameter summary), just split across two
        # separate Text artists here since set_ylabel() only places one.
        unit = param_unit(pname)
        if unit:
            ax.set_ylabel(f'({unit})')
        if k == 0:
            ax.text(-0.60, 0.5, 'Parameter value', transform=ax.transAxes,
                    rotation=90, va='center', ha='center')
        ax.set_xlabel('SNR')
        if k == 0:
            handles = [Patch(color=COLORS['msLSQ'], alpha=0.75, label='msLSQ')]
            handles += [Patch(color=v['color'], alpha=0.75, label=v['label'])
                        for v in NN_VARIANTS]
            handles.append(Line2D([0], [0], color=COLORS['GT'], linestyle='--',
                                  label='Ground truth'))
            ax.legend(handles=handles, fontsize=LEGEND_SMALL)

    fig.text(0.01, 0.99,
             f'H2 anecdote — {model_name}, case #{case_idx} '
             f'({N_REALIZATIONS} noise realizations per SNR level)',
             ha='left', va='top', fontsize=10, alpha=0.7)
    out_path = os.path.join(out_dir, f'H2_shift_per_parameter_case{case_idx}.png')
    save_fig(fig, out_path)
    plt.show()
    print(f"Saved: {out_path}")


def _load_estimates(d, cfg, snr_levels):
    """Rebuild (true_params, lsq_by_snr, nn_by_snr) from estimates.csv and
    ground_truth.csv in data folder ``d``."""
    names = list(cfg.param_names)
    e = fd.load_table(d, 'estimates')
    stores = {'msLSQ': {}}
    stores.update({f"NN_{v['key']}": {} for v in NN_VARIANTS})
    for est, store in stores.items():
        for snr in snr_levels:
            m = (e['estimator'] == est) & (e['snr'] == snr)
            order = np.argsort(e['realization'][m], kind='stable')
            store[snr] = np.column_stack([e[nm][m][order] for nm in names])
    g = fd.load_table(d, 'ground_truth')
    true_params = np.array([g[nm][0] for nm in names])
    return (true_params, stores['msLSQ'],
            {v['key']: stores[f"NN_{v['key']}"] for v in NN_VARIANTS})


def replot(model_name='DCE', case_idx=None):
    """Re-draw both figures of the saved case(s) from the data files
    (no computation). Restores SNR levels / realization count from meta.json."""
    global SNR_LEVELS, N_REALIZATIONS
    apply_style()
    cfg = get_model_config(model_name)
    out_dir = outdir(model_name)
    names = list(cfg.param_names)
    n = len(names)
    root = os.path.join(out_dir, 'data')
    prefix = 'H2_shift_landscape_case'
    cases = sorted(int(x[len(prefix):]) for x in os.listdir(root) if x.startswith(prefix))
    if case_idx is not None:
        cases = [c for c in cases if c in set(case_idx)]
    if not cases:
        print(f'No saved noise-shift data found under {root}')
        return

    for case in cases:
        d = os.path.join(root, f'{prefix}{case}')
        meta = fd.load_meta(d)
        SNR_LEVELS = list(meta['snr_levels'])
        N_REALIZATIONS = int(meta['n_realizations'])
        true_params, lsq_by_snr, nn_by_snr = _load_estimates(d, cfg, SNR_LEVELS)

        t1 = fd.load_table(d, 'loss_1d_noiseless')
        grids_1d, loss_1d = {}, {}
        for p, nm in enumerate(names):
            m = t1['parameter'] == nm
            grids_1d[p], loss_1d[p] = t1['value'][m], t1['loss'][m]

        t2 = fd.load_table(d, 'loss_2d_noiseless')
        grids_2d, loss_2d = {}, {}
        for r in range(1, n):
            for c in range(r):
                gx, gy, z = fd.grid2d_from_table(t2, names[c], names[r])
                grids_2d[(r, c)], loss_2d[(r, c)] = (gx, gy), z

        t3 = fd.load_table(d, 'loss_1d_by_snr')
        grids_1d_o = {}
        loss_1d_by_snr = {snr: {} for snr in SNR_LEVELS}
        for snr in SNR_LEVELS:
            for p, nm in enumerate(names):
                m = (t3['snr'] == snr) & (t3['parameter'] == nm)
                grids_1d_o[p] = t3['value'][m]
                loss_1d_by_snr[snr][p] = t3['loss'][m]

        draw_corner_drift(cfg, true_params, lsq_by_snr, nn_by_snr, grids_1d,
                          loss_1d, grids_2d, loss_2d, grids_1d_o, loss_1d_by_snr,
                          meta['vmin'], meta['vmax'], meta['title'],
                          os.path.join(out_dir, f'{prefix}{case}.png'))
        _draw_per_parameter_drift(cfg, true_params, lsq_by_snr, nn_by_snr,
                                  model_name, out_dir, case)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IVIM', choices=['IVIM', 'DCE'])
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--case_idx', type=int, nargs='+', default=None,
                        help='Specific ground-truth candidate indices to '
                             'render (e.g. --case_idx 0 2 5). Default: the '
                             'first --n_cases candidates.')
    parser.add_argument('--n_cases', type=int, default=N_CASES_TO_PLOT,
                        help='Number of ground-truth cases to render when '
                             '--case_idx is not given (default: %(default)s). '
                             'The NNs are trained once per SNR level and '
                             'shared by all cases; the remaining cost '
                             '(LSQ restarts, landscape grids) scales '
                             'linearly with this.')
    parser.add_argument('--replot', action='store_true',
                        help='Re-draw the saved case(s) from the data files '
                             '(no computation). Combine with --case_idx to '
                             'pick specific cases.')
    args = parser.parse_args()
    if args.replot:
        replot(args.model, case_idx=args.case_idx)
    else:
        run_experiment(args.model, seed=args.seed, case_idx=args.case_idx,
                       n_cases=args.n_cases)