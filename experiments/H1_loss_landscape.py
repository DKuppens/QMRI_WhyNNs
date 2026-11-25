import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from backend.QMRI.model_registry import get_model_config
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.experiment_utils import signals_to_array
from backend.plot_style import apply_style, COLORS, LEGEND_SMALL, param_label, outdir, save_fig
from backend import figure_data as fd

# --- tunables -------------------------------------------------------------
N_CANDIDATES     = 25    # candidate signals scanned to auto-pick cases from
N_CASES_TO_PLOT  = 20    # number of illustrative cases to auto-select and
                          # render (each saved to its own file) when
                          # --case_idx isn't given, so you can pick the
                          # clearest one yourself
N_RESTARTS_SCAN  = 20    # restarts used only to *score* candidates (cheap)
N_RESTARTS_FULL  = 40    # restarts used for the final plotted case (dense)
GRID_RES         = 60    # resolution per axis of the 2D loss grid
N_COLOR_LEVELS   = 20    # number of distinguishable filled-contour color
                          # bands, log-spaced to match the shared LogNorm
                          # color scale. NOTE: contourf/contour's `levels=N`
                          # shorthand spaces levels *linearly* in data value
                          # even under a LogNorm color scale, which crams
                          # almost all of the levels into the high-loss end
                          # and leaves only 2-3 effectively different colors
                          # over the log-relevant low-loss region where the
                          # landscape's actual structure lives -- this is
                          # why the previous version could look like only
                          # "2-3 blocks". Levels are generated explicitly
                          # with np.logspace below to fix this.
N_ISO_LINES      = 8     # number of labeled isolines on top of the filled
                          # contour, also log-spaced for the same reason
N_TRAIN_NN       = 100000 # only used if --no_nn is not set
# ---------------------------------------------------------------------------


# ------------------------------------------------------------------ #
# Core numerics
# ------------------------------------------------------------------ #

def residual(params, signal, xvalues, cfg):
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def multi_start_lsq_pool(signal, xvalues, bounds, n_restarts, cfg, rng):
    """Run n_restarts independent LSQ fits and return ALL of them (not just
    the best) so we can see how they scatter across the loss landscape."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    pool_params = np.empty((n_restarts, cfg.n_params))
    pool_resid  = np.empty(n_restarts)
    for r in range(n_restarts):
        init = rng.uniform(lo, hi)
        p = cfg.fit_lsq_1d(xvalues, signal, init)
        pool_params[r] = p
        pool_resid[r]  = residual(p, signal, xvalues, cfg)
    return pool_params, pool_resid


def cluster_count(pool_params, bounds, thresh=0.08):
    """Greedy distance-threshold clustering in bounds-normalized parameter
    space. This is only a heuristic used to auto-select an illustrative
    case for the figure -- it is not meant as a rigorous clustering method,
    just a cheap way to say "these restarts landed in ~N distinct places"."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    norm = (pool_params - lo) / (hi - lo)
    centers = []
    for p in norm:
        if not centers or min(np.linalg.norm(p - c) for c in centers) > thresh:
            centers.append(p)
    return len(centers)


def compute_loss_1d(fixed_params, p, grid_p, xvalues, signal, cfg):
    """1D data-residual profile over parameter p, all others fixed
    (typically at ground truth)."""
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


def save_corner_data(out_path, cfg, title, true_params, signal, pool_params,
                     pool_resid, best_lsq, nn_pred, grids_1d, loss_1d,
                     grids_2d, loss_2d, vmin, vmax):
    """
    Store everything the corner plot draws under
    ``<dir>/data/<figure name>/`` (see backend/figure_data.py):
      loss_1d.csv     diagonal panels   (parameter, value, loss)
      loss_2d.csv     lower triangle    (param_x, param_y, ix, iy, x, y, loss);
                      loss.reshape(n_y, n_x) after sorting by (iy, ix)
      restarts.csv    one row per LSQ restart (residual + estimated params)
      markers.csv     ground truth / best LSQ / NN (one row each)
      signal.csv      the noisy signal that was fitted
    """
    d = fd.data_dir_for(out_path)
    names = list(cfg.param_names)
    n = len(names)

    # 1D profiles
    cols = dict(parameter=[], value=[], loss=[])
    for p in range(n):
        cols['parameter'] += [names[p]] * len(grids_1d[p])
        cols['value'] += list(grids_1d[p])
        cols['loss'] += list(loss_1d[p])
    fd.save_table(d, 'loss_1d', cols)

    # 2D slices (rows ordered iy-outer / ix-inner, matching loss[iy, ix])
    cols = dict(param_x=[], param_y=[], ix=[], iy=[], x=[], y=[], loss=[])
    for (r, c), lg in loss_2d.items():
        gc, gr = grids_2d[(r, c)]
        X, Y = np.meshgrid(gc, gr)
        IX, IY = np.meshgrid(np.arange(len(gc)), np.arange(len(gr)))
        m = lg.size
        cols['param_x'] += [names[c]] * m
        cols['param_y'] += [names[r]] * m
        cols['ix'] += IX.ravel().tolist()
        cols['iy'] += IY.ravel().tolist()
        cols['x'] += X.ravel().tolist()
        cols['y'] += Y.ravel().tolist()
        cols['loss'] += lg.ravel().tolist()
    fd.save_table(d, 'loss_2d', cols)

    # LSQ restarts
    cols = dict(restart=list(range(len(pool_resid))), residual=list(pool_resid))
    for i, nm in enumerate(names):
        cols[nm] = list(pool_params[:, i])
    fd.save_table(d, 'restarts', cols)

    # markers
    labels = ['ground_truth', 'best_lsq']
    rows = [true_params, best_lsq]
    if nn_pred is not None:
        for v in NN_VARIANTS:
            labels.append(f"nn_{v['key']}"); rows.append(nn_pred[v['key']])
    cols = dict(name=labels)
    for i, nm in enumerate(names):
        cols[nm] = [row[i] for row in rows]
    fd.save_table(d, 'markers', cols)

    fd.save_table(d, 'signal', dict(x=np.asarray(cfg.xvalues).ravel(),
                                    signal=np.asarray(signal).ravel()))
    fd.save_meta(d, title=title, param_names=names,
                 bounds=[list(b) for b in cfg.bounds], snr=cfg.default_snr,
                 vmin=vmin, vmax=vmax, grid_res=len(grids_1d[0]),
                 n_color_levels=N_COLOR_LEVELS, n_iso_lines=N_ISO_LINES,
                 n_restarts=len(pool_resid),
                 colour_scale='LogNorm(vmin, vmax); levels = logspace(vmin, vmax, N_COLOR_LEVELS+1)')


def plot_corner(cfg, true_params, signal, pool_params, pool_resid, best_lsq,
                nn_pred, grid_res, title, out_path):
    """
    Corner-plot visualization of the loss landscape:
      * diagonal panels    -> 1D loss profile for each parameter
      * lower-triangle     -> 2D loss slice for each parameter pair, with
                              a small number of labeled isolines on top of
                              the filled contour
      * upper-triangle     -> left empty (as in a standard corner plot);
                              used to host one shared colorbar + legend

    All panels vary the plotted parameter(s) over the full bound range
    while holding every other parameter fixed at ground truth, and share a
    single, global color scale so panel "steepness" is directly comparable.
    """
    n = cfg.n_params

    # --- compute all 1D and 2D grids up front, so we can share one color
    #     scale (based on the true global min/max across every panel) ---
    grids_1d, loss_1d = {}, {}
    for p in range(n):
        grids_1d[p] = np.linspace(*cfg.bounds[p], grid_res)
        loss_1d[p]  = compute_loss_1d(true_params, p, grids_1d[p],
                                       cfg.xvalues, signal, cfg)

    grids_2d, loss_2d = {}, {}
    for r in range(1, n):
        for c in range(r):
            grid_c = np.linspace(*cfg.bounds[c], grid_res)
            grid_r = np.linspace(*cfg.bounds[r], grid_res)
            loss_2d[(r, c)] = compute_loss_grid(true_params, c, r, grid_c,
                                                grid_r, cfg.xvalues, signal, cfg)
            grids_2d[(r, c)] = (grid_c, grid_r)

    all_loss = np.concatenate(
        [v.ravel() for v in loss_1d.values()] +
        [v.ravel() for v in loss_2d.values()]
    )
    vmin, vmax = max(all_loss.min(), 1e-12), all_loss.max()
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Save the raw numbers behind this figure (before plotting)
    save_corner_data(out_path, cfg, title, true_params, signal, pool_params,
                     pool_resid, best_lsq, nn_pred, grids_1d, loss_1d,
                     grids_2d, loss_2d, vmin, vmax)

    draw_corner(cfg, true_params, pool_params, best_lsq, nn_pred, grids_1d,
                loss_1d, grids_2d, loss_2d, vmin, vmax, title, out_path)


def draw_corner(cfg, true_params, pool_params, best_lsq, nn_pred, grids_1d,
                loss_1d, grids_2d, loss_2d, vmin, vmax, title, out_path):
    """Draw the corner plot from precomputed loss grids (used both by the
    experiment and by ``--replot``). Edit the styling here."""
    n = cfg.n_params
    norm = LogNorm(vmin=vmin, vmax=vmax)

    # Log-spaced level boundaries, shared by every panel's filled contour
    # and isolines, so the color/line resolution actually matches the log
    # color scale instead of being crammed into the high-loss end (see the
    # N_COLOR_LEVELS comment near the top of the file).
    fill_levels = np.logspace(np.log10(vmin), np.log10(vmax), N_COLOR_LEVELS + 1)
    iso_levels  = np.logspace(np.log10(vmin), np.log10(vmax), N_ISO_LINES + 2)[1:-1]

    fig, axes = plt.subplots(n, n, figsize=(2.9 * n + 1.4, 2.9 * n),
                             squeeze=False)
    last_cf = None

    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                ax.axis('off')
                continue

            if r == c:
                ax.plot(grids_1d[r], loss_1d[r], color='black', lw=1.3)
                ax.set_yscale('log')
                ax.set_ylim(vmin, vmax)
                ax.set_xlim(cfg.bounds[r])
                ax.axvline(true_params[r], color=COLORS['GT'], lw=2.2, ls='--', zorder=5)
                ax.axvline(best_lsq[r], color=COLORS['msLSQ'], lw=1.8, ls='--', zorder=4)
                if nn_pred is not None:
                    for v in NN_VARIANTS:
                        ax.axvline(nn_pred[v['key']][r], color=v['color'],
                                   lw=1.8, ls='--', zorder=4)
                # where each restart landed along this parameter: one thin
                # vertical dashed line per restart (same line language as the
                # best-LSQ and NN lines above), so restarts are visible on
                # the diagonal too, not just the 2D panels
                for x_restart in pool_params[:, r]:
                    ax.axvline(x_restart, color='black', lw=1.0, ls='--',
                               alpha=0.35, zorder=3)
                if r != n - 1:
                    ax.set_xticklabels([])
                # no numeric loss scale on the diagonal either (set_yticklabels
                # alone leaves the log axis' minor tick labels showing)
                ax.tick_params(axis='y', which='both', labelleft=False)
            else:
                grid_c, grid_r = grids_2d[(r, c)]
                lg = loss_2d[(r, c)]
                last_cf = ax.contourf(grid_c, grid_r, lg, levels=fill_levels,
                                      norm=norm, cmap='viridis')
                cs = ax.contour(grid_c, grid_r, lg, levels=iso_levels,
                                colors='white', linewidths=0.6, alpha=0.8)
                ax.clabel(cs, inline=True, fontsize=6, fmt='%.2g')

                # LSQ restarts: bright, high-contrast markers with a bold
                # outline so they read clearly against every color in the
                # viridis scale, from dark purple to bright yellow
                ax.scatter(pool_params[:, c], pool_params[:, r],
                          c='white', edgecolors='black', linewidths=1.0,
                          s=55, alpha=0.9, zorder=4)
                ax.scatter(true_params[c], true_params[r], marker='*', s=260,
                          color=COLORS['GT'], edgecolors='k', linewidths=0.7, zorder=5)
                ax.scatter(best_lsq[c], best_lsq[r], marker='D', s=80,
                          color=COLORS['msLSQ'], edgecolors='k', linewidths=0.9, zorder=5)
                if nn_pred is not None:
                    for v in NN_VARIANTS:
                        ax.scatter(nn_pred[v['key']][c], nn_pred[v['key']][r],
                                  marker=v['marker'], s=80, color=v['color'],
                                  edgecolors='k', linewidths=0.9, zorder=5)
                ax.set_xlim(cfg.bounds[c])
                ax.set_ylim(cfg.bounds[r])
                if r != n - 1:
                    ax.set_xticklabels([])
                if c != 0:
                    ax.set_yticklabels([])

            if r == n - 1:
                ax.set_xlabel(param_label(cfg.param_names[c]))
            if c == 0 and r != 0:
                ax.set_ylabel(param_label(cfg.param_names[r]))
            if r == 0 and c == 0:
                ax.set_ylabel('loss (log)')

    # Shared colorbar + legend, placed in the empty upper-triangle area.
    # Row 0, columns 1..n-1 are always empty for n >= 2 (the upper
    # triangle starts at row 0), so their combined, *measured* bounding
    # box is a safe, correctly-positioned free rectangle to draw into --
    # unlike the fixed figure-fraction constants this replaces, which
    # didn't track n or the figure's actual size and could land on top of
    # the (0, 0) diagonal panel instead.
    free_left_ax  = axes[0, 1]
    free_right_ax = axes[0, n - 1]
    free_x0 = free_left_ax.get_position().x0
    free_x1 = free_right_ax.get_position().x1
    free_y0 = free_left_ax.get_position().y0
    free_y1 = free_left_ax.get_position().y1

    cbar_w = 0.02
    cbar_margin = 0.01
    # leave a strip above and below the bar for the two end labels
    cbar_pad = 0.15 * (free_y1 - free_y0)
    cbar_ax = fig.add_axes([free_x1 - cbar_w - cbar_margin, free_y0 + cbar_pad,
                            cbar_w, (free_y1 - free_y0) - 2 * cbar_pad])
    cbar = fig.colorbar(last_cf, cax=cbar_ax)
    # no numeric scale: only the two ends are named
    cbar.set_ticks([])
    cbar.minorticks_off()
    cbar.ax.tick_params(which='both', length=0)
    cbar_ax.text(1.0, 1.02, 'high data residual', transform=cbar_ax.transAxes,
                 ha='right', va='bottom', fontsize=LEGEND_SMALL)
    cbar_ax.text(1.0, -0.02, 'low data residual', transform=cbar_ax.transAxes,
                 ha='right', va='top', fontsize=LEGEND_SMALL)

    legend_elems = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['GT'],
              markeredgecolor='k', markersize=15, label='Ground truth', linestyle='None'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=COLORS['msLSQ'],
              markeredgecolor='k', markersize=9, label='Best LSQ', linestyle='None'),
    ]
    if nn_pred is not None:
        legend_elems += [
            Line2D([0], [0], marker=v['marker'], color='w',
                  markerfacecolor=v['color'], markeredgecolor='k',
                  markersize=9, label=v['label'], linestyle='None')
            for v in NN_VARIANTS]
    legend_elems.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
              markeredgecolor='k', markersize=9, label='LSQ restarts', linestyle='None'))
    fig.legend(handles=legend_elems, loc='upper left',
              bbox_to_anchor=(free_x0, free_y1), fontsize=LEGEND_SMALL,
              framealpha=0.9)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.96, bottom=0.08,
                        wspace=0.08, hspace=0.08)
    save_fig(fig, out_path, tight=False)
    plt.show()
    print(f"Saved: {out_path}")
    print(f"  {title}")


def compute_loss_grid(fixed_params, i, j, grid_i, grid_j, xvalues, signal, cfg):
    """2D data-residual surface over parameters (i, j); all other parameters
    are held fixed at fixed_params (typically the ground truth)."""
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


# ------------------------------------------------------------------ #
# Case selection
# ------------------------------------------------------------------ #

def pick_illustrative_cases(model_name, cfg, params_cand, signals_arr, rng, n_pick):
    """
    Scan a handful of candidate signals and score each by how its restart
    pool scatters in parameter/residual space, then rank candidates by how
    well they exemplify the model's expected behaviour and return the top
    n_pick indices (best first):

      DCE  -> high *residual* spread across restarts (restarts really do
              land at genuinely different loss values -> local minima)
      IVIM -> high *parameter* spread but LOW residual spread (restarts
              land all over the place but at essentially the same loss
              -> one wide, flat valley, not isolated minima)

    Returning several ranked candidates (rather than just the single best)
    lets you render each one and pick whichever renders clearest -- the
    scoring heuristic is cheap but crude, and the "best" case by these
    numbers isn't always the most visually illustrative one.
    """
    lo = np.array([b[0] for b in cfg.bounds])
    hi = np.array([b[1] for b in cfg.bounds])
    n_scan = min(N_CANDIDATES, len(params_cand))

    scores = []
    for c in range(n_scan):
        pool_p, pool_r = multi_start_lsq_pool(
            signals_arr[c], cfg.xvalues, cfg.bounds, N_RESTARTS_SCAN, cfg, rng
        )
        resid_cv   = np.std(pool_r) / (np.mean(pool_r) + 1e-12)
        param_norm = (pool_p - lo) / (hi - lo)
        param_std  = float(np.mean(np.std(param_norm, axis=0)))
        scores.append((resid_cv, param_std))
        print(f"  candidate {c}: residual CV={resid_cv:.3f}  "
              f"param spread={param_std:.3f}")

    scores = np.array(scores)
    if model_name == 'DCE':
        ranking = np.argsort(-scores[:, 0])   # descending residual CV
    else:
        flat_mask       = scores[:, 0] <= np.median(scores[:, 0])
        flat_candidates  = np.where(flat_mask)[0]
        ranked_flat      = flat_candidates[np.argsort(-scores[flat_candidates, 1])]
        # if n_pick exceeds the flat pool, fall back to the remaining
        # candidates ranked by param spread too, so n_pick is always met
        remaining = np.array([i for i in np.argsort(-scores[:, 1])
                              if i not in set(ranked_flat.tolist())])
        ranking = np.concatenate([ranked_flat, remaining]).astype(int)

    picked = ranking[:n_pick].tolist()
    print(f"  -> auto-selected top {len(picked)} candidate(s): {picked}")
    return picked


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run_experiment(model_name='DCE', case_idx=None, include_nn=True,
                   n_cases=N_CASES_TO_PLOT):
    apply_style()
    cfg = get_model_config(model_name)
    rng = np.random.default_rng(seed=0)

    print(f"\n=== H1 anecdote — Loss landscape | model={model_name} ===")

    if cfg.n_params < 2:
        print(f"{model_name} has fewer than 2 parameters; a pairwise "
              f"loss-landscape slice is not applicable.")
        return None

    case_idx_list = list(case_idx) if case_idx is not None else None
    largest_requested = max(case_idx_list) if case_idx_list else 0
    n_pool = max(N_CANDIDATES, largest_requested + 1, n_cases)
    params_cand, coeffs_cand = cfg.sample_parameters(n_pool, cfg.param_distribution)
    signals_cand = cfg.simulate_signals(params_cand, cfg.xvalues, cfg.default_snr)
    signals_arr  = signals_to_array(signals_cand)

    nn_preds_all = None
    if include_nn:
        print("Training NNs (supervised + self-supervised; for the reference "
              "overlay; trained once, reused across all rendered cases)...")
        params_train, coeffs_train = cfg.sample_parameters(N_TRAIN_NN, cfg.param_distribution)
        signals_train = cfg.simulate_signals(params_train, cfg.xvalues, cfg.default_snr)
        models = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg)
        nn_preds_all = predict_variants(models, signals_cand, cfg.xvalues, cfg)

    if case_idx_list is None:
        print(f"Scanning candidates to auto-select the top {n_cases} "
              f"illustrative case(s)...")
        selected_indices = pick_illustrative_cases(model_name, cfg, params_cand,
                                                    signals_arr, rng, n_cases)
    else:
        selected_indices = case_idx_list

    out_dir = outdir(model_name)

    results = []
    for rank, selected_idx in enumerate(selected_indices):
        print(f"\n--- Case {rank + 1}/{len(selected_indices)}: "
              f"candidate #{selected_idx} ---")

        true_params = params_cand[selected_idx]
        signal      = signals_arr[selected_idx]
        nn_pred     = ({k: p[selected_idx] for k, p in nn_preds_all.items()}
                       if nn_preds_all is not None else None)

        print(f"Ground truth: "
              f"{dict(zip(cfg.param_names, np.round(true_params, 4)))}")

        print("Running full restart pool for this case...")
        pool_params, pool_resid = multi_start_lsq_pool(
            signal, cfg.xvalues, cfg.bounds, N_RESTARTS_FULL, cfg, rng
        )
        best_idx = int(np.argmin(pool_resid))
        best_lsq = pool_params[best_idx]
        n_clusters = cluster_count(pool_params, cfg.bounds)
        print(f"  {n_clusters} distinct basin(s) found among {N_RESTARTS_FULL} restarts")

        # ---- corner plot: diagonal = 1D slices, lower triangle = 2D slices ----
        title = ('DCE-MRI: multiple local minima in the loss landscape'
                  if model_name == 'DCE' else
                  f'{model_name}: wide, shallow valley in the loss landscape')
        full_title = (f'H1 anecdote — {title}\n'
                     f'({n_clusters} distinct basin(s) among {N_RESTARTS_FULL} restarts, '
                     f'SNR={cfg.default_snr}, case #{selected_idx})')

        out_path = os.path.join(out_dir,
                                f'H1_loss_landscape_anecdote_case{selected_idx}.png')

        plot_corner(cfg, true_params, signal, pool_params, pool_resid, best_lsq,
                   nn_pred, GRID_RES, full_title, out_path)

        results.append(dict(case_idx=selected_idx, true_params=true_params,
                            pool_params=pool_params, pool_resid=pool_resid,
                            best_lsq=best_lsq, nn_pred=nn_pred,
                            n_clusters=n_clusters, out_path=out_path))

    print(f"\n=== Rendered {len(results)} case(s) — pick whichever looks clearest ===")
    for r in results:
        print(f"  case #{r['case_idx']}: {r['n_clusters']} basin(s), "
              f"best LSQ residual={np.min(r['pool_resid']):.4g}  -> {r['out_path']}")

    return results


def load_corner_data(d, cfg):
    """Rebuild the inputs of draw_corner() from one case's data folder."""
    meta = fd.load_meta(d)
    names = list(cfg.param_names)
    n = len(names)
    t1 = fd.load_table(d, 'loss_1d')
    t2 = fd.load_table(d, 'loss_2d')
    grids_1d, loss_1d = {}, {}
    for p, nm in enumerate(names):
        m = t1['parameter'] == nm
        grids_1d[p], loss_1d[p] = t1['value'][m], t1['loss'][m]
    grids_2d, loss_2d = {}, {}
    for r in range(1, n):
        for c in range(r):
            gx, gy, z = fd.grid2d_from_table(t2, names[c], names[r])
            grids_2d[(r, c)], loss_2d[(r, c)] = (gx, gy), z

    mk = fd.load_table(d, 'markers')
    labels = list(mk['name'])

    def row(label):
        i = labels.index(label)
        return np.array([mk[nm][i] for nm in names])

    true_params = row('ground_truth')
    best_lsq = row('best_lsq')
    nn_keys = [f"nn_{v['key']}" for v in NN_VARIANTS]
    nn_pred = ({v['key']: row(k) for v, k in zip(NN_VARIANTS, nn_keys)}
               if all(k in labels for k in nn_keys) else None)
    rs = fd.load_table(d, 'restarts')
    pool_params = np.column_stack([rs[nm] for nm in names])
    return dict(true_params=true_params, pool_params=pool_params,
                best_lsq=best_lsq, nn_pred=nn_pred, grids_1d=grids_1d,
                loss_1d=loss_1d, grids_2d=grids_2d, loss_2d=loss_2d,
                vmin=meta['vmin'], vmax=meta['vmax'], title=meta['title'])


def replot(model_name='DCE', case_idx=None):
    """Re-draw the saved case(s) from their data files (no computation)."""
    apply_style()
    cfg = get_model_config(model_name)
    out_dir = outdir(model_name)
    root = os.path.join(out_dir, 'data')
    prefix = 'H1_loss_landscape_anecdote_case'
    stems = sorted(x for x in os.listdir(root) if x.startswith(prefix))
    if case_idx is not None:
        wanted = {f'{prefix}{i}' for i in case_idx}
        stems = [x for x in stems if x in wanted]
    if not stems:
        print(f'No saved loss-landscape data found under {root}')
        return
    for stem in stems:
        data = load_corner_data(os.path.join(root, stem), cfg)
        draw_corner(cfg, out_path=os.path.join(out_dir, stem + '.png'), **data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IVIM', choices=['IVIM', 'DCE'])
    parser.add_argument('--case_idx', type=int, nargs='+', default=None,
                        help='Manually pick one or more specific candidate '
                             'indices instead of auto-selecting illustrative '
                             'cases (e.g. --case_idx 3 7 12).')
    parser.add_argument('--n_cases', type=int, default=N_CASES_TO_PLOT,
                        help='Number of illustrative cases to auto-select '
                             'and render when --case_idx is not given '
                             '(default: %(default)s).')
    parser.add_argument('--no_nn', action='store_true',
                        help='Skip NN training/overlay (faster).')
    parser.add_argument('--replot', action='store_true',
                        help='Re-draw the saved case(s) from the data files '
                             '(no computation). Combine with --case_idx to '
                             'pick specific cases.')
    args = parser.parse_args()
    if args.replot:
        replot(args.model, case_idx=args.case_idx)
    else:
        run_experiment(args.model, case_idx=args.case_idx, include_nn=not args.no_nn,
                       n_cases=args.n_cases)