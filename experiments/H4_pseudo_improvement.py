import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from backend.QMRI.model_registry import get_model_config
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.experiment_utils import (
    normalize_params, rmse_normalized, get_prior_mean, signals_to_array,
    compute_crlb,
)
from backend.plot_style import (
    apply_style, COLORS, PANEL, TICK_SMALL, LEGEND_SMALL, param_label,
    param_label_bold, line_kw, outdir, save_fig,
)
from backend import figure_data as fd

N_TRAIN = 100000
N_TEST  = 1000
N_BINS  = 3
MAX_RESTARTS = 20

# Units shown on the general (all-parameter) summary's y-axes. Variance
# ratio and Pearson r are both scale-free ratios/correlations, labeled
# "a.u." for consistency with how every other genuinely-scale-free or
# model-dependent quantity in this project is labeled (residuals in
# H1/H2, and now these); RMSE here is computed on bounds-normalized
# ([0, 1]) parameters, so it is also unitless in the sense of not
# depending on the parameter's physical scale, but is kept as its own
# label since "normalized [0-1]" is more informative than "a.u." for a
# quantity people might otherwise expect to be in physical units.
METRIC_UNITS = {
    'var_ratio': 'a.u.',
    'corr':      'a.u.',
    'rmse':      'normalized [0-1]',
}


# The estimators compared in every H6 figure: (key, legend label, colour,
# marker). Keys are used as suffixes of the per-bin statistics (e.g.
# 'corr_lsq', 'rmse_sup', 'var_ratio_self').
ESTIMATORS = [dict(key='lsq', label='msLSQ', color=COLORS['msLSQ'], marker='o')] + [
    dict(key=v['key'], label=v['label'], color=v['color'], marker=v['marker'])
    for v in NN_VARIANTS
]


# Noise standard deviation (per real/imaginary channel) that each model's
# simulator adds, as a function of the noiseless signal and the SNR. Keep in
# sync with the simulators (ivim_model._simulate_ivim_signal_single and
# dce_model._simulate_dce_signal_single):
#   IVIM: absolute noise, sigma = 1/SNR (the signal is NOT renormalized, so
#         S0 in [0.5, 1.5] changes the effective SNR of a sample)
#   DCE : sigma = max(noiseless signal)/SNR
NOISE_SIGMA = {
    'IVIM': lambda signal, snr: 1.0 / snr,
    'DCE':  lambda signal, snr: float(np.max(signal)) / snr,
}


def _sigma_for_sample(params_i: np.ndarray, cfg, snr: float) -> float:
    """
    Noise standard deviation the simulator used for this parameter vector, in
    the units of the forward-model signal, for ``compute_crlb``.

    The CRLB scales with sigma**2, so sigma must be the *actual* simulated
    noise level: any other (e.g. amplitude-based) value that varies with the
    sample distorts the CRLB bins by signal shape/amplitude instead of
    identifiability. Raises for a model without an entry in NOISE_SIGMA
    rather than guessing.

    Parameters
    ----------
    params_i : (n_params,) physical parameter vector
    cfg      : ModelConfig
    snr      : float
    """
    name = str(cfg.name).upper()
    if name not in NOISE_SIGMA:
        raise ValueError(f"No simulator noise model registered for {cfg.name!r} "
                         f"in NOISE_SIGMA (known: {sorted(NOISE_SIGMA)}).")
    signal_i = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params_i],
        torch.tensor(cfg.xvalues, dtype=torch.float64),
    ).numpy()
    return NOISE_SIGMA[name](signal_i, snr)


def compute_residual(params, signal, xvalues, cfg):
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def multi_start_lsq(signal, xvalues, bounds, n_restarts, cfg, rng):
    """
    Multi-start LSQ (as in H1): run n_restarts random-initialization local
    fits and return the parameters with the lowest data residual, rather
    than a single fit from the prior mean. This ensures that when H6 finds
    LSQ railing against bounds, it is because that genuinely is the LSQ
    global optimum in a low-information regime, not because a single-start
    fit got stuck in a local minimum on the way there.
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best_params   = None
    best_residual = np.inf

    for r in range(n_restarts):
        init = rng.uniform(lo, hi)
        params_est = cfg.fit_lsq_1d(xvalues, signal, init)
        residual = compute_residual(params_est, signal, xvalues, cfg)
        if residual < best_residual:
            best_residual = residual
            best_params   = params_est.copy()

    return best_params


def _safe_filename(pname: str) -> str:
    return (pname.replace('*', 'star')
                 .replace('/', '_')
                 .replace(' ', '_'))


def _bin_tick_labels(bin_stats, x_pos):
    return [
        (f'Bin {b}\n{bin_stats[b - 1]["crlb_sqrt"]:.2f}'
         if bin_stats[b - 1] is not None else f'Bin {b}\n\u2014')
        for b in x_pos
    ]

_METRICS = ('corr', 'rmse', 'var_ratio')
_BIN_STAT_KEYS = (('n_pts', 'crlb_sqrt', 'var_true')
                  + tuple(f"{m}_{e['key']}" for e in ESTIMATORS for m in _METRICS))


def _save_bin_stats(fig_file, cfg, bin_stats_by_param, model_name):
    """
    bin_stats.csv: one row per (parameter, CRLB bin) with every statistic the
    summary panels use. Suffix _lsq = LSQ, _sup / _self = supervised /
    self-supervised NN. A bin with too few samples
    has n_pts = 0 and NaN statistics (the figure leaves it blank). Returns the
    data folder.
    """
    d = fd.data_dir_for(fig_file)
    cols = dict(parameter=[], bin=[], **{k: [] for k in _BIN_STAT_KEYS})
    for pname, stats in bin_stats_by_param.items():
        for b, s in enumerate(stats, start=1):
            cols['parameter'].append(pname); cols['bin'].append(b)
            for k in _BIN_STAT_KEYS:
                cols[k].append((0 if k == 'n_pts' else float('nan')) if s is None else s[k])
    fd.save_table(d, 'bin_stats', cols)
    fd.save_meta(d, model=model_name, snr=cfg.default_snr,
                 param_names=list(cfg.param_names), n_bins=N_BINS,
                 n_test=N_TEST, max_restarts=MAX_RESTARTS,
                 bins='equal-count percentile bins of the per-sample CRLB (bin 1 = lowest CRLB)',
                 suffixes='_lsq = multi-start LSQ, _sup = supervised NN, '
                          '_self = self-supervised NN')
    return d


def plot_all_parameter_summary(cfg, all_bin_stats, model_name, out, save_data=True):
    """
    One consolidated figure covering every parameter at once: rows =
    parameters, columns = metric (variance ratio, Pearson correlation,
    RMSE), each panel showing LSQ vs NN across CRLB bins.

    This complements (does not replace) the per-parameter figures built
    in the main loop above, which pair each parameter's own scatter
    detail with the correct DCE math notation in a compact caption.
    Reading off "which parameters show the pseudo-improvement pattern"
    from those means opening n_params separate files and comparing
    metrics by eye; this figure puts all of them on one page instead.

    VarRatio and correlation are scale-free (bounded, comparable across
    parameters), so their columns share fixed y-limits across every row.
    RMSE is in each parameter's own normalized-but-not-scale-free units,
    so its column is left free to autoscale per row rather than forcing
    a shared axis that would flatten whichever parameter has the
    smallest RMSE range.
    """
    param_names = cfg.param_names
    n_params = len(param_names)
    metrics = ['var_ratio', 'corr', 'rmse']
    metric_labels = ['Variance ratio', 'Pearson r', 'RMSE (normalized [0-1])']

    if save_data:
        _save_bin_stats(f'{out}/H6_summary_all_parameters.png', cfg,
                        all_bin_stats, model_name)

    fig, axes = plt.subplots(n_params, 3, figsize=(PANEL * 3, PANEL * n_params),
                             squeeze=False)

    for row, pname in enumerate(param_names):
        bin_stats = all_bin_stats[pname]
        n_bins = len(bin_stats)
        x_pos = np.arange(1, n_bins + 1)

        for col, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
            ax = axes[row, col]
            ax.axhline(0.0, color='black', linestyle='-', linewidth=0.7, alpha=0.35)
            for e in ESTIMATORS:
                vals = [s[f"{metric}_{e['key']}"] if s is not None else np.nan
                        for s in bin_stats]
                ax.plot(x_pos, vals, label=e['label'], zorder=3,
                        **line_kw(e['color'], e['marker'], '-'))

            ax.set_xticks(x_pos)
            ax.set_xticklabels(_bin_tick_labels(bin_stats, x_pos), fontsize=TICK_SMALL)
            ax.set_xlim(0.5, n_bins + 0.5)
            ax.set_xlabel('CRLB bin (mean $\\sqrt{CRLB}$)', fontsize=TICK_SMALL + 2)

            if metric == 'var_ratio':
                ax.set_ylim(-0.15, 1.3)
            elif metric == 'corr':
                ax.set_ylim(-0.2, 1.05)
            # rmse: left to autoscale per row

            # y-axis: every subplot carries an explicit unit label; the
            # leftmost column additionally carries the row's parameter
            # name (in correct DCE math notation via param_label), since
            # that column also serves as the row label. Only the
            # parameter symbol itself is bolded (via param_label_bold's
            # LaTeX \mathbf{}, not this Text object's fontweight), so the
            # accompanying unit stays plain/normal weight rather than
            # being swept into bold along with it.
            unit = METRIC_UNITS[metric]
            if col == 0:
                # fontweight='normal' is explicit here (rather than relying
                # on the default) because leaving it unset let the "(a.u.)"
                # unit text pick up a bold weight from elsewhere in some
                # renders -- \mathbf{} inside param_label_bold()'s math mode
                # still overrides to bold for just the symbol regardless of
                # this Text object's own fontweight.
                ax.set_ylabel(f'{param_label(pname)}\n({unit})',
                              fontsize=FONT_LABEL, fontweight='normal')
            else:
                ax.set_ylabel(f'({unit})', fontsize=TICK_SMALL + 2)
            if row == 0:
                ax.set_title(mlabel)
            ax.legend(loc='lower left')

    save_fig(fig, f'{out}/H6_summary_all_parameters.png')
    plt.show()
    print(f"Saved: {out}/H6_summary_all_parameters.png")


FONT_LABEL = 22   # bold row label of the all-parameter summary


# --- compact (side-by-side) summary figure ---------------------------------
COMPACT_PANEL_W = 1.25    # inches per metric column
COMPACT_PANEL_H = 1.15    # inches per parameter row
COMPACT_LEFT_IN = 0.45    # extra width for the row (parameter) labels
COMPACT_TOP_IN  = 0.55    # extra height for the legend + column titles
COMPACT_RC = {
    'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 7,
    'axes.linewidth': 0.6, 'xtick.major.size': 2.0, 'ytick.major.size': 2.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5,
}


def plot_all_parameter_summary_compact(cfg, all_bin_stats, model_name, out,
                                       save_data=True):
    """
    Compact version of plot_all_parameter_summary() for placing the IVIM and
    DCE figures side by side in a manuscript. Same rows (parameters), same
    columns (variance ratio, Pearson r, RMSE), same statistics; only the
    layout differs (see change log item 13). Saved as PNG (300 dpi) and PDF
    (vector; scales without loss to the final column width).

    Requires matplotlib >= 3.7 (``loc='outside upper center'`` legend).
    """
    from matplotlib.ticker import MaxNLocator

    param_names = cfg.param_names
    n_params = len(param_names)
    metrics = ['var_ratio', 'corr', 'rmse']
    col_titles = ['Variance ratio', 'Pearson $r$', 'RMSE (norm.)']

    # shared variance-ratio upper limit that covers the data (never below 1.3)
    vr_all = [s[f"var_ratio_{e['key']}"] for st in all_bin_stats.values()
              for s in st if s is not None for e in ESTIMATORS]
    vr_all = [v for v in vr_all if np.isfinite(v)]
    vr_top = max(1.3, 1.05 * max(vr_all)) if vr_all else 1.3

    if save_data:
        _save_bin_stats(f'{out}/H6_summary_all_parameters_compact.png', cfg,
                        all_bin_stats, model_name)

    fig_w = COMPACT_LEFT_IN + 3 * COMPACT_PANEL_W
    fig_h = COMPACT_TOP_IN + n_params * COMPACT_PANEL_H

    with plt.rc_context(COMPACT_RC):
        fig, axes = plt.subplots(n_params, 3, figsize=(fig_w, fig_h),
                                 squeeze=False, layout='constrained')
        lw, ms = 1.2, 3.5
        for row, pname in enumerate(param_names):
            bin_stats = all_bin_stats[pname]
            n_bins = len(bin_stats)
            x_pos = np.arange(1, n_bins + 1)
            tick_labels = [f'{s["crlb_sqrt"]:.2g}' if s is not None else '\u2014'
                           for s in bin_stats]

            for col, metric in enumerate(metrics):
                ax = axes[row, col]
                ax.axhline(0.0, color='black', linewidth=0.5, alpha=0.35)
                for e in ESTIMATORS:
                    vals = [s[f"{metric}_{e['key']}"] if s is not None else np.nan
                            for s in bin_stats]
                    ax.plot(x_pos, vals, label=e['label'], zorder=3,
                            color=e['color'], marker=e['marker'], linestyle='-',
                            linewidth=lw, markersize=ms,
                            markeredgecolor='black', markeredgewidth=0.4)

                ax.set_xticks(x_pos)
                ax.set_xticklabels(tick_labels)
                ax.set_xlim(0.5, n_bins + 0.5)

                if metric == 'var_ratio':
                    ax.set_ylim(-0.1, vr_top)
                    ax.yaxis.set_major_locator(MaxNLocator(4))
                elif metric == 'corr':
                    ax.set_ylim(-0.2, 1.05)
                    ax.set_yticks([0, 0.5, 1.0])
                else:
                    ax.margins(y=0.15)
                    ax.yaxis.set_major_locator(MaxNLocator(3))

                if row == 0:
                    ax.set_title(col_titles[col])
                if col == 0:
                    ax.set_ylabel(param_label(pname))
                if row == n_params - 1:
                    ax.set_xlabel('bin: mean $\\sqrt{CRLB}$')

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center',
                   ncol=len(ESTIMATORS), frameon=False, handlelength=1.5,
                   columnspacing=1.5)

        base = f'{out}/H6_summary_all_parameters_compact'
        fig.savefig(base + '.png', dpi=300)
        plt.show()
    print(f"Saved: {base}.png  ({fig_w:.2f} x {fig_h:.2f} in)")


# --- compact per-parameter scatter / density figure ------------------------
SCATTER_PANEL_IN  = 1.4    # inches per (square) panel
SCATTER_LEFT_IN   = 0.6    # row labels
SCATTER_RIGHT_IN  = 0.55   # one colorbar per row
SCATTER_TOP_IN    = 0.7    # caption + two-line column titles
SCATTER_BOTTOM_IN = 0.75   # x label + legend
KDE_BW            = 0.25   # KDE bandwidth factor (Scott's factor is ~0.25 for 333 points)
POINT_SIZE        = 6      # marker area (pt^2) at the final printed size


def _density_cmap(color):
    """Sequential colormap from a light tint of ``color`` (sparse) through
    ``color`` to a dark shade of it (dense), so density reads as 'darker'
    in the estimator's own colour."""
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    base  = np.array(to_rgb(color))
    light = 1.0 - 0.6 * (1.0 - base)
    dark  = base * 0.40
    return LinearSegmentedColormap.from_list('density', [light, base, dark])


def _point_density(x, y):
    """Gaussian-KDE density at each point (x, y). Falls back to a constant
    when the KDE is undefined (fewer than 3 points, or all points identical
    along one axis -> singular covariance)."""
    from scipy.stats import gaussian_kde
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3:
        return np.ones(len(x))
    try:
        return gaussian_kde(np.vstack([x, y]), bw_method=KDE_BW)(np.vstack([x, y]))
    except (np.linalg.LinAlgError, ValueError):
        return np.ones(len(x))


def plot_scatter_figure(pname, bin_stats, model_name, out):
    """
    Draw one parameter's estimator (rows: msLSQ, NN sup., NN self-sup.) x
    CRLB-bin (columns) grid of estimate-vs-truth scatter plots. Built at its
    final printed size with small text (like the compact summary figure) so it
    stays readable when scaled down. Each panel scatters the estimate against
    the ground truth (both normalized to [0, 1]) with every point coloured by
    its local density (Gaussian KDE, normalized within the panel; light =
    sparse, dark = dense, densest drawn on top); dashed black = y = x (perfect tracking), dotted gray = the
    parameter bounds (LSQ railing collapses onto them). In-panel text: Pearson
    r, normalized RMSE and variance ratio (VR).
    Edit the styling here, then re-draw with ``--replot``.
    """
    from matplotlib.lines import Line2D

    lims  = (-0.05, 1.05)
    n_est = len(ESTIMATORS)
    fig_w = SCATTER_LEFT_IN + SCATTER_RIGHT_IN + SCATTER_PANEL_IN * N_BINS
    fig_h = SCATTER_TOP_IN + SCATTER_BOTTOM_IN + SCATTER_PANEL_IN * n_est

    with plt.rc_context(COMPACT_RC):
        fig, axes = plt.subplots(n_est, N_BINS, figsize=(fig_w, fig_h),
                                 squeeze=False, layout='constrained')

        for row, e in enumerate(ESTIMATORS):
            cmap = _density_cmap(e['color'])
            dens = {}      # col -> (x, y, density), drawn after the row's scale is known
            for col, s in enumerate(bin_stats):
                ax = axes[row, col]
                if s is None:
                    ax.axis('off')
                    continue
                ax.set_box_aspect(1)

                z = _point_density(s['t'], s['est'][e['key']])
                dens[col] = (s['t'], s['est'][e['key']], z / np.max(z))

                # reference lines: bounds (where LSQ rails) and y = x
                ax.axhline(0.0, color='gray', linestyle=':', linewidth=0.9,
                           alpha=0.9, zorder=3)
                ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.9,
                           alpha=0.9, zorder=3)
                ax.plot(lims, lims, 'k--', linewidth=0.7, zorder=3)
                ax.set_xlim(lims); ax.set_ylim(lims)
                ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
                ax.set_xticklabels(['0', '0.5', '1']); ax.set_yticklabels(['0', '0.5', '1'])

                k = e['key']
                ax.text(0.97, 0.03,
                        f"r={s[f'corr_{k}']:.2f}\n"
                        f"RMSE={s[f'rmse_{k}']:.2f}\n"
                        f"VR={s[f'var_ratio_{k}']:.2f}",
                        transform=ax.transAxes, ha='right', va='bottom',
                        fontsize=5.5, zorder=5, linespacing=1.15,
                        bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                  ec='none', alpha=0.75))

                if row == 0:
                    ax.set_title(f"bin {col + 1}\n"
                                 f"$\\sqrt{{CRLB}}$={s['crlb_sqrt']:.2g}")
                if col == 0:
                    ax.set_ylabel(f"{e['label']}\nestimate (norm.)")
                else:
                    ax.tick_params(labelleft=False)
                if row == n_est - 1:
                    ax.set_xlabel('ground truth (norm.)')
                else:
                    ax.tick_params(labelbottom=False)

            # density is normalized within each panel (darkest = that panel's
            # densest region), so structure is visible in every panel even if
            # one bin has a much tighter cluster; the colorbar is therefore
            # only "low" to "high". Points drawn sparse -> dense so the dense
            # ones stay visible.
            if dens:
                vmin, vmax = 0.0, 1.0
                for col, (x, y, z) in dens.items():
                    o = np.argsort(z)
                    sc = axes[row, col].scatter(
                        np.asarray(x)[o], np.asarray(y)[o], c=z[o], cmap=cmap,
                        vmin=vmin, vmax=vmax, s=POINT_SIZE, linewidths=0,
                        zorder=2)
                cb = fig.colorbar(sc, ax=axes[row, :], location='right',
                                  shrink=0.7, aspect=12, pad=0.02)
                cb.set_label('point density', fontsize=6)
                cb.set_ticks([vmin, vmax]); cb.set_ticklabels(['low', 'high'])
                cb.ax.tick_params(labelsize=5.5, length=0)

        handles = [Line2D([0], [0], color='black', linestyle='--', linewidth=0.8,
                          label='y = x'),
                   Line2D([0], [0], color='gray', linestyle=':', linewidth=1.0,
                          label='bounds (0, 1)')]
        fig.legend(handles=handles, loc='outside lower center', ncol=2,
                   frameon=False, handlelength=1.8, columnspacing=1.5)
        fig.suptitle(f'{param_label(pname)} ({model_name})', fontsize=8,
                     x=0.02, ha='left')

        path = f'{out}/H6_scatter_{_safe_filename(pname)}.png'
        fig.savefig(path, dpi=300)
        plt.show()
    print(f"Saved: {path}  ({fig_w:.2f} x {fig_h:.2f} in)")


def _load_bin_stats(d, pname, with_points=False):
    """Rebuild the per-bin statistics dicts for one parameter from
    bin_stats.csv (and, if requested, the scatter points from points.csv)."""
    t = fd.load_table(d, 'bin_stats')
    pts = fd.load_table(d, 'points') if with_points else None
    stats = []
    for i in np.flatnonzero(t['parameter'] == pname):
        if t['n_pts'][i] == 0:
            stats.append(None)
            continue
        s = {k: t[k][i] for k in _BIN_STAT_KEYS}
        if with_points:
            m = pts['bin'] == t['bin'][i]
            s['t'] = pts['ground_truth_norm'][m]
            s['est'] = {e['key']: pts[f"{e['key']}_norm"][m] for e in ESTIMATORS}
        stats.append(s)
    return stats


def replot(model_name: str = 'IVIM'):
    """Re-draw all H6 figures from the saved data files (no computation)."""
    global N_BINS
    apply_style()
    cfg = get_model_config(model_name)
    out = outdir(model_name)
    d_sum = fd.data_dir_for(f'{out}/H6_summary_all_parameters.png')
    N_BINS = int(fd.load_meta(d_sum)['n_bins'])
    all_bin_stats = {p: _load_bin_stats(d_sum, p) for p in cfg.param_names}
    for pname in cfg.param_names:
        d_fig = fd.data_dir_for(f'{out}/H6_scatter_{_safe_filename(pname)}.png')
        plot_scatter_figure(pname, _load_bin_stats(d_fig, pname, True), model_name, out)
    plot_all_parameter_summary(cfg, all_bin_stats, model_name, out, save_data=False)
    plot_all_parameter_summary_compact(cfg, all_bin_stats, model_name, out, save_data=False)


def run_experiment(model_name: str = 'IVIM'):
    apply_style()
    cfg = get_model_config(model_name)
    snr = cfg.default_snr
    rng = np.random.default_rng(seed=42)

    print(f"\n=== H6 — Pseudo-Improvement (CRLB) | model={model_name} ===")

    params_train, coeffs_train = cfg.sample_parameters(N_TRAIN, cfg.param_distribution)
    signals_train = cfg.simulate_signals(params_train, cfg.xvalues, snr)
    params_test, _ = cfg.sample_parameters(N_TEST, cfg.param_distribution)
    signals_test   = cfg.simulate_signals(params_test, cfg.xvalues, snr)
    signals_test_arr = signals_to_array(signals_test)

    print("Training NNs (supervised + self-supervised)...")
    models    = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg)
    params_nn = predict_variants(models, signals_test, cfg.xvalues, cfg)   # {key: (N, n_params)}

    print(f"Running multi-start LSQ ({MAX_RESTARTS} restarts/sample)...")
    params_lsq = np.array([
        multi_start_lsq(signals_test_arr[i], cfg.xvalues, cfg.bounds,
                         MAX_RESTARTS, cfg, rng)
        for i in range(N_TEST)
    ])

    params_test_norm = normalize_params(params_test, cfg.bounds)
    # raw / normalized estimates of every estimator, keyed like ESTIMATORS
    est_raw  = {'lsq': params_lsq, **params_nn}
    est_norm = {k: normalize_params(p, cfg.bounds) for k, p in est_raw.items()}
    training_mean      = np.mean(params_train, axis=0)
    training_mean_norm = normalize_params(training_mean[None, :], cfg.bounds)[0]

    # ------------------------------------------------------------------
    # Per-sample sigma: scales noise to the actual signal amplitude so
    # that the CRLB reflects true identifiability, not signal magnitude.
    # ------------------------------------------------------------------
    print("Computing CRLB per voxel...")
    crlb_all = np.array([
        compute_crlb(
            params_test[i],
            cfg.xvalues,
            _sigma_for_sample(params_test[i], cfg, snr),   # FIX: per-sample sigma
            cfg.bounds,
            cfg.parameters_to_signal,
        )
        for i in range(N_TEST)
    ])

    out = outdir(model_name)

    # Raw material behind every H6 figure (per-sample truth, estimates, CRLB)
    d_raw = fd.data_dir_for(f'{out}/H6_raw.png')
    fd.save_arrays(d_raw, 'arrays', params_test=params_test,
                   params_test_norm=params_test_norm, params_lsq=params_lsq,
                   crlb_all=crlb_all, training_mean=training_mean,
                   **{f'params_nn_{k}': p for k, p in params_nn.items()})
    fd.save_meta(d_raw, model=model_name, snr=snr, param_names=list(cfg.param_names),
                 bounds=[list(b) for b in cfg.bounds], n_train=N_TRAIN,
                 n_test=N_TEST, n_bins=N_BINS, max_restarts=MAX_RESTARTS, seed=42)

    # ------------------------------------------------------------------
    # Per-parameter scatter-vs-truth figure (primary output).
    # ------------------------------------------------------------------
    all_bin_stats = {}   # collected across parameters, for the all-parameter
                          # summary figure built after this loop

    for k, pname in enumerate(cfg.param_names):
        crlb_k    = crlb_all[:, k]
        crlb_sqrt = np.sqrt(crlb_k)

        true_k = params_test_norm[:, k]
        est_k  = {ek: p[:, k] for ek, p in est_norm.items()}

        bin_edges = np.percentile(crlb_k, np.linspace(0, 100, N_BINS + 1))
        bin_idx   = np.clip(np.digitize(crlb_k, bin_edges, right=True), 1, N_BINS)

        lims = (-0.05, 1.05)

        # First pass: gather everything the scatter panels and the
        # all-parameter summary figure both need, computed once here.
        bin_stats = []
        for b in range(1, N_BINS + 1):
            mask = bin_idx == b
            if mask.sum() < 2:
                bin_stats.append(None)
                continue
            t = true_k[mask]
            var_true = np.var(t)
            stats = dict(t=t, est={ek: a[mask] for ek, a in est_k.items()},
                         var_true=var_true, n_pts=mask.sum(),
                         crlb_sqrt=np.mean(crlb_sqrt[mask]))
            for ek, a in est_k.items():
                x = a[mask]
                stats[f'corr_{ek}'] = np.corrcoef(x, t)[0, 1] if len(t) > 2 else np.nan
                # per-bin normalized RMSE via the shared helper (as in H1):
                # raw estimates + normalized truth, parameter k's component
                stats[f'rmse_{ek}'] = rmse_normalized(
                    est_raw[ek][mask], params_test_norm[mask], cfg.bounds)[k]
                stats[f'var_ratio_{ek}'] = np.var(x) / var_true if var_true > 0 else np.nan
            bin_stats.append(stats)

        all_bin_stats[pname] = bin_stats

        # ------------------------------------------------------------------
        # Per-parameter scatter grid: 2 rows (LSQ / NN) x N_BINS columns.
        # The two-sided headline summary panel that used to sit above this
        # grid has been removed (change log item 12) -- its content is
        # exactly what plot_all_parameter_summary() already shows, one
        # panel per metric, across every parameter at once.
        # ------------------------------------------------------------------
        # Save the raw numbers behind this figure (before plotting)
        d_fig = _save_bin_stats(f'{out}/H6_scatter_{_safe_filename(pname)}.png', cfg,
                                {pname: bin_stats}, model_name)
        pts = dict(bin=[], ground_truth_norm=[],
                   **{f"{e['key']}_norm": [] for e in ESTIMATORS})
        for b, s in enumerate(bin_stats, start=1):
            if s is None:
                continue
            m = len(s['t'])
            pts['bin'] += [b] * m
            pts['ground_truth_norm'] += list(s['t'])
            for e in ESTIMATORS:
                pts[f"{e['key']}_norm"] += list(s['est'][e['key']])
        fd.save_table(d_fig, 'points', pts)

        plot_scatter_figure(pname, bin_stats, model_name, out)

    plot_all_parameter_summary(cfg, all_bin_stats, model_name, out)
    plot_all_parameter_summary_compact(cfg, all_bin_stats, model_name, out)

    return dict(crlb_all=crlb_all, params_lsq=params_lsq,
                params_nn=params_nn, params_test=params_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IVIM', choices=['IVIM', 'DCE'])
    parser.add_argument('--replot', action='store_true',
                        help='Re-draw all figures from the saved data files '
                             '(no computation).')
    args = parser.parse_args()
    if args.replot:
        replot(args.model)
    else:
        run_experiment(args.model)