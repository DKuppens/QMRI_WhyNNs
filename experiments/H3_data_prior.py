import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from backend.QMRI.model_registry import get_model_config
from backend.experiment_utils import (
    normalize_params, rmse_normalized, signals_to_array,
)
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.plot_style import (
    apply_style, COLORS, LEGEND_SMALL, param_label, line_kw, heatmap_cmap,
    outdir, save_fig,
)
from backend import figure_data as fd

N_TRAIN = 100000
N_TEST  = 1000
MAX_RESTARTS = 20
N_SEEDS = 5   # number of independent NN training runs per condition (NNcor/NNuncor)

# Panel size for this (denser) figure, in inches
PANEL_W, PANEL_H = 7, 6.5

# Marker convention: circle = estimator that saw correlation information
# (NNcor), square = one that didn't (NNuncor, msLSQ).
MARKER_COV    = 'o'   # NNcor
MARKER_NO_COV = 's'   # NNuncor, msLSQ

# Line style per NN type (key -> style); msLSQ is dash-dot.
NN_LS = {'sup': '-', 'self': '--'}

CONDITIONS = ('cor', 'uncor')     # NN training data
TESTS      = ('corr', 'uncorr')   # test data


def _default_correlation_matrix(n_params: int) -> np.ndarray:
    """
    Identity (no correlation) as a safe default.
    Override for specific models if physiologically motivated correlations exist.
    """
    return np.eye(n_params)


def _ivim_correlation_matrix():
    return np.array([
        [ 1.00,  0.00,  0.00,  0.00],
        [ 0.00,  1.00,  0.50,  0.00],
        [ 0.00,  0.50,  1.00,  0.00],
        [ 0.00,  0.00,  0.00,  1.00],
    ])


def _dce_correlation_matrix(n_params: int) -> np.ndarray:
    """
    Correlation matrix for the standard/extended Tofts DCE parameterization.
    Assumed parameter order: [ke, ve, vp, dt].
    """
    if n_params == 3:
        return np.array([
            [ 1.00,  0.00,  0.00],
            [ 0.00,  1.00,  0.50],
            [ 0.00,  0.50,  1.00],
        ])
    elif n_params == 4:
        return np.array([
            [ 1.00,  0.00,  0.00,  0.00],
            [ 0.00,  1.00,  0.50,  0.00],
            [ 0.00,  0.50,  1.00,  0.00],
            [ 0.00,  0.00,  0.00,  1.00],
        ])
    return _default_correlation_matrix(n_params)


def get_correlation_matrix(model_name: str, n_params: int) -> np.ndarray:
    if model_name == 'IVIM':
        return _ivim_correlation_matrix()
    if model_name == 'DCE':
        return _dce_correlation_matrix(n_params)
    return _default_correlation_matrix(n_params)


def sample_correlated(N, cfg, corr):
    """
    Sample N parameter sets with UNIFORM marginals on ``cfg.bounds`` (the same
    marginals as ``cfg.sample_parameters``, used by H1/H2/H6) and a chosen
    dependence between parameters: a Gaussian copula. Correlated standard
    normals z ~ N(0, corr) are mapped through the normal CDF to u in [0, 1]
    (uniform marginals), then scaled to the bounds. With ``corr`` = identity
    the parameters are independent uniforms, i.e. the same distribution as the
    registry sampler.

    ``corr`` is the correlation in the Gaussian (copula) space; the Pearson
    correlation of the resulting parameters is slightly smaller (6/pi *
    arcsin(r/2), e.g. 0.483 for r = 0.5), so use the *realized* correlation of
    the samples (np.corrcoef) when comparing against it.

    Returns (params, coeffs): physical parameters and coefficients in [0, 1].
    """
    from scipy.stats import norm
    corr = np.asarray(corr, dtype=float)
    z = np.random.multivariate_normal(np.zeros(corr.shape[0]), corr, size=N)
    coeffs = norm.cdf(z)
    bounds_arr = np.array(cfg.bounds, dtype=float)
    params = bounds_arr[:, 0] + coeffs * (bounds_arr[:, 1] - bounds_arr[:, 0])
    return params, coeffs


def compute_residual(params, signal, xvalues, cfg):
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def multi_start_lsq(signal, xvalues, bounds, n_restarts, cfg, rng):
    """
    Multi-start LSQ (as in H1/H2/H6): run n_restarts random-initialization
    local fits (``cfg.fit_lsq_1d``) and return the parameters with the lowest
    data residual.
    """
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    best_params   = None
    best_residual = np.inf

    for _ in range(n_restarts):
        init = rng.uniform(lo, hi)
        params_est = np.asarray(cfg.fit_lsq_1d(xvalues, signal, init))
        residual = compute_residual(params_est, signal, xvalues, cfg)
        if residual < best_residual:
            best_residual = residual
            best_params   = params_est.copy()

    return best_params


def _nn_name(cond, variant):
    """Legend label, e.g. 'NNcor sup.' / 'NNuncor self-sup.'."""
    return variant['label'].replace('NN', f'NN{cond}', 1)


def save_figure_data(fig_file, cfg, model_name, snr, *, corr, corr_nominal,
                     output_corrmat,
                     pct_learned, pct_learned_clipped, plot_names,
                     rmse_nn, rmse_lsq_on_corr, rmse_lsq_on_uncorr,
                     params_nncor_on_corr_pooled, params_test_corr,
                     params_test_uncorr):
    """
    Store the numbers behind the H5 figure under ``<dir>/data/<figure name>/``.
    All parameters are saved; ``meta.json['plotted_parameters']`` says which
    ones the figure actually shows. ``rmse_nn[variant][cond][test]`` is a
    (N_SEEDS, n_params) array (cond in cor/uncor = NN training data, test in
    corr/uncorr = test data); the per-variant dicts (``output_corrmat``,
    ``pct_learned``, ...) are keyed by variant key ('sup', 'self').

      nn_rmse_per_seed.csv   (parameter, nn_variant, network, test_data, seed, rmse)
      slopegraph.csv         row 0: (parameter, estimator, test_data, value,
                             err_lo, err_hi); NN error bars = min/max over
                             seeds, NaN for msLSQ
      correlation_heatmap.csv  row 1: training (realized) vs. nominal (copula)
                             vs. learned correlation per NN type
      arrays.npz             pooled NNcor predictions and the two test sets
    """
    d = fd.data_dir_for(fig_file)
    names = list(cfg.param_names)
    n = len(names)
    nan = float('nan')

    # --- NN RMSE for every seed ---
    cols = dict(parameter=[], nn_variant=[], network=[], test_data=[],
                seed=[], rmse=[])
    for v in NN_VARIANTS:
        for cond in CONDITIONS:
            for test in TESTS:
                arr = rmse_nn[v['key']][cond][test]
                for k in range(n):
                    for sd in range(arr.shape[0]):
                        for key, val in (('parameter', names[k]),
                                         ('nn_variant', v['key']),
                                         ('network', f'NN{cond}'),
                                         ('test_data', test), ('seed', sd),
                                         ('rmse', arr[sd, k])):
                            cols[key].append(val)
    fd.save_table(d, 'nn_rmse_per_seed', cols)

    # --- Row 0: slopegraph ---
    cols = dict(parameter=[], estimator=[], test_data=[], value=[],
                err_lo=[], err_hi=[])

    def add(k, est, test, val, lo=nan, hi=nan):
        for key, v_ in (('parameter', names[k]), ('estimator', est),
                        ('test_data', test), ('value', val),
                        ('err_lo', lo), ('err_hi', hi)):
            cols[key].append(v_)

    for k in range(n):
        for v in NN_VARIANTS:
            for cond in CONDITIONS:
                for test in TESTS:
                    a = rmse_nn[v['key']][cond][test][:, k]
                    add(k, f"NN{cond}_{v['key']}", test, a.mean(), a.min(), a.max())
        add(k, 'msLSQ', 'corr', rmse_lsq_on_corr[k])
        add(k, 'msLSQ', 'uncorr', rmse_lsq_on_uncorr[k])
    fd.save_table(d, 'slopegraph', cols)

    # --- Row 1: heatmaps ---
    cols = dict(row_param=[], col_param=[], training_corr=[], nominal_corr=[])
    for v in NN_VARIANTS:
        for pre in ('nn_corr', 'pct_learned', 'pct_learned_clipped'):
            cols[f"{pre}_{v['key']}"] = []
    for i in range(n):
        for j in range(n):
            cols['row_param'].append(names[i]); cols['col_param'].append(names[j])
            cols['training_corr'].append(corr[i, j])
            cols['nominal_corr'].append(corr_nominal[i, j])
            for v in NN_VARIANTS:
                k_ = v['key']
                cols[f'nn_corr_{k_}'].append(output_corrmat[k_][i, j])
                cols[f'pct_learned_{k_}'].append(pct_learned[k_][i, j])
                cols[f'pct_learned_clipped_{k_}'].append(pct_learned_clipped[k_][i, j])
    fd.save_table(d, 'correlation_heatmap', cols)

    arrays = dict(params_test_corr=params_test_corr,
                  params_test_uncorr=params_test_uncorr, training_corr=corr,
                  nominal_corr=corr_nominal)
    for v in NN_VARIANTS:
        arrays[f"params_nncor_on_corr_pooled_{v['key']}"] = \
            params_nncor_on_corr_pooled[v['key']]
    fd.save_arrays(d, 'arrays', **arrays)
    fd.save_meta(d, model=model_name, snr=snr, param_names=names,
                 plotted_parameters=list(plot_names),
                 nn_variants=[v['key'] for v in NN_VARIANTS],
                 n_train=N_TRAIN, n_test=N_TEST, n_seeds=N_SEEDS,
                 max_restarts=MAX_RESTARTS,
                 note='Training/test parameter sets are drawn with the global '
                      'numpy RNG (unseeded); these files are the only exact '
                      'record of the plotted numbers.')
    print(f"Figure data saved under {d}")


def plot_figure(cfg, fig_file, plot_idx, *, pct_learned_clipped, rmse_nn,
                rmse_lsq_on_corr, rmse_lsq_on_uncorr, **_unused):
    """Draw the H5 figure from computed (or reloaded) numbers. Edit the
    styling here, then re-draw with ``--replot``."""
    plot_names = [cfg.param_names[k] for k in plot_idx]
    n_plot     = len(plot_names)
    total_cols = max(len(NN_VARIANTS), n_plot)   # row 1: one heatmap per NN type

    fig, axes = plt.subplots(2, total_cols,
                             figsize=(PANEL_W * total_cols, PANEL_H * 2),
                             squeeze=False)

    # --- Row 0: per-parameter slopegraph (msLSQ, and NNcor / NNuncor of each
    # NN type), min-max error bars across seeds for the NNs. One shared y
    # axis; y label and legend on the leftmost panel only. ---
    scenario_labels = ['correlated\ndata', 'uncorrelated\ndata']
    x_pos = [0, 1]
    for col, (pname, pidx) in enumerate(zip(plot_names, plot_idx)):
        ax = axes[0, col]

        for v in NN_VARIANTS:
            for cond, marker, colour in (('cor',   MARKER_COV,    v['color']),
                                         ('uncor', MARKER_NO_COV, v['color_uncor'])):
                a_c = rmse_nn[v['key']][cond]['corr'][:, pidx]
                a_u = rmse_nn[v['key']][cond]['uncorr'][:, pidx]
                means = np.array([a_c.mean(), a_u.mean()])
                lo    = np.array([a_c.min(),  a_u.min()])
                hi    = np.array([a_c.max(),  a_u.max()])
                ax.errorbar(x_pos, means, yerr=[np.clip(means - lo, 0, None),   # clip: float round-off
                                  np.clip(hi - means, 0, None)],
                            capsize=5, elinewidth=2,
                            label=_nn_name(cond, v),
                            **line_kw(colour, marker, NN_LS[v['key']]))

        ax.plot(x_pos, [rmse_lsq_on_corr[pidx], rmse_lsq_on_uncorr[pidx]],
                label='msLSQ',
                **line_kw(COLORS['msLSQ'], MARKER_NO_COV, '-.'))

        ax.set_xticks(x_pos); ax.set_xticklabels(scenario_labels)
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(param_label(pname))
        if col == 0:
            ax.set_ylabel('RMSE (normalized [0-1])')
            ax.legend(fontsize=LEGEND_SMALL)

    # align the y axes of the slopegraph panels
    row0 = [axes[0, c] for c in range(n_plot)]
    y_lo = min(a.get_ylim()[0] for a in row0)
    y_hi = max(a.get_ylim()[1] for a in row0)
    for c, ax in enumerate(row0):
        ax.set_ylim(y_lo, y_hi)
        if c > 0:
            ax.tick_params(axis='y', labelleft=False)

    for col in range(n_plot, total_cols):
        axes[0, col].axis('off')

    # --- Row 1: Test B heatmaps, one per NN type ---
    for i, v in enumerate(NN_VARIANTS):
        ax  = axes[1, i]
        sub = pct_learned_clipped[v['key']][:n_plot, :n_plot]
        im  = ax.imshow(sub, cmap=heatmap_cmap(), vmin=0, vmax=100, aspect='auto')
        cb  = plt.colorbar(im, ax=ax)
        cb.set_label('% of training correlation\nreproduced by NNcor', fontsize=16)
        cb.ax.tick_params(labelsize=16)
        ax.set_xticks(range(n_plot)); ax.set_xticklabels([param_label(n) for n in plot_names])
        ax.set_yticks(range(n_plot)); ax.set_yticklabels([param_label(n) for n in plot_names])
        for r in range(n_plot):
            for c in range(n_plot):
                val = sub[r, c]
                txt = '—' if r == c else ('n/a' if np.isnan(val) else f'{val:.0f}%')
                ax.text(c, r, txt, ha='center', va='center', fontsize=16, color='black')
        ax.set_title(f"Correlation learned by {_nn_name('cor', v)}\n"
                     f"(pooled over {N_SEEDS} seeds)", fontsize=16)

    for col in range(len(NN_VARIANTS), total_cols):
        axes[1, col].axis('off')

    save_fig(fig, fig_file)
    plt.show()


def load_figure_data(fig_file, cfg):
    """Rebuild the keyword arguments of plot_figure() from the data files."""
    global N_SEEDS
    d = fd.data_dir_for(fig_file)
    meta = fd.load_meta(d)
    N_SEEDS = int(meta['n_seeds'])
    names = list(cfg.param_names)
    n = len(names)

    t = fd.load_table(d, 'nn_rmse_per_seed')
    rmse_nn = {}
    for v in NN_VARIANTS:
        rmse_nn[v['key']] = {}
        for cond in CONDITIONS:
            rmse_nn[v['key']][cond] = {}
            for test in TESTS:
                m = ((t['nn_variant'] == v['key']) & (t['network'] == f'NN{cond}')
                     & (t['test_data'] == test))
                out = np.empty((int(t['seed'][m].max()) + 1, n))
                for k, nm in enumerate(names):
                    mk = m & (t['parameter'] == nm)
                    out[t['seed'][mk], k] = t['rmse'][mk]
                rmse_nn[v['key']][cond][test] = out

    sg = fd.load_table(d, 'slopegraph')

    def lsq_vec(test):
        return np.array([sg['value'][(sg['parameter'] == nm) & (sg['estimator'] == 'msLSQ')
                                     & (sg['test_data'] == test)][0] for nm in names])

    hm = fd.load_table(d, 'correlation_heatmap')
    return dict(
        pct_learned_clipped={v['key']: hm[f"pct_learned_clipped_{v['key']}"].reshape(n, n)
                             for v in NN_VARIANTS},
        rmse_nn=rmse_nn,
        rmse_lsq_on_corr=lsq_vec('corr'),
        rmse_lsq_on_uncorr=lsq_vec('uncorr'),
        plot_names=list(meta['plotted_parameters']),
    )


def replot(model_name: str = 'IVIM'):
    """Re-draw the figure from the saved data files (no computation)."""
    apply_style()
    cfg = get_model_config(model_name)
    fig_file = f'{outdir(model_name)}/H5_inter_parameter_correlations.png'
    data = load_figure_data(fig_file, cfg)
    plot_idx = [list(cfg.param_names).index(p) for p in data['plot_names']]
    plot_figure(cfg, fig_file, plot_idx, **data)


def run_experiment(model_name: str = 'IVIM'):
    apply_style()
    cfg  = get_model_config(model_name)
    snr  = cfg.default_snr
    corr = get_correlation_matrix(model_name, cfg.n_params)
    rng  = np.random.default_rng(seed=42)

    print(f"\n=== H5 — Inter-Parameter Correlations | model={model_name} ===")

    # ------------------------------------------------------------------
    # Two independent, fixed test sets: correlated and uncorrelated
    # ------------------------------------------------------------------
    params_test_corr,   _ = sample_correlated(N_TEST, cfg, corr)
    params_test_uncorr, _ = sample_correlated(N_TEST, cfg, np.eye(cfg.n_params))

    signals_test_corr     = cfg.simulate_signals(params_test_corr,   cfg.xvalues, snr)
    signals_test_uncorr   = cfg.simulate_signals(params_test_uncorr, cfg.xvalues, snr)
    signals_test_corr_arr   = signals_to_array(signals_test_corr)
    signals_test_uncorr_arr = signals_to_array(signals_test_uncorr)

    params_test_corr_norm   = normalize_params(params_test_corr,   cfg.bounds)
    params_test_uncorr_norm = normalize_params(params_test_uncorr, cfg.bounds)
    truth_norm = {'corr': params_test_corr_norm, 'uncorr': params_test_uncorr_norm}

    # ------------------------------------------------------------------
    # Fixed training data (shared across all seeds): correlated and
    # uncorrelated.
    # ------------------------------------------------------------------
    params_train_corr,   coeffs_corr   = sample_correlated(N_TRAIN, cfg, corr)
    params_train_uncorr, coeffs_uncorr = sample_correlated(N_TRAIN, cfg, np.eye(cfg.n_params))
    signals_train_corr   = cfg.simulate_signals(params_train_corr,   cfg.xvalues, snr)
    signals_train_uncorr = cfg.simulate_signals(params_train_uncorr, cfg.xvalues, snr)

    # ------------------------------------------------------------------
    # Test A NN part: 2x2 design, N_SEEDS repeats each, for each NN type
    # ------------------------------------------------------------------
    rmse_nn = {v['key']: {cond: {test: np.empty((N_SEEDS, cfg.n_params))
                                 for test in TESTS}
                          for cond in CONDITIONS}
               for v in NN_VARIANTS}
    params_nncor_on_corr_all_seeds = {v['key']: [] for v in NN_VARIANTS}

    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed + 1}/{N_SEEDS} ---")

        print("NNcor (correlated training data):")
        models_cor = train_variants(signals_train_corr, coeffs_corr,
                                    cfg.xvalues, cfg, seed=seed)
        print("NNuncor (uncorrelated training data):")
        models_uncor = train_variants(signals_train_uncorr, coeffs_uncorr,
                                      cfg.xvalues, cfg, seed=seed)

        models = {'cor': models_cor, 'uncor': models_uncor}
        test_signals = {'corr': signals_test_corr, 'uncorr': signals_test_uncorr}
        for cond in CONDITIONS:
            for test in TESTS:
                preds = predict_variants(models[cond], test_signals[test],
                                         cfg.xvalues, cfg)
                for vk, p in preds.items():
                    rmse_nn[vk][cond][test][seed] = rmse_normalized(
                        p, truth_norm[test], cfg.bounds)
                    if cond == 'cor' and test == 'corr':
                        params_nncor_on_corr_all_seeds[vk].append(p)

    print(f"\n=== Test A RMSE — mean [min, max] over {N_SEEDS} seeds ===")
    for v in NN_VARIANTS:
        print(f"  --- {v['label']} ---")
        for k, pname in enumerate(cfg.param_names):
            for test, tname in (('corr', 'correlated'), ('uncorr', 'uncorrelated')):
                parts = []
                for cond in CONDITIONS:
                    a = rmse_nn[v['key']][cond][test][:, k]
                    parts.append(f"NN{cond}={a.mean():.4f} [{a.min():.4f},{a.max():.4f}]")
                print(f"  {pname}: on {tname} data -> " + "  ".join(parts))

    # ------------------------------------------------------------------
    # Test B (per NN type). The reference is the correlation actually present
    # in the correlated training data (the copula's nominal ``corr`` is the
    # correlation in Gaussian space, slightly larger than the Pearson
    # correlation of the sampled parameters).
    # ------------------------------------------------------------------
    corr_train = np.corrcoef(params_train_corr.T)
    params_nncor_on_corr_pooled = {
        k: np.concatenate(p, axis=0)
        for k, p in params_nncor_on_corr_all_seeds.items()}
    output_corrmat, pct_learned, pct_learned_clipped = {}, {}, {}
    for k, pooled in params_nncor_on_corr_pooled.items():
        output_corrmat[k] = np.corrcoef(pooled.T)
        with np.errstate(divide='ignore', invalid='ignore'):
            # only pairs that are correlated by design (nominal != 0); the
            # realized value of the others is just sampling noise
            pct_learned[k] = np.where(
                np.abs(corr) > 1e-6,
                (output_corrmat[k] / corr_train) * 100.0,
                np.nan,
            )
        pct_learned_clipped[k] = np.clip(pct_learned[k], -100, 200)

    # ------------------------------------------------------------------
    # msLSQ on both test sets
    # ------------------------------------------------------------------
    print(f"\nRunning multi-start LSQ ({MAX_RESTARTS} restarts/sample) on "
          f"both test sets...")
    params_lsq_corr = np.array([
        multi_start_lsq(signals_test_corr_arr[i], cfg.xvalues, cfg.bounds,
                        MAX_RESTARTS, cfg, rng)
        for i in range(N_TEST)
    ])
    params_lsq_uncorr = np.array([
        multi_start_lsq(signals_test_uncorr_arr[i], cfg.xvalues, cfg.bounds,
                        MAX_RESTARTS, cfg, rng)
        for i in range(N_TEST)
    ])
    rmse_lsq_on_corr   = rmse_normalized(params_lsq_corr,   params_test_corr_norm,   cfg.bounds)
    rmse_lsq_on_uncorr = rmse_normalized(params_lsq_uncorr, params_test_uncorr_norm, cfg.bounds)

    print("\nmsLSQ reference:")
    for k, pname in enumerate(cfg.param_names):
        print(f"  {pname}: RMSE on correlated data={rmse_lsq_on_corr[k]:.4f}, "
              f"on uncorrelated data={rmse_lsq_on_uncorr[k]:.4f}")

    # ------------------------------------------------------------------
    # Plots (style from plot_style.py, derived from sequence_boxplot_paper)
    # ------------------------------------------------------------------
    plot_idx = list(range(min(3, cfg.n_params)))

    # --- Save the raw numbers behind the figure (before plotting) ---
    fig_file = f'{outdir(model_name)}/H5_inter_parameter_correlations.png'
    fig_data = dict(
        corr=corr_train, corr_nominal=corr, output_corrmat=output_corrmat,
        pct_learned=pct_learned, pct_learned_clipped=pct_learned_clipped,
        plot_names=[cfg.param_names[k] for k in plot_idx],
        rmse_nn=rmse_nn,
        rmse_lsq_on_corr=rmse_lsq_on_corr, rmse_lsq_on_uncorr=rmse_lsq_on_uncorr,
        params_nncor_on_corr_pooled=params_nncor_on_corr_pooled,
        params_test_corr=params_test_corr, params_test_uncorr=params_test_uncorr)
    save_figure_data(fig_file, cfg, model_name, snr, **fig_data)

    plot_figure(cfg, fig_file, plot_idx, **fig_data)

    return {
        'A': dict(rmse_nn=rmse_nn,
                  rmse_lsq_on_corr=rmse_lsq_on_corr,
                  rmse_lsq_on_uncorr=rmse_lsq_on_uncorr),
        'B': dict(pct_learned=pct_learned),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IVIM', choices=['IVIM', 'DCE'])
    parser.add_argument('--replot', action='store_true',
                        help='Re-draw the figure from the saved data files '
                             '(no computation).')
    args = parser.parse_args()
    if args.replot:
        replot(args.model)
    else:
        run_experiment(args.model)