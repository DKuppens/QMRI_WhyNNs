import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from backend.QMRI.model_registry import get_model_config
from backend.experiment_utils import (
    normalize_params, rmse_normalized, signals_to_array,
)
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.plot_style import (
    apply_style, COLORS, PANEL, MODEL_TITLES, param_label, line_kw, sci_y,
    plain_log_x_ticks, outdir, save_fig,
)
from backend import figure_data as fd

# =============================================================================
# DEFAULTS (overridden by model config)
# =============================================================================

N_TRAIN    = 100000
N_TEST     = 1000
N_RESTARTS = [1, 2, 5, 10, 25, 50]   # numbers of initializations n
N_REPS     = 20                      # independent runs per n

X_LABEL = 'number of initializations'


# =============================================================================
# LSQ WITH n INITIALIZATIONS
# =============================================================================

def residual(params, signal, xvalues, cfg):
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def best_of_n_initializations(signal, xvalues, n, cfg, rng, lo, hi):
    """
    One run of n-initialization LSQ: fit from n independent uniform random
    initializations and return the parameters (and data residual) of the fit
    with the lowest data residual.
    """
    best_params, best_res = None, np.inf
    for _ in range(n):
        init = rng.uniform(lo, hi)
        p = np.asarray(cfg.fit_lsq_1d(xvalues, signal, init))
        r = residual(p, signal, xvalues, cfg)
        if r < best_res:
            best_res, best_params = r, p.copy()
    return best_params, best_res


# =============================================================================
# MAIN
# =============================================================================

def save_figure_data(out, model_name, cfg, snr, rmse_runs, res_runs, rmse_nn,
                     best_params, best_residuals, params_test_norm):
    """
    Store the numbers behind both H1 figures (and the raw per-run estimates)
    under ``<out>/data/<figure name>/`` so the figures can be re-drawn without
    re-running anything. See backend/figure_data.py for the layout.
    """
    names = list(cfg.param_names)
    common = dict(model=model_name, snr=snr, param_names=names,
                  n_restarts=N_RESTARTS, n_reps=N_REPS, n_train=N_TRAIN,
                  n_test=N_TEST, seed=42,
                  nn_variants=[v['key'] for v in NN_VARIANTS],
                  line_definition='mean over the N_REPS independent runs',
                  band_definition='min-max over the N_REPS independent runs')

    # ---- Figure 1: RMSE vs. number of initializations ----
    d = fd.data_dir_for(f'{out}/H1_local_minima_rmse.png')
    summ = {k: [] for k in ('parameter', 'n_restarts', 'line_mean',
                            'band_min', 'band_max')}
    summ.update({f"rmse_nn_{v['key']}": [] for v in NN_VARIANTS})
    reps = {k: [] for k in ('parameter', 'n_restarts', 'rep', 'rmse')}
    for k, pname in enumerate(names):
        for n in N_RESTARTS:
            r = rmse_runs[n][:, k]
            row = [('parameter', pname), ('n_restarts', n),
                   ('line_mean', r.mean()), ('band_min', r.min()),
                   ('band_max', r.max())]
            row += [(f"rmse_nn_{v['key']}", rmse_nn[v['key']][k])
                    for v in NN_VARIANTS]
            for key, val in row:
                summ[key].append(val)
            for rep, val in enumerate(r):
                reps['parameter'].append(pname); reps['n_restarts'].append(n)
                reps['rep'].append(rep); reps['rmse'].append(val)
    fd.save_table(d, 'rmse_summary', summ)      # what the figure draws
    fd.save_table(d, 'rmse_all_reps', reps)     # every run behind the band
    fd.save_meta(d, ylabel='RMSE (normalized [0-1])', **common)

    # ---- Figure 2: data residual vs. number of initializations ----
    d = fd.data_dir_for(f'{out}/H1_local_minima_residual.png')
    summ = {k: [] for k in ('n_restarts', 'line_mean', 'band_min', 'band_max')}
    reps = {k: [] for k in ('n_restarts', 'rep', 'mean_residual')}
    for n in N_RESTARTS:
        r = res_runs[n]
        for key, val in (('n_restarts', n), ('line_mean', r.mean()),
                         ('band_min', r.min()), ('band_max', r.max())):
            summ[key].append(val)
        for rep, val in enumerate(r):
            reps['n_restarts'].append(n); reps['rep'].append(rep)
            reps['mean_residual'].append(val)
    fd.save_table(d, 'residual_summary', summ)
    fd.save_table(d, 'residual_all_reps', reps)
    fd.save_meta(d, title=MODEL_TITLES.get(model_name, model_name),
                 ylabel='cumulative data residual (a.u.)', **common)

    # ---- Raw material: the selected estimate of every run ----
    d = fd.data_dir_for(f'{out}/H1_local_minima_raw.png')
    arrays = dict(params_test_norm=np.asarray(params_test_norm))
    for n in N_RESTARTS:
        arrays[f'best_params_n{n}'] = best_params[n]        # (N_REPS, N_TEST, n_params)
        arrays[f'best_residuals_n{n}'] = best_residuals[n]  # (N_REPS, N_TEST)
    for v in NN_VARIANTS:
        arrays[f"rmse_nn_{v['key']}"] = np.asarray(rmse_nn[v['key']])
    fd.save_arrays(d, 'arrays', **arrays)
    fd.save_meta(d, **common)
    print(f"Figure data saved under {out}/data/")


def plot_figures(cfg, model_name, out, rmse_runs, res_runs, rmse_nn):
    """Draw both H1 figures from computed (or reloaded) numbers. Edit the
    styling here, then re-draw with ``--replot``.

    rmse_runs[n] : (N_REPS, n_params) normalized RMSE of every run
    res_runs[n]  : (N_REPS,)          mean data residual of every run
    rmse_nn      : {variant key: (n_params,)}
    """
    n_params = cfg.n_params

    # Figure 1: RMSE vs. number of initializations, one panel per parameter.
    # Line = mean RMSE over the N_REPS runs; band = min-max across runs.
    # One shared y axis; y label and legend on the leftmost panel only.
    fig1, axes1 = plt.subplots(1, n_params, figsize=(PANEL * n_params, PANEL),
                               squeeze=False, sharey=True)
    axes1 = axes1[0]

    for k, (ax, pname) in enumerate(zip(axes1, cfg.param_names)):
        line = np.array([rmse_runs[n][:, k].mean() for n in N_RESTARTS])
        lo   = np.array([rmse_runs[n][:, k].min()  for n in N_RESTARTS])
        hi   = np.array([rmse_runs[n][:, k].max()  for n in N_RESTARTS])

        ax.semilogx(N_RESTARTS, line, label='msLSQ',
                    **line_kw(COLORS['msLSQ'], 'o', '-'))
        ax.fill_between(N_RESTARTS, lo, hi, alpha=0.4, color=COLORS['msLSQ'],
                        linewidth=0)
        for v in NN_VARIANTS:
            ax.axhline(rmse_nn[v['key']][k], color=v['color'], linestyle='--',
                       linewidth=3, label=v['label'])
        plain_log_x_ticks(ax, N_RESTARTS)
        ax.set_xlabel(X_LABEL)
        ax.set_title(param_label(pname))
        if k == 0:
            ax.set_ylabel('RMSE (normalized [0-1])')
            ax.legend(loc='best')

    save_fig(fig1, f'{out}/H1_local_minima_rmse.png')

    # Figure 2: data residual vs. number of initializations. Line = mean over
    # runs of the mean data residual of the lowest-residual estimates (the
    # same estimates as in Figure 1); band = min-max across runs.
    fig2, ax2 = plt.subplots(figsize=(PANEL * 1.2, PANEL))
    line_res = np.array([res_runs[n].mean() for n in N_RESTARTS])
    lo_res   = np.array([res_runs[n].min()  for n in N_RESTARTS])
    hi_res   = np.array([res_runs[n].max()  for n in N_RESTARTS])
    print(f"\n=== Residuals — mean over runs [min, max over runs] ===")
    for n, m, l, h in zip(N_RESTARTS, line_res, lo_res, hi_res):
        print(f"{n:<12} {m:.4f} [{l:.4f},{h:.4f}]")
    ax2.semilogx(N_RESTARTS, line_res, label='msLSQ',
                 **line_kw(COLORS['msLSQ'], 'o', '-'))
    ax2.fill_between(N_RESTARTS, lo_res, hi_res, alpha=0.4,
                     color=COLORS['msLSQ'], linewidth=0)
    plain_log_x_ticks(ax2, N_RESTARTS)
    sci_y(ax2)
    ax2.set_xlabel(X_LABEL)
    ax2.set_ylabel('cumulative data residual (a.u.)')
    ax2.set_title(MODEL_TITLES.get(model_name, model_name))
    ax2.legend(loc='upper right')

    save_fig(fig2, f'{out}/H1_local_minima_residual.png')
    plt.show()


def load_figure_data(model_name, cfg):
    """Rebuild the numbers plot_figures() needs from the saved data files."""
    global N_RESTARTS
    out = outdir(model_name)
    d1 = fd.data_dir_for(f'{out}/H1_local_minima_rmse.png')
    d2 = fd.data_dir_for(f'{out}/H1_local_minima_residual.png')
    meta = fd.load_meta(d1)
    N_RESTARTS = [int(n) for n in meta['n_restarts']]
    names = list(cfg.param_names)

    summ = fd.load_table(d1, 'rmse_summary')
    reps = fd.load_table(d1, 'rmse_all_reps')
    n_reps = int(reps['rep'].max()) + 1
    rmse_runs = {}
    for n in N_RESTARTS:
        arr = np.empty((n_reps, len(names)))
        for k, p in enumerate(names):
            m = (reps['parameter'] == p) & (reps['n_restarts'] == n)
            arr[reps['rep'][m], k] = reps['rmse'][m]
        rmse_runs[n] = arr
    rmse_nn = {v['key']: np.array([summ[f"rmse_nn_{v['key']}"][summ['parameter'] == p][0]
                                   for p in names])
               for v in NN_VARIANTS}

    rreps = fd.load_table(d2, 'residual_all_reps')
    res_runs = {}
    for n in N_RESTARTS:
        m = rreps['n_restarts'] == n
        arr = np.empty(n_reps)
        arr[rreps['rep'][m]] = rreps['mean_residual'][m]
        res_runs[n] = arr
    return rmse_runs, res_runs, rmse_nn


def replot(model_name: str = 'DCE'):
    """Re-draw both figures from the saved data (no computation)."""
    apply_style()
    cfg = get_model_config(model_name)
    out = outdir(model_name)
    data = load_figure_data(model_name, cfg)
    plot_figures(cfg, model_name, out, *data)


def run_experiment(model_name: str = 'DCE'):
    apply_style()
    cfg = get_model_config(model_name)
    rng = np.random.default_rng(seed=42)
    snr = cfg.default_snr

    print(f"\n=== H1 — Local Minima | model={model_name} ===")

    # --- Simulate data ---
    params_train, coeffs_train = cfg.sample_parameters(N_TRAIN, cfg.param_distribution)
    signals_train = cfg.simulate_signals(params_train, cfg.xvalues, snr)

    params_test, _ = cfg.sample_parameters(N_TEST, cfg.param_distribution)
    signals_test   = cfg.simulate_signals(params_test, cfg.xvalues, snr)
    signals_test_arr = signals_to_array(signals_test)

    params_test_norm = normalize_params(params_test, cfg.bounds)

    # --- Train the two NNs (supervised / self-supervised) ---
    print("Training NNs...")
    models    = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg)
    params_nn = predict_variants(models, signals_test, cfg.xvalues, cfg)
    rmse_nn   = {k: rmse_normalized(p, params_test_norm, cfg.bounds)
                 for k, p in params_nn.items()}

    # --- LSQ: for every n, N_REPS independent runs of n initializations ---
    lo = np.array([b[0] for b in cfg.bounds])
    hi = np.array([b[1] for b in cfg.bounds])
    rmse_runs      = {n: [] for n in N_RESTARTS}
    res_runs       = {n: [] for n in N_RESTARTS}
    best_params    = {n: np.empty((N_REPS, N_TEST, cfg.n_params)) for n in N_RESTARTS}
    best_residuals = {n: np.empty((N_REPS, N_TEST)) for n in N_RESTARTS}

    print(f"Running LSQ: {len(N_RESTARTS)} values of n x {N_REPS} runs x "
          f"{N_TEST} samples...")
    for n in N_RESTARTS:
        for rep in range(N_REPS):
            print(f"  n={n}: run {rep + 1}/{N_REPS}")
            for i in range(N_TEST):
                bp, br = best_of_n_initializations(
                    signals_test_arr[i], cfg.xvalues, n, cfg, rng, lo, hi)
                best_params[n][rep, i]    = bp
                best_residuals[n][rep, i] = br
            rmse_runs[n].append(
                rmse_normalized(best_params[n][rep], params_test_norm, cfg.bounds))
            res_runs[n].append(best_residuals[n][rep].mean())

    for n in N_RESTARTS:
        rmse_runs[n] = np.array(rmse_runs[n])     # (N_REPS, n_params)
        res_runs[n]  = np.array(res_runs[n])      # (N_REPS,)

    # --- Print results ---
    print(f"\n=== RMSE — mean over runs (min, max over runs) ===")
    print(f"{'Inits':<12}", *[f"{p:<36}" for p in cfg.param_names])
    for n in N_RESTARTS:
        means = rmse_runs[n].mean(axis=0)
        lo_r  = rmse_runs[n].min(axis=0)
        hi_r  = rmse_runs[n].max(axis=0)
        cols  = [f"{m:.4f} [{l:.4f},{h:.4f}]" for m, l, h in zip(means, lo_r, hi_r)]
        print(f"{n:<12}", *[f"{c:<36}" for c in cols])
    for v in NN_VARIANTS:
        print(f"{v['label']:<12}", *[f"{x:<36.4f}" for x in rmse_nn[v['key']]])

    # --- Plots ---
    out = outdir(model_name)

    # --- Save the raw numbers behind the figures (before any plotting) ---
    save_figure_data(out, model_name, cfg, snr, rmse_runs, res_runs, rmse_nn,
                     best_params, best_residuals, params_test_norm)

    plot_figures(cfg, model_name, out, rmse_runs, res_runs, rmse_nn)

    return rmse_runs, res_runs, rmse_nn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='IVIM',
                        choices=['IVIM', 'DCE'],
                        help='QMRI model to use')
    parser.add_argument('--replot', action='store_true',
                        help='Re-draw the figures from the saved data files '
                             '(no computation).')
    args = parser.parse_args()
    if args.replot:
        replot(args.model)
    else:
        run_experiment(args.model)