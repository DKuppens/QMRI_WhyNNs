import argparse
import numpy as np
import matplotlib.pyplot as plt

from backend.QMRI.model_registry import get_model_config
from backend.NN.nn_variants import NN_VARIANTS, train_variants, predict_variants
from backend.experiment_utils import (
    normalize_params, rmse_normalized, get_prior_mean, signals_to_array,
)
from backend.plot_style import (
    apply_style, COLORS, PANEL, param_label, line_kw, sci_y, categorical_x,
    outdir, save_fig,
)
from backend import figure_data as fd

N_TRAIN = 100000
N_TEST  = 1000
MAX_RESTARTS = 20
SNR_LEVELS = [5, 10, 20, 50]


def compute_residual(params, signal, xvalues, cfg):
    import torch
    signal_pred = cfg.parameters_to_signal(
        *[torch.tensor(float(p)) for p in params],
        torch.tensor(xvalues, dtype=torch.float64),
    )
    return float(np.sum((signal - signal_pred.numpy()) ** 2))


def multi_start_lsq(signal, xvalues, bounds, n_restarts, cfg, rng):
    """
    Multi-start LSQ (as in H1): run n_restarts random-initialization local
    fits and return the parameters with the lowest data residual. This
    gives a much more reliable estimate of the LSQ global optimum than a
    single-start (or single-refinement) fit, especially for models like DCE
    whose loss landscape is prone to local minima.
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


def save_figure_data(fig_file, cfg, model_name, snr_axis, results, params_test):
    """
    Store the numbers behind the H2 figure under ``<dir>/data/<figure name>/``:
      rmse_by_snr.csv      top row  (parameter, snr, rmse_lsq, rmse_nn)
      residual_by_snr.csv  bottom-left panel (snr, mean residual LSQ / NN)
      arrays.npz           per-sample estimates and residuals (raw material)
    """
    d = fd.data_dir_for(fig_file)
    names = list(cfg.param_names)

    cols = dict(parameter=[], snr=[], rmse_lsq=[])
    cols.update({f"rmse_nn_{v['key']}": [] for v in NN_VARIANTS})
    for k, pname in enumerate(names):
        for s in snr_axis:
            cols['parameter'].append(pname); cols['snr'].append(s)
            cols['rmse_lsq'].append(results[s]['rmse_lsq'][k])
            for v in NN_VARIANTS:
                cols[f"rmse_nn_{v['key']}"].append(results[s]['rmse_nn'][v['key']][k])
    fd.save_table(d, 'rmse_by_snr', cols)

    res_cols = dict(
        snr=list(snr_axis),
        mean_residual_lsq=[np.mean(results[s]['residual_lsq']) for s in snr_axis])
    for v in NN_VARIANTS:
        res_cols[f"mean_residual_nn_{v['key']}"] = [
            np.mean(results[s]['residual_nn'][v['key']]) for s in snr_axis]
    fd.save_table(d, 'residual_by_snr', res_cols)

    arrays = dict(params_test=np.asarray(params_test))
    for s in snr_axis:
        arrays[f'params_lsq_snr{s}'] = np.asarray(results[s]['params_lsq'])
        arrays[f'residual_lsq_snr{s}'] = np.asarray(results[s]['residual_lsq'])
        for v in NN_VARIANTS:
            k_ = v['key']
            arrays[f'params_nn_{k_}_snr{s}'] = np.asarray(results[s]['params_nn'][k_])
            arrays[f'residual_nn_{k_}_snr{s}'] = np.asarray(results[s]['residual_nn'][k_])
    fd.save_arrays(d, 'arrays', **arrays)

    fd.save_meta(d, model=model_name, param_names=names, snr_levels=list(snr_axis),
                 n_train=N_TRAIN, n_test=N_TEST, max_restarts=MAX_RESTARTS, seed=42,
                 nn_variants=[v['key'] for v in NN_VARIANTS],
                 ylabel_top='RMSE (normalized [0-1])',
                 ylabel_bottom='cumulative residual (a.u.)')
    print(f"Figure data saved under {d}")


def plot_figure(cfg, snr_axis, results, fig_file):
    """Draw the H2 figure from computed (or reloaded) results. Edit the
    styling here, then re-draw with ``--replot``."""
    snr_axis_keys = list(snr_axis)
    n_params = cfg.n_params
    # Top row shares one y axis (sharey='row'); y label and legend only on
    # the leftmost panel.
    fig, axes = plt.subplots(2, n_params, figsize=(PANEL * n_params, 2 * PANEL),
                             squeeze=False, sharey='row')

    for k, pname in enumerate(cfg.param_names):
        ax = axes[0, k]
        x = categorical_x(ax, snr_axis)
        ax.plot(x, [results[s]['rmse_lsq'][k] for s in snr_axis_keys],
                label='msLSQ', **line_kw(COLORS['msLSQ'], 'o', '-'))
        for v in NN_VARIANTS:
            ax.plot(x, [results[s]['rmse_nn'][v['key']][k] for s in snr_axis_keys],
                    label=v['label'], **line_kw(v['color'], v['marker'], '-'))
        ax.set_title(param_label(pname))
        ax.set_xlabel('SNR')
        if k == 0:
            ax.set_ylabel('RMSE (normalized [0-1])')
            ax.legend(loc='upper right')

    ax = axes[1, 0]
    x = categorical_x(ax, snr_axis)
    ax.plot(x, [np.mean(results[s]['residual_lsq']) for s in snr_axis_keys],
            label='msLSQ', **line_kw(COLORS['msLSQ'], 'o', '-'))
    for v in NN_VARIANTS:
        ax.plot(x, [np.mean(results[s]['residual_nn'][v['key']]) for s in snr_axis_keys],
                label=v['label'], **line_kw(v['color'], v['marker'], '-'))
    sci_y(ax)
    ax.set_xlabel('SNR')
    ax.set_ylabel('cumulative residual (a.u.)')
    ax.set_title('data consistency')
    ax.legend(loc='upper right')
    for ax in axes[1, 1:]:
        ax.axis('off')

    save_fig(fig, fig_file)
    plt.show()


def replot(model_name: str = 'IVIM'):
    """Re-draw the figure from the saved data files (no computation)."""
    apply_style()
    cfg = get_model_config(model_name)
    fig_file = f'{outdir(model_name)}/H2_noise_dominated_minimum.png'
    d = fd.data_dir_for(fig_file)
    meta = fd.load_meta(d)
    snr_axis = list(meta['snr_levels'])
    names = list(cfg.param_names)
    t = fd.load_table(d, 'rmse_by_snr')
    arrays = fd.load_arrays(d)
    results = {}
    for s in snr_axis:
        def col(key):
            return np.array([t[key][(t['parameter'] == nm) & (t['snr'] == s)][0]
                             for nm in names])
        r = dict(rmse_lsq=col('rmse_lsq'),
                 rmse_nn={v['key']: col(f"rmse_nn_{v['key']}") for v in NN_VARIANTS},
                 residual_lsq=arrays[f'residual_lsq_snr{s}'],
                 residual_nn={v['key']: arrays[f"residual_nn_{v['key']}_snr{s}"]
                              for v in NN_VARIANTS})
        results[s] = r
    plot_figure(cfg, snr_axis, results, fig_file)


def run_experiment(model_name: str = 'IVIM'):
    apply_style()
    cfg = get_model_config(model_name)
    rng = np.random.default_rng(seed=42)

    print(f"\n=== H2 — Noise-Dominated Minimum | model={model_name} ===")

    params_train, coeffs_train = cfg.sample_parameters(N_TRAIN, cfg.param_distribution)
    params_test, _ = cfg.sample_parameters(N_TEST, cfg.param_distribution)

    # Use string labels as dict keys to avoid any float/int key mismatch
    results = {}

    for snr in SNR_LEVELS:
        snr_key   = 'inf' if snr == np.inf else snr
        snr_label = f"SNR={snr_key}"
        print(f"\n{snr_label}...")

        # Train the two NNs (supervised / self-supervised) at the same SNR
        # as evaluation
        signals_train = cfg.simulate_signals(params_train, cfg.xvalues, snr)
        print("  Training NNs...")
        models = train_variants(signals_train, coeffs_train, cfg.xvalues, cfg)

        signals_test     = cfg.simulate_signals(params_test, cfg.xvalues, snr)
        signals_test_arr = signals_to_array(signals_test)
        params_test_norm = normalize_params(params_test, cfg.bounds)

        print("  Multi-start LSQ...")
        params_lsq = np.array([
            multi_start_lsq(signals_test_arr[i], cfg.xvalues, cfg.bounds,
                             MAX_RESTARTS, cfg, rng)
            for i in range(N_TEST)
        ])

        params_nn = predict_variants(models, signals_test, cfg.xvalues, cfg)

        rmse_lsq = rmse_normalized(params_lsq, params_test_norm, cfg.bounds)
        rmse_nn  = {k: rmse_normalized(p, params_test_norm, cfg.bounds)
                    for k, p in params_nn.items()}

        residuals_lsq = np.array([
            compute_residual(params_lsq[i], signals_test_arr[i], cfg.xvalues, cfg)
            for i in range(N_TEST)
        ])
        residuals_nn = {
            k: np.array([compute_residual(p[i], signals_test_arr[i], cfg.xvalues, cfg)
                         for i in range(N_TEST)])
            for k, p in params_nn.items()
        }

        results[snr_key] = dict(
            rmse_lsq=rmse_lsq, rmse_nn=rmse_nn,
            residual_lsq=residuals_lsq, residual_nn=residuals_nn,
            params_lsq=params_lsq, params_nn=params_nn,
        )
        print(f"  RMSE LSQ:  {rmse_lsq}")
        for v in NN_VARIANTS:
            print(f"  RMSE {v['label']}: {rmse_nn[v['key']]}")
        print(f"  Residual LSQ: {np.mean(residuals_lsq):.4f}  " +
              "  ".join(f"{v['label']}: {np.mean(residuals_nn[v['key']]):.4f}"
                        for v in NN_VARIANTS))

    # Build axis using the same keys (excluding inf)
    snr_axis      = [s for s in SNR_LEVELS if s != np.inf]
    snr_axis_keys = [s for s in snr_axis]   # ints, matching keys in results

    fig_file = f'{outdir(model_name)}/H2_noise_dominated_minimum.png'
    save_figure_data(fig_file, cfg, model_name, snr_axis, results, params_test)

    plot_figure(cfg, snr_axis, results, fig_file)

    return results


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