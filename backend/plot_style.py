"""
plot_style.py -- shared plotting style for the H1 / H2 / H5 / H6 scripts
========================================================================
Everything here is derived from ``sequence_boxplot_paper`` so that all
hypothesis figures look like they belong to the same paper:

  * seaborn "paper" context, all font sizes = 20
  * seaborn 'pastel' palette; LSQ uses exactly the colour it has in
    ``sequence_boxplot_paper`` (pastel index 1, the pastel orange)
  * ~6 x 6 inch panels, ``fig.tight_layout()``, saved at dpi=300
  * LaTeX-style parameter titles ($D$, $D^*$, $f$)
  * labels with units / "a.u.", scientific notation on y-axes where the
    values are small (residuals)
  * no figure-level ``suptitle`` (paper figures carry their message in the
    caption and the panel titles)

Colour assignment (pastel palette indices in brackets)
------------------------------------------------------
    msLSQ                 pastel[1]  orange   <- as in sequence_boxplot_paper
    NN / nncor          pastel[0]  blue
    nnuncor             pastel[9]  cyan
    MAP-LSQ (cov)       pastel[2]  green
    MAP-LSQ (identity)  pastel[4]  purple
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

OUTPUT_ROOT = '/net/beegfs/projects/rng-topqmri/topqmr/NCDE-papers/Paper4'

FONT_SIZE    = 20
LEGEND_SMALL = 12      # for dense panels with many legend entries
TICK_SMALL   = 14      # for crowded categorical tick labels
PANEL        = 6       # inches per panel (sequence_boxplot_paper: 18 / 3)
DPI          = 300

_PASTEL = sns.color_palette('pastel', 10)

COLORS = dict(
    msLSQ     = _PASTEL[1],   # identical to LSQ in sequence_boxplot_paper
    NN      = _PASTEL[0],
    NN_UNCOR= _PASTEL[9],
    MAP_COV = _PASTEL[2],
    MAP_ISO = _PASTEL[4],
    GT      = '#FFD700',    # ground truth reference (gold) — a role, not
                             # an estimator, so deliberately outside the
                             # LSQ/NN pastel family used by the anecdotal
                             # loss-landscape corner plots
)

_PARAM_LABELS = {
    'd':       r'$D$',
    'dp':      r'$D^*$',
    'd*':      r'$D^*$',
    'f':       r'$f$',
    'ke':      r'$k_e$',
    'ktrans':  r'$K^{trans}$',
    've':      r'$v_e$',
    'vp':      r'$v_p$',
    'dt':      r'$dt$',
    't0':      r'$t_0$',
    's0':      r'$S_0$',
}

# Physical unit per parameter, for axis labels that show a raw (not
# bounds-normalized) parameter value. An empty string means the parameter
# is genuinely dimensionless (f, v_e and v_p are volume/signal fractions
# here, not just "some unspecified unit") -- with_unit() below shows
# nothing at all for these, rather than a placeholder like "a.u.", since
# there is no unit to name. Anything not listed here (Ktrans, dt, t0, S0,
# ...) falls back to 'a.u.' in param_unit(), since this codebase's model
# config doesn't expose a physical unit string to read the real one from,
# and guessing a specific unit would be worse than flagging it as
# unspecified.
_PARAM_UNITS = {
    'd':   'mm$^2$/s',
    'dp':  'mm$^2$/s',
    'd*':  'mm$^2$/s',
    'f':   '',
    've':  '',
    'vp':  '',
    'ke':  'min$^{-1}$',
}

MODEL_TITLES = {'IVIM': 'IVIM MRI', 'VFA': 'VFA T1-mapping', 'DCE': 'DCE-MRI'}


def apply_style():
    """Same seaborn setup as sequence_boxplot_paper."""
    sns.set_context("paper", rc={
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
        'ytick.labelsize': FONT_SIZE, 'legend.fontsize': FONT_SIZE,
    })
    sns.set_palette('pastel')


def param_label(pname: str) -> str:
    """LaTeX-style parameter title; falls back to the raw name. Matching
    is case-insensitive: model configs in this codebase aren't consistent
    about whether a parameter is named e.g. 'S0', 's0', or 'Ke', 'ke', but
    it's always the same physical parameter regardless of source casing."""
    return _PARAM_LABELS.get(pname.lower(), pname)


def param_label_bold(pname: str) -> str:
    """Bold variant of param_label(), for a row/column label that needs
    emphasis (e.g. the leftmost column of a summary grid) without also
    bolding an accompanying unit string on the same Axes label. Only the
    math symbol itself is bolded, via LaTeX's \\mathbf{} *inside* the
    existing $...$ math mode, so the Text object's own fontweight can stay
    'normal' and any plain (non-math) text appended to the same label --
    such as a unit in parentheses -- is unaffected."""
    latex = param_label(pname)
    if latex.startswith('$') and latex.endswith('$'):
        inner = latex[1:-1]
        return f'$\\mathbf{{{inner}}}$'
    return latex  # fallback: raw name, not in math mode, nothing to bold safely


def param_unit(pname: str) -> str:
    """Physical unit for a parameter (e.g. 'mm$^2$/s' for D), matched
    case-insensitively like param_label(). Returns '' for parameters that
    are genuinely dimensionless (f, v_e, v_p, k_e), and falls back to
    'a.u.' for any parameter without a known physical unit in this
    codebase (there is a real unit, it's just not available here, unlike
    the dimensionless case)."""
    return _PARAM_UNITS.get(pname.lower(), 'a.u.')


def with_unit(label: str, pname: str) -> str:
    """Append a parenthetical unit to `label` via param_unit(pname), or
    return `label` unchanged if the parameter is dimensionless -- there is
    nothing to put in the parentheses, so no empty '()' is shown."""
    unit = param_unit(pname)
    return f'{label} ({unit})' if unit else label


def line_kw(color, marker='o', linestyle='-'):
    """Common line/marker look: pastel fill, black marker edge (like the
    black-edged mean markers in sequence_boxplot_paper)."""
    return dict(color=color, marker=marker, linestyle=linestyle,
                linewidth=2.5, markersize=9,
                markeredgecolor='black', markeredgewidth=1.0)


def sci_y(ax):
    """Scientific notation on a linear y-axis, as in sequence_boxplot_paper."""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(FONT_SIZE)


def plain_log_x_ticks(ax, values):
    """Show plain numbers (1, 2, 5, ...) instead of 10^x on a log x-axis."""
    ax.set_xticks(list(values))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()


def categorical_x(ax, labels, fontsize=None):
    """Evenly spaced categorical x positions (like the SNR boxplots)."""
    pos = list(range(len(labels)))
    ax.set_xticks(pos)
    ax.set_xticklabels([str(l) for l in labels],
                       **({'fontsize': fontsize} if fontsize else {}))
    return pos


def heatmap_cmap():
    """Sequential map built from the NN colour (replaces RdYlGn)."""
    return sns.light_palette(COLORS['NN'], as_cmap=True)


def snr_colormap(snr_sorted):
    """
    Map each SNR level to a colour from a plasma-based sequential scale
    (light = high SNR, dark = low SNR). SNR is a nuisance/noise axis, not
    an estimator, so it deliberately sits outside the LSQ/NN pastel
    family used everywhere else in these figures.
    """
    cmap = plt.get_cmap('plasma')
    n = max(len(snr_sorted) - 1, 1)
    return {snr: cmap(k / n) for k, snr in enumerate(snr_sorted)}


def outdir(model_name: str) -> str:
    d = f'{OUTPUT_ROOT}/{model_name}'
    os.makedirs(d, exist_ok=True)
    return d


def save_fig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches='tight')