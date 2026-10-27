import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# mesmo layout de model_fitting.plot_fitting_mle_sessions_per_rat
sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.set_context("paper")
COLOR_NO, COLOR_YES = '#5B9E95', '#C5AA97'

data_path = os.path.realpath('../../data/processed/')+'/'
df = pd.read_csv(data_path + 'BG3_tentative.csv')

out = os.path.realpath('../../output/tmp/BG3')
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):  # remove os gráficos individuais anteriores
    if f.endswith('.png'):
        os.remove(os.path.join(out, f))

for rat, data in df.groupby('rat'):
    group = data['group'].iloc[0]
    sessions = sorted(data['session'].unique())
    n_cols = 3
    n_rows = math.ceil(len(sessions) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for ax, ses in zip(axes, sessions):
        d = data.query('session == @ses')
        d = d[d.duration > 0]
        hit = d['reinforced'] == 1
        ax.scatter(d.duration[~hit], d.trial[~hit], s=14, color=COLOR_NO, alpha=0.7, label='Trial')
        ax.scatter(d.duration[hit], d.trial[hit], s=14, color=COLOR_YES, alpha=0.9, label='Reinforced')
        ax.axvline(d.criterion.max(), color='grey', linestyle='--', lw=1.5, label='Criterion')
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Trial')
        ax.set_title(f'Session {ses}')
        ax.legend(loc='upper left')

    for ax in axes[len(sessions):]:
        fig.delaxes(ax)

    fig.suptitle(f'BG3 Trials - Rat {rat} (Group {group})', fontsize=16)
    fig.savefig(f'{out}/BG3_Trials_Rat{rat}_all_sessions.png', dpi=150)
    plt.close(fig)
