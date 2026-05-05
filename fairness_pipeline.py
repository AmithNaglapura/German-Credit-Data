"""
Fairness-Aware Machine Learning Pipeline
=========================================
Dataset  : German Credit Data (german_credit_data.csv)
Task     : Binary credit-risk classification  (0 = good, 1 = bad)
Sensitive : Sex  (male / female)
Baseline : Logistic Regression (no fairness constraint)
Debiased : Logistic Regression + per-group decision thresholds
           (Threshold Optimisation, demographic-parity constraint)
Libraries: scikit-learn, fairlearn, pandas, matplotlib, seaborn
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             roc_auc_score)
from fairlearn.metrics import (demographic_parity_difference,
                               equalized_odds_difference,
                               demographic_parity_ratio)
from fairlearn.postprocessing import ThresholdOptimizer
warnings.filterwarnings('ignore')

# ── paths ────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE, 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── palette & plot style ─────────────────────────────────────────────────────
P = dict(
    baseline='#4361EE', debiased='#F72585',
    male='#4CC9F0',     female='#FF9F1C',
    bg='#0D1117',       panel='#161B22',
    text='#E6EDF3',     grid='#30363D',
    gold='#F9C74F',     green='#06D6A0',
)
plt.rcParams.update({
    'figure.facecolor': P['bg'],   'axes.facecolor':   P['panel'],
    'axes.edgecolor':   P['grid'], 'axes.labelcolor':  P['text'],
    'xtick.color':      P['text'], 'ytick.color':      P['text'],
    'text.color':       P['text'], 'grid.color':       P['grid'],
    'grid.alpha': 0.5,  'font.family': 'DejaVu Sans',
    'font.size': 11,    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'legend.facecolor': P['panel'], 'legend.edgecolor': P['grid'],
})

def save_fig(fig, name):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=P['bg'])
    plt.close(fig)
    print(f"  Saved → {path}")

def bar_labels(ax, bars, fmt='.3f', dy=0.012):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + dy,
                f'{h:{fmt}}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=P['text'])

# ═══════════════════════════════════════════════════════════════════════
# 1. LOAD & ENGINEER TARGET
# ═══════════════════════════════════════════════════════════════════════
df = pd.read_csv(os.path.join(BASE, 'german_credit_data.csv'), index_col=0)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

# ordinal maps for financial buffers
sav_map = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}
chk_map = {'little': 0, 'moderate': 1, 'rich': 2}
df['sav'] = df['Saving accounts'].map(sav_map).fillna(-1)
df['chk'] = df['Checking account'].map(chk_map).fillna(-1)

# rule-based risk score (domain knowledge)
score = (
    (df['Credit amount'] > df['Credit amount'].quantile(0.60)).astype(int) * 2 +
    (df['Duration']      > df['Duration'].quantile(0.60)).astype(int) * 2 +
    (df['sav'] < 1).astype(int) +
    (df['chk'] < 1).astype(int)
)
df['Risk'] = (score >= 4).astype(int)   # 1 = bad credit risk

print(f"Overall bad rate : {df['Risk'].mean():.1%}")
print(df.groupby('Sex')['Risk'].mean().rename('bad_rate'))

# ═══════════════════════════════════════════════════════════════════════
# 2. PRE-PROCESS
# ═══════════════════════════════════════════════════════════════════════
FEATURES = ['Age','Sex','Job','Housing',
            'Saving accounts','Checking account',
            'Credit amount','Duration','Purpose']

data = df[FEATURES + ['Risk']].copy()
cat_cols = ['Sex','Housing','Saving accounts','Checking account','Purpose']
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('Unknown'))

# keep raw Sex string as sensitive feature for fairlearn
sensitive_all = df['Sex'].values            # 'male' | 'female'

X        = data[FEATURES].values
y        = data['Risk'].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

(X_tr, X_te,
 y_tr, y_te,
 s_tr, s_te) = train_test_split(
    X_scaled, y, sensitive_all,
    test_size=0.25, random_state=42, stratify=y)

print(f"\nTrain {len(X_tr)} | Test {len(X_te)}")
print(f"Groups: {np.unique(s_te)}")

# ═══════════════════════════════════════════════════════════════════════
# 3. BASELINE  — standard Logistic Regression
# ═══════════════════════════════════════════════════════════════════════
base_clf = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
base_clf.fit(X_tr, y_tr)
yp_base  = base_clf.predict(X_te)
prob_base = base_clf.predict_proba(X_te)[:, 1]

def calc_metrics(y_true, y_pred, sens):
    return dict(
        Accuracy  = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall    = recall_score(y_true, y_pred, zero_division=0),
        F1        = f1_score(y_true, y_pred, zero_division=0),
        DPD       = demographic_parity_difference(y_true, y_pred,
                        sensitive_features=sens),
        DPR       = demographic_parity_ratio(y_true, y_pred,
                        sensitive_features=sens),
        EOD       = equalized_odds_difference(y_true, y_pred,
                        sensitive_features=sens),
    )

bm = calc_metrics(y_te, yp_base, s_te)
print("\n=== BASELINE ===")
for k, v in bm.items():
    print(f"  {k}: {v:.4f}")
for g in np.unique(s_te):
    print(f"  Positive rate [{g}]: {yp_base[s_te==g].mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════
# 4. DEBIASING — Threshold Optimisation (post-processing)
#    Uses per-group decision thresholds to enforce demographic parity.
#    Inner estimator is re-fit fresh so weights are independent.
# ═══════════════════════════════════════════════════════════════════════
inner = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
inner.fit(X_tr, y_tr)

to = ThresholdOptimizer(
    estimator      = inner,
    constraints    = "demographic_parity",
    objective      = "balanced_accuracy_score",
    predict_method = "predict_proba",
    flip           = True,
)
to.fit(X_tr, y_tr, sensitive_features=s_tr)
yp_deb = to.predict(X_te, sensitive_features=s_te, random_state=42)

dm = calc_metrics(y_te, yp_deb, s_te)
print("\n=== DEBIASED (Threshold Optimisation) ===")
for k, v in dm.items():
    print(f"  {k}: {v:.4f}")
for g in np.unique(s_te):
    print(f"  Positive rate [{g}]: {yp_deb[s_te==g].mean():.3f}")

GROUPS = np.unique(s_te)
base_pr = {g: yp_base[s_te==g].mean() for g in GROUPS}
deb_pr  = {g: yp_deb [s_te==g].mean() for g in GROUPS}
BW      = 0.35

# ═══════════════════════════════════════════════════════════════════════
# 5. CHARTS
# ═══════════════════════════════════════════════════════════════════════

# ── Chart 01: Predictive performance ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
keys = ['Accuracy','Precision','Recall','F1']
x    = np.arange(len(keys))
b1 = ax.bar(x-BW/2, [bm[k] for k in keys], BW,
            label='Baseline', color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x+BW/2, [dm[k] for k in keys], BW,
            label='Debiased (Threshold Opt.)', color=P['debiased'],
            alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2))
ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=12)
ax.set_ylim(0,1.15)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Predictive Performance: Baseline vs Debiased Model\n'
             'Sensitive Attribute: Sex  |  Task: Credit Risk Classification',
             pad=12)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'01_predictive_metrics.png')

# ── Chart 02: Fairness metrics ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
fkeys  = ['DPD','EOD']
flbls  = ['Demographic\nParity Difference','Equalized\nOpportunity Difference']
bfv    = [abs(bm[k]) for k in fkeys]
dfv    = [abs(dm[k]) for k in fkeys]
x      = np.arange(len(fkeys))
b1 = ax.bar(x-BW/2, bfv, BW, label='Baseline',
            color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x+BW/2, dfv, BW, label='Debiased (Threshold Opt.)',
            color=P['debiased'], alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2), dy=0.005)
ceiling = max(max(bfv),max(dfv))*1.6+0.05
ax.set_xticks(x); ax.set_xticklabels(flbls, fontsize=12)
ax.set_ylim(0, ceiling)
ax.set_ylabel('|Metric Value|  — lower = fairer', fontsize=12)
ax.set_title('Fairness Metrics: Baseline vs Debiased Model\n'
             'Sensitive Attribute: Sex', pad=12)
ax.axhline(0.1, color=P['gold'], lw=1.8, ls='--',
           label='Fairness threshold (0.1)', zorder=4)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'02_fairness_metrics.png')

# ── Chart 03: Per-group positive rates ──────────────────────────────────
gcols = [P['female'] if g=='female' else P['male'] for g in GROUPS]
fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax, pr, title in zip(axes,
                          [base_pr, deb_pr],
                          ['Baseline Model',
                           'Debiased Model (Threshold Opt.)']):
    vals = [pr[g] for g in GROUPS]
    bars = ax.bar(GROUPS, vals, color=gcols, alpha=0.9,
                  edgecolor=P['bg'], linewidth=1.5, zorder=3)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=P['text'])
    ax.set_ylim(0,0.8)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel('Positive Prediction Rate\n(predicted bad credit)',fontsize=11)
    ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)

fig.suptitle('Per-Group Positive Prediction Rates by Sex',
             fontsize=14, fontweight='bold', y=1.03)
pm = mpatches.Patch(color=P['male'],   label='Male')
pf = mpatches.Patch(color=P['female'], label='Female')
fig.legend(handles=[pm,pf], loc='upper right', fontsize=11)
fig.tight_layout()
save_fig(fig,'03_group_positive_rates.png')

# ── Chart 04: Confusion matrices ────────────────────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,5))
for ax, yp, title in zip(axes,
                           [yp_base, yp_deb],
                           ['Baseline','Debiased (Threshold Opt.)']):
    cm = confusion_matrix(y_te, yp)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                linewidths=0.5, linecolor=P['grid'],
                annot_kws={'size':16,'weight':'bold'},
                xticklabels=['Good (0)','Bad (1)'],
                yticklabels=['Good (0)','Bad (1)'], cbar=False)
    ax.set_title(f'{title} — Confusion Matrix', pad=8)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    ax.set_facecolor(P['panel'])
fig.tight_layout()
save_fig(fig,'04_confusion_matrices.png')

# ── Chart 05: Accuracy–Fairness trade-off ──────────────────────────────
fig, ax = plt.subplots(figsize=(9,6))
pts = [
    ('Baseline', bm, P['baseline']),
    ('Debiased', dm, P['debiased']),
]
for label, m, c in pts:
    ax.scatter(abs(m['DPD']), m['Accuracy'], s=900,
               color=c, alpha=0.9, zorder=5, edgecolors='white', lw=2)
    ax.annotate(label, xy=(abs(m['DPD']), m['Accuracy']),
                xytext=(abs(m['DPD'])+0.003, m['Accuracy']+0.004),
                fontsize=13, fontweight='bold', color=c)
d_vals = [abs(bm['DPD']), abs(dm['DPD'])]
a_vals = [bm['Accuracy'],  dm['Accuracy']]
ax.annotate('',
    xy    =(min(d_vals)-0.003, max(a_vals)+0.003),
    xytext=(max(d_vals)+0.003, min(a_vals)-0.003),
    arrowprops=dict(arrowstyle='<->', color=P['gold'], lw=2))
ax.text(np.mean(d_vals), np.mean(a_vals)+0.025,
        '← Accuracy vs Fairness →',
        ha='center', fontsize=11, color=P['gold'], style='italic')
ax.set_xlabel('|Demographic Parity Difference|  (lower = fairer)', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy – Fairness Trade-off\n'
             'Each point represents one model configuration', fontsize=14, pad=12)
ax.grid(True,zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'05_accuracy_fairness_tradeoff.png')

# ── Chart 06: Executive dashboard ──────────────────────────────────────
fig = plt.figure(figsize=(18,11))
fig.suptitle('Fairness-Aware Credit Risk Classification — Executive Dashboard',
             fontsize=17, fontweight='bold', y=0.99)
gs = GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.42)

# metrics table
ax_t = fig.add_subplot(gs[0,:2]); ax_t.axis('off')
rows = [
    ['Metric','Baseline','Debiased','Δ Change','Effect'],
    ['Accuracy',  f"{bm['Accuracy']:.3f}", f"{dm['Accuracy']:.3f}",
     f"{dm['Accuracy']-bm['Accuracy']:+.3f}", 'Performance cost of fairness'],
    ['Precision', f"{bm['Precision']:.3f}",f"{dm['Precision']:.3f}",
     f"{dm['Precision']-bm['Precision']:+.3f}", 'Precision shift'],
    ['Recall',    f"{bm['Recall']:.3f}",   f"{dm['Recall']:.3f}",
     f"{dm['Recall']-bm['Recall']:+.3f}", 'Recall shift'],
    ['F1 Score',  f"{bm['F1']:.3f}",       f"{dm['F1']:.3f}",
     f"{dm['F1']-bm['F1']:+.3f}", 'Overall balance'],
    ['|DPD|',     f"{abs(bm['DPD']):.3f}", f"{abs(dm['DPD']):.3f}",
     f"{abs(dm['DPD'])-abs(bm['DPD']):+.3f}", '↓ = fairer (key gain)'],
    ['|EOD|',     f"{abs(bm['EOD']):.3f}", f"{abs(dm['EOD']):.3f}",
     f"{abs(dm['EOD'])-abs(bm['EOD']):+.3f}", '↓ = fairer'],
    ['DPR',       f"{bm['DPR']:.3f}",      f"{dm['DPR']:.3f}",
     f"{dm['DPR']-bm['DPR']:+.3f}", '→ 1.0 = perfect parity'],
]
def row_colors(r, row):
    if r == 0: return ['#1F3A5F']*5
    try:
        v    = float(row[3])
        fair = row[0] in ('|DPD|','|EOD|')
        dpr  = row[0] == 'DPR'
        good = (v < 0 if fair else (v > 0 if dpr else v >= 0))
        c3   = '#0D3B28' if good else '#3B0D0D'
    except:
        c3 = P['panel']
    return [P['panel'], P['panel'], P['panel'], c3, P['panel']]

cc  = [row_colors(r, rows[r]) for r in range(len(rows))]
tbl = ax_t.table(cellText=rows, cellLoc='center', loc='center', cellColours=cc)
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.0)
for (_, __), cell in tbl.get_celld().items():
    cell.set_edgecolor(P['grid']); cell.get_text().set_color(P['text'])
ax_t.set_title('Comprehensive Metric Comparison', fontsize=13, pad=6)

# DPR gauge
ax_dpr = fig.add_subplot(gs[0,2])
bars_h = ax_dpr.barh(['Baseline','Debiased'],
                     [bm['DPR'], dm['DPR']],
                     color=[P['baseline'],P['debiased']],
                     alpha=0.9, height=0.45, zorder=3)
ax_dpr.axvline(1.0, color=P['green'], lw=2.2, ls='--', label='Perfect (1.0)')
ax_dpr.axvline(0.8, color=P['gold'],  lw=1.5, ls=':',  label='80% rule (0.8)')
for b in bars_h:
    v = b.get_width()
    ax_dpr.text(v+0.01, b.get_y()+b.get_height()/2,
                f'{v:.3f}', va='center', fontsize=12,
                fontweight='bold', color=P['text'])
ax_dpr.set_xlim(0,1.4)
ax_dpr.set_title('Demographic Parity Ratio\n(→ 1.0 = fairer)', fontsize=12)
ax_dpr.legend(fontsize=9); ax_dpr.grid(axis='x',zorder=0)

# per-group bars
ax_g = fig.add_subplot(gs[1,:])
x_   = np.arange(len(GROUPS)); bw2 = 0.3
b1 = ax_g.bar(x_-bw2/2-0.05, [base_pr[g] for g in GROUPS], bw2,
               label='Baseline',  color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax_g.bar(x_+bw2/2+0.05, [deb_pr[g]  for g in GROUPS], bw2,
               label='Debiased',  color=P['debiased'],  alpha=0.9, zorder=3)
for b in list(b1)+list(b2):
    h = b.get_height()
    ax_g.text(b.get_x()+b.get_width()/2, h+0.01,
              f'{h:.3f}', ha='center', va='bottom',
              fontsize=12, fontweight='bold', color=P['text'])
ax_g.set_xticks(x_); ax_g.set_xticklabels(GROUPS, fontsize=14)
ax_g.set_ylim(0, 0.85)
ax_g.set_ylabel('Positive Prediction Rate', fontsize=12)
ax_g.set_title('Per-Group Positive Prediction Rates: Baseline vs Debiased',
               fontsize=13)
ax_g.legend(fontsize=12); ax_g.grid(axis='y',zorder=0); ax_g.set_axisbelow(True)
save_fig(fig,'06_executive_dashboard.png')

# ═══════════════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print('\n'+'='*60)
print('PIPELINE COMPLETE — RESULTS SUMMARY')
print('='*60)
for label, m in [('Baseline', bm), ('Debiased', dm)]:
    print(f'\n{label}:')
    for k, v in m.items():
        print(f'  {k:12s}: {v:.4f}')

dpd_delta = abs(dm['DPD']) - abs(bm['DPD'])
eod_delta = abs(dm['EOD']) - abs(bm['EOD'])
acc_delta = dm['Accuracy'] - bm['Accuracy']

print(f'\n{"─"*40}')
print(f'|DPD| change (neg = fairer) : {dpd_delta:+.4f}')
print(f'|EOD| change (neg = fairer) : {eod_delta:+.4f}')
print(f'Accuracy delta              : {acc_delta:+.4f}')
print(f'\nAll charts saved to: {CHARTS_DIR}')
