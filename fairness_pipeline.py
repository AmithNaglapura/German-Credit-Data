"""
Fairness-Aware Machine Learning Pipeline — v2
==============================================

Dataset   : German Credit Data (german_credit_data.csv)
Task      : Binary credit-risk classification  (0 = good, 1 = bad)
Sensitive : Sex (male / female)  AND  Age (Young / Middle / Senior)
Baseline  : Logistic Regression (no fairness constraint)
Debiased  : Logistic Regression + per-group decision thresholds
            (Threshold Optimisation, demographic-parity constraint)
Libraries : scikit-learn, fairlearn, pandas, matplotlib, seaborn
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from fairlearn.metrics import (demographic_parity_difference,
                               equalized_odds_difference,
                               demographic_parity_ratio)
from fairlearn.postprocessing import ThresholdOptimizer

warnings.filterwarnings('ignore')

# ── paths ─────────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.abspath(__file__))
CHARTS_DIR = os.path.join(BASE, 'charts_v2')
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── palette & plot style ──────────────────────────────────────────────────────
P = dict(
    baseline='#4361EE', debiased='#F72585',
    male='#4CC9F0',     female='#FF9F1C',
    young='#06D6A0',    middle='#4361EE',   senior='#FF6B6B',
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
                fontsize=9, fontweight='bold', color=P['text'])

# ═════════════════════════════════════════════════════════════════════════════
# 1. LOAD & ENGINEER TARGET
# ═════════════════════════════════════════════════════════════════════════════

df = pd.read_csv(os.path.join(BASE, 'german_credit_data.csv'), index_col=0)
print(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

sav_map = {'little': 0, 'moderate': 1, 'quite rich': 2, 'rich': 3}
chk_map = {'little': 0, 'moderate': 1, 'rich': 2}
df['sav'] = df['Saving accounts'].map(sav_map).fillna(-1)
df['chk'] = df['Checking account'].map(chk_map).fillna(-1)

score = (
    (df['Credit amount'] > df['Credit amount'].quantile(0.60)).astype(int) * 2 +
    (df['Duration']      > df['Duration'].quantile(0.60)).astype(int) * 2 +
    (df['sav'] < 1).astype(int) +
    (df['chk'] < 1).astype(int)
)
df['Risk'] = (score >= 4).astype(int)

print(f"Overall bad rate : {df['Risk'].mean():.1%}")
print("Bad rate by Sex:")
print(df.groupby('Sex')['Risk'].mean().rename('bad_rate'))

# ── Age grouping (tertile-based, balanced) ────────────────────────────────────
age_bins   = [0, 30, 49, 200]
age_labels = ['Young (≤30)', 'Middle (31–49)', 'Senior (50+)']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

print("\nBad rate by AgeGroup:")
print(df.groupby('AgeGroup', observed=True)['Risk'].mean().rename('bad_rate'))
print("\nAge group counts:")
print(df['AgeGroup'].value_counts().sort_index())

# ═════════════════════════════════════════════════════════════════════════════
# 2. PRE-PROCESS
# ═════════════════════════════════════════════════════════════════════════════

FEATURES = ['Age','Sex','Job','Housing',
            'Saving accounts','Checking account',
            'Credit amount','Duration','Purpose']

data = df[FEATURES + ['Risk']].copy()

cat_cols = ['Sex','Housing','Saving accounts','Checking account','Purpose']
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].fillna('Unknown'))

sensitive_sex = df['Sex'].values            # 'male' | 'female'
sensitive_age = df['AgeGroup'].astype(str).values   # 'Young' | 'Middle' | 'Senior'

X        = data[FEATURES].values
y        = data['Risk'].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

(X_tr, X_te,
 y_tr, y_te,
 ssex_tr, ssex_te,
 sage_tr, sage_te) = train_test_split(
    X_scaled, y, sensitive_sex, sensitive_age,
    test_size=0.25, random_state=42, stratify=y)

print(f"\nTrain {len(X_tr)} | Test {len(X_te)}")
print(f"Sex groups : {np.unique(ssex_te)}")
print(f"Age groups : {np.unique(sage_te)}")

# ═════════════════════════════════════════════════════════════════════════════
# 3. BASELINE MODEL
# ═════════════════════════════════════════════════════════════════════════════

base_clf = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
base_clf.fit(X_tr, y_tr)
yp_base = base_clf.predict(X_te)

def calc_metrics(y_true, y_pred, sens):
    return dict(
        Accuracy  = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall    = recall_score(y_true, y_pred, zero_division=0),
        F1        = f1_score(y_true, y_pred, zero_division=0),
        DPD       = demographic_parity_difference(y_true, y_pred, sensitive_features=sens),
        DPR       = demographic_parity_ratio(y_true, y_pred, sensitive_features=sens),
        EOD       = equalized_odds_difference(y_true, y_pred, sensitive_features=sens),
    )

bm_sex = calc_metrics(y_te, yp_base, ssex_te)
bm_age = calc_metrics(y_te, yp_base, sage_te)

print("\n=== BASELINE — Sex attribute ===")
for k, v in bm_sex.items():
    print(f"  {k}: {v:.4f}")

print("\n=== BASELINE — Age attribute ===")
for k, v in bm_age.items():
    print(f"  {k}: {v:.4f}")

SEX_GROUPS = np.unique(ssex_te)
AGE_GROUPS = sorted(np.unique(sage_te),
                    key=lambda x: ['Young (≤30)','Middle (31–49)','Senior (50+)'].index(x)
                    if x in ['Young (≤30)','Middle (31–49)','Senior (50+)'] else 0)

base_pr_sex = {g: yp_base[ssex_te==g].mean() for g in SEX_GROUPS}
base_pr_age = {g: yp_base[sage_te==g].mean() for g in AGE_GROUPS}

# ═════════════════════════════════════════════════════════════════════════════
# 4. DEBIASED MODEL — Sex threshold optimisation
# ═════════════════════════════════════════════════════════════════════════════

inner_sex = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
inner_sex.fit(X_tr, y_tr)

to_sex = ThresholdOptimizer(
    estimator      = inner_sex,
    constraints    = "demographic_parity",
    objective      = "balanced_accuracy_score",
    predict_method = "predict_proba",
    flip           = True,
)
to_sex.fit(X_tr, y_tr, sensitive_features=ssex_tr)
yp_deb_sex = to_sex.predict(X_te, sensitive_features=ssex_te, random_state=42)

dm_sex = calc_metrics(y_te, yp_deb_sex, ssex_te)

print("\n=== DEBIASED — Sex attribute ===")
for k, v in dm_sex.items():
    print(f"  {k}: {v:.4f}")

deb_pr_sex = {g: yp_deb_sex[ssex_te==g].mean() for g in SEX_GROUPS}

# ═════════════════════════════════════════════════════════════════════════════
# 5. DEBIASED MODEL — Age threshold optimisation
# ═════════════════════════════════════════════════════════════════════════════

inner_age = LogisticRegression(max_iter=1000, random_state=42, C=0.5)
inner_age.fit(X_tr, y_tr)

to_age = ThresholdOptimizer(
    estimator      = inner_age,
    constraints    = "demographic_parity",
    objective      = "balanced_accuracy_score",
    predict_method = "predict_proba",
    flip           = True,
)
to_age.fit(X_tr, y_tr, sensitive_features=sage_tr)
yp_deb_age = to_age.predict(X_te, sensitive_features=sage_te, random_state=42)

dm_age = calc_metrics(y_te, yp_deb_age, sage_te)

print("\n=== DEBIASED — Age attribute ===")
for k, v in dm_age.items():
    print(f"  {k}: {v:.4f}")

deb_pr_age = {g: yp_deb_age[sage_te==g].mean() for g in AGE_GROUPS}

# ═════════════════════════════════════════════════════════════════════════════
# 6. CHARTS
# ═════════════════════════════════════════════════════════════════════════════

BW = 0.35

# ── Chart A: Sex — Predictive Performance ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
keys = ['Accuracy','Precision','Recall','F1']
x    = np.arange(len(keys))
b1 = ax.bar(x-BW/2, [bm_sex[k] for k in keys], BW, label='Baseline',
            color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x+BW/2, [dm_sex[k] for k in keys], BW,
            label='Debiased (Threshold Opt.)', color=P['debiased'], alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2))
ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=12)
ax.set_ylim(0,1.15); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Predictive Performance: Baseline vs Debiased Model\n'
             'Sensitive Attribute: Sex  |  Task: Credit Risk Classification', pad=12)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'01_sex_predictive_metrics.png')

# ── Chart B: Age — Predictive Performance ───────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
b1 = ax.bar(x-BW/2, [bm_age[k] for k in keys], BW, label='Baseline',
            color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x+BW/2, [dm_age[k] for k in keys], BW,
            label='Debiased (Threshold Opt.)', color=P['debiased'], alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2))
ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=12)
ax.set_ylim(0,1.15); ax.set_ylabel('Score', fontsize=12)
ax.set_title('Predictive Performance: Baseline vs Debiased Model\n'
             'Sensitive Attribute: Age Group  |  Task: Credit Risk Classification', pad=12)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'02_age_predictive_metrics.png')

# ── Chart C: Sex — Fairness Metrics ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10,6))
fkeys = ['DPD','EOD']
flbls = ['Demographic\nParity Difference','Equalized\nOpportunity Difference']
bfv = [abs(bm_sex[k]) for k in fkeys]
dfv = [abs(dm_sex[k]) for k in fkeys]
x_  = np.arange(len(fkeys))
b1 = ax.bar(x_-BW/2, bfv, BW, label='Baseline', color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x_+BW/2, dfv, BW, label='Debiased (Threshold Opt.)',
            color=P['debiased'], alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2), dy=0.005)
ceiling = max(max(bfv),max(dfv))*1.6+0.05
ax.set_xticks(x_); ax.set_xticklabels(flbls, fontsize=12)
ax.set_ylim(0, ceiling); ax.set_ylabel('|Metric Value|  — lower = fairer', fontsize=12)
ax.set_title('Fairness Metrics: Baseline vs Debiased Model\nSensitive Attribute: Sex', pad=12)
ax.axhline(0.1, color=P['gold'], lw=1.8, ls='--', label='Fairness threshold (0.1)', zorder=4)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'03_sex_fairness_metrics.png')

# ── Chart D: Age — Fairness Metrics ─────────────────────────────────────────
bfv_age = [abs(bm_age[k]) for k in fkeys]
dfv_age = [abs(dm_age[k]) for k in fkeys]

fig, ax = plt.subplots(figsize=(10,6))
b1 = ax.bar(x_-BW/2, bfv_age, BW, label='Baseline', color=P['baseline'], alpha=0.9, zorder=3)
b2 = ax.bar(x_+BW/2, dfv_age, BW, label='Debiased (Threshold Opt.)',
            color=P['debiased'], alpha=0.9, zorder=3)
bar_labels(ax, list(b1)+list(b2), dy=0.005)
ceiling_age = max(max(bfv_age),max(dfv_age))*1.6+0.05
ax.set_xticks(x_); ax.set_xticklabels(flbls, fontsize=12)
ax.set_ylim(0, ceiling_age); ax.set_ylabel('|Metric Value|  — lower = fairer', fontsize=12)
ax.set_title('Fairness Metrics: Baseline vs Debiased Model\nSensitive Attribute: Age Group', pad=12)
ax.axhline(0.1, color=P['gold'], lw=1.8, ls='--', label='Fairness threshold (0.1)', zorder=4)
ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.tight_layout()
save_fig(fig,'04_age_fairness_metrics.png')

# ── Chart E: Per-group positive rates — Sex ──────────────────────────────────
gcols_sex = [P['female'] if g=='female' else P['male'] for g in SEX_GROUPS]
fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
for ax, pr, title in zip(axes,
                          [base_pr_sex, deb_pr_sex],
                          ['Baseline Model','Debiased Model (Threshold Opt.)']):
    vals = [pr[g] for g in SEX_GROUPS]
    bars = ax.bar(SEX_GROUPS, vals, color=gcols_sex, alpha=0.9,
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
pm = mpatches.Patch(color=P['male'], label='Male')
pf = mpatches.Patch(color=P['female'], label='Female')
fig.legend(handles=[pm,pf], loc='upper right', fontsize=11)
fig.tight_layout()
save_fig(fig,'05_sex_group_positive_rates.png')

# ── Chart F: Per-group positive rates — Age ──────────────────────────────────
age_colors = [P['young'], P['middle'], P['senior']]

fig, axes = plt.subplots(1,2,figsize=(14,5),sharey=True)
for ax, pr, title in zip(axes,
                          [base_pr_age, deb_pr_age],
                          ['Baseline Model','Debiased Model (Threshold Opt.)']):
    vals = [pr.get(g,0) for g in AGE_GROUPS]
    short_labels = ['Young\n(≤30)', 'Middle\n(31–49)', 'Senior\n(50+)']
    bars = ax.bar(short_labels, vals, color=age_colors[:len(AGE_GROUPS)],
                  alpha=0.9, edgecolor=P['bg'], linewidth=1.5, zorder=3)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold', color=P['text'])
    ax.set_ylim(0,0.9)
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_ylabel('Positive Prediction Rate\n(predicted bad credit)',fontsize=11)
    ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)
fig.suptitle('Per-Group Positive Prediction Rates by Age Group',
             fontsize=14, fontweight='bold', y=1.03)
py = mpatches.Patch(color=P['young'],  label='Young (≤30)')
pm = mpatches.Patch(color=P['middle'], label='Middle (31–49)')
ps = mpatches.Patch(color=P['senior'], label='Senior (50+)')
fig.legend(handles=[py,pm,ps], loc='upper right', fontsize=11)
fig.tight_layout()
save_fig(fig,'06_age_group_positive_rates.png')

# ── Chart G: Combined Fairness — Side-by-side attribute comparison ────────────
fig, axes = plt.subplots(1,2,figsize=(15,6))
metrics_labels = ['|DPD| Baseline','|DPD| Debiased','|EOD| Baseline','|EOD| Debiased']
attr_labels     = ['Sex','Age']
x_g = np.arange(len(metrics_labels))

sex_vals = [abs(bm_sex['DPD']), abs(dm_sex['DPD']),
            abs(bm_sex['EOD']), abs(dm_sex['EOD'])]
age_vals = [abs(bm_age['DPD']), abs(dm_age['DPD']),
            abs(bm_age['EOD']), abs(dm_age['EOD'])]

for ax, vals, title, color in zip(
        axes,
        [sex_vals, age_vals],
        ['Sensitive Attribute: Sex', 'Sensitive Attribute: Age Group'],
        [P['male'], P['young']]):
    bars = ax.bar(x_g, vals, color=[P['baseline'],P['debiased'],
                                    P['baseline'],P['debiased']],
                  alpha=0.9, zorder=3)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.005,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', color=P['text'])
    ax.axhline(0.1, color=P['gold'], lw=1.8, ls='--',
               label='Threshold (0.1)', zorder=4)
    ax.set_xticks(x_g); ax.set_xticklabels(metrics_labels, fontsize=10)
    ax.set_ylim(0, max(vals)*1.6+0.05)
    ax.set_ylabel('|Metric Value|', fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.legend(fontsize=10); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)

fig.suptitle('Fairness Metrics Comparison: Sex vs Age Group\n'
             'Baseline and Debiased Models', fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig,'07_combined_fairness_comparison.png')

# ── Chart H: Intersectional — Sex × Age positive rates (baseline) ────────────
fig, axes = plt.subplots(1,2,figsize=(15,6))
age_short = {'Young (≤30)':'Young\n(≤30)','Middle (31–49)':'Middle\n(31–49)','Senior (50+)':'Senior\n(50+)'}

for ax, yp, title in zip(axes, [yp_base, yp_deb_sex],
                           ['Baseline Model','Debiased (Sex) Model']):
    male_rates   = []
    female_rates = []
    for ag in AGE_GROUPS:
        mask_m = (ssex_te=='male')   & (sage_te==ag)
        mask_f = (ssex_te=='female') & (sage_te==ag)
        male_rates.append(yp[mask_m].mean() if mask_m.sum()>0 else 0)
        female_rates.append(yp[mask_f].mean() if mask_f.sum()>0 else 0)

    xl = np.arange(len(AGE_GROUPS))
    b1 = ax.bar(xl-0.2, male_rates,   0.38, label='Male',   color=P['male'],   alpha=0.9, zorder=3)
    b2 = ax.bar(xl+0.2, female_rates, 0.38, label='Female', color=P['female'], alpha=0.9, zorder=3)
    for b,v in zip(list(b1)+list(b2), male_rates+female_rates):
        ax.text(b.get_x()+b.get_width()/2, v+0.012,
                f'{v:.3f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color=P['text'])
    ax.set_xticks(xl)
    ax.set_xticklabels([age_short[a] for a in AGE_GROUPS], fontsize=12)
    ax.set_ylim(0,0.95); ax.set_ylabel('Positive Prediction Rate', fontsize=11)
    ax.set_title(title, fontsize=13, pad=8)
    ax.legend(fontsize=11); ax.grid(axis='y',zorder=0); ax.set_axisbelow(True)

fig.suptitle('Intersectional Analysis: Positive Prediction Rates by Sex × Age Group',
             fontsize=14, fontweight='bold')
fig.tight_layout()
save_fig(fig,'08_intersectional_sex_age.png')

# ═════════════════════════════════════════════════════════════════════════════
# 7. FINAL SUMMARY
# ═════════════════════════════════════════════════════════════════════════════

print('\n'+'='*65)
print('PIPELINE COMPLETE — RESULTS SUMMARY')
print('='*65)

print('\n--- SEX ATTRIBUTE ---')
for label, m in [('Baseline', bm_sex), ('Debiased', dm_sex)]:
    print(f'\n  {label}:')
    for k, v in m.items():
        print(f'    {k:12s}: {v:.4f}')

print('\n--- AGE ATTRIBUTE ---')
for label, m in [('Baseline', bm_age), ('Debiased', dm_age)]:
    print(f'\n  {label}:')
    for k, v in m.items():
        print(f'    {k:12s}: {v:.4f}')

print(f'\n{"─"*50}')
print('FAIRNESS DELTA SUMMARY')
print(f'{"─"*50}')
print('Sex:')
print(f'  |DPD| change : {abs(dm_sex["DPD"])-abs(bm_sex["DPD"]):+.4f}')
print(f'  |EOD| change : {abs(dm_sex["EOD"])-abs(bm_sex["EOD"]):+.4f}')
print(f'  Accuracy Δ   : {dm_sex["Accuracy"]-bm_sex["Accuracy"]:+.4f}')
print('Age:')
print(f'  |DPD| change : {abs(dm_age["DPD"])-abs(bm_age["DPD"]):+.4f}')
print(f'  |EOD| change : {abs(dm_age["EOD"])-abs(bm_age["EOD"]):+.4f}')
print(f'  Accuracy Δ   : {dm_age["Accuracy"]-bm_age["Accuracy"]:+.4f}')

print(f'\nAll charts saved to: {CHARTS_DIR}')
