# References:
#
# https://cpeg-gcep.net/sites/default/files/upload/bookfiles/CPEG_StatisticalMethods&Models_Jan30.pdf
#
# Data (stat parameters) downloaded from:
#
# https://cpeg-gcep.net/content/who-macro-files-cpeg-revision
#
# z-scores from WHO
#
# https://www.who.int/childgrowth/standards/height_for_age/en/
#
# https://www.who.int/childgrowth/standards/en/
#
# LMS params for Japan
#
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4860518/pdf/cpe-25-071.pdf

%matplotlib inline

from scipy.optimize.nonlin import NoConvergence
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sns
import scipy.stats as st

# # Read in LMS model parameters into a DataFrame `mp`

# ## WHO Reference Data (5-10 yr, girls&boys)

path = './growth_chart/who2007_R/wfawho2007.txt'
df = pd.read_csv(path, sep='\t')
df['metric'] = 'weight'
df['source'] = 'who_ref'
df.sex = df.sex.replace({1: 'm', 2:'f'})
df.head()

df.tail()

mp = df

path = './growth_chart/who2007_R/hfawho2007.txt'
df = pd.read_csv(path, sep='\t')
df['metric'] = 'height'
df['source'] = 'who_ref'
df.sex = df.sex.replace({1: 'm', 2:'f'})
df.head()

df.tail()

mp = pd.concat([df, mp])

# ## WHO Standard Data
#
# 6 tables total:
# * height 0-2yr for girls/boys
# * height 2-5yr for girls/boys
# * weight 0-5yr for girsl/boys

path = './growth_chart/lhfa_girls_0_2_zscores.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'f'
df['metric'] = 'height'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df.iloc[:-1]])  # remove last row to prevent duplicate

path = './growth_chart/tab_lhfa_girls_p_2_5.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'f'
df['metric'] = 'height'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df])

path = './growth_chart/wfa_girls_0_5_zscores.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'f'
df['metric'] = 'weight'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df])

path = './growth_chart/lhfa_boys_0_2_zscores.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'm'
df['metric'] = 'height'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df.iloc[:-1]])

path = './growth_chart/lhfa_boys_2_5_zscores.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'm'
df['metric'] = 'height'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df])

path = './growth_chart/wfa_boys_0_5_zscores.txt'
df = pd.read_csv(path, comment='#', sep='\t')
rename = dict(Month='age', L='l', M='m', S='s')
df = df.rename(columns=rename)[rename.values()]
df['sex'] = 'm'
df['metric'] = 'weight'
df['source'] = 'who_std'
df.head()

df.tail()

mp = pd.concat([mp, df])

mp['country'] = 'who'

path = './growth_chart/height_lms_jp.txt'
usecols = ['year', 'l', 'm', 's']
cols = ['year', 'l', 'm', 's', '_1', '_2', '_3']
f1 = pd.read_csv(path, sep='\t', skiprows=7, names=cols, usecols=usecols)
f1['sex'] = 'm'
cols = ['year', '_1', '_2', '_3', 'l', 'm', 's']
f2 = pd.read_csv(path, sep='\t', skiprows=7, names=cols, usecols=usecols)
f2['sex'] = 'f'
df = pd.concat([f1, f2])
df['age'] = (df.year * 12).astype(int)
df = df.drop('year', axis=1)
df['source'] = 'jp'
df['metric'] = 'height'
df['country'] = 'jp'
df.head()

mp = pd.concat([mp, df])

path = './growth_chart/weight_lms_jp.txt'
usecols = ['year', 'l', 'm', 's']
cols = ['year', 'l', 'm', 's', '_1', '_2', '_3']
f1 = pd.read_csv(path, sep='\t', skiprows=7, names=cols, usecols=usecols)
f1['sex'] = 'm'
cols = ['year', '_1', '_2', '_3', 'l', 'm', 's']
f2 = pd.read_csv(path, sep='\t', skiprows=7, names=cols, usecols=usecols)
f2['sex'] = 'f'
df = pd.concat([f1, f2])
df['age'] = (df.year * 12).astype(int)
df = df.drop('year', axis=1)
df['source'] = 'jp'
df['metric'] = 'weight'
df['country'] = 'jp'
df.head()

f1.tail()

mp = pd.concat([mp, df])

# Take out age 230 ... it makes down stream code simpler
assert (mp.age == 230).sum() == 2
mp = mp.loc[mp.age != 230]

# Make sure we have no duplicates
cols = ['sex', 'age', 'metric', 'country']
assert not mp[cols].duplicated().any()

# # Core function

def _get_metric(percentile, sex, agemo, metric, country='who'):
    rows = mp.query('sex==@sex and age==@agemo and country==@country and metric==@metric')
    assert len(rows) == 1, (rows, (percentile, sex, agemo, metric, country))
    r = rows.iloc[0]
    l, m, s = r['l'], r['m'], r['s']
    target_z = st.norm.ppf(percentile)
    def F(value):
        if l == 0:
            return np.log(value / m) / s - target_z
        return ((value / m) ** l - 1) / (s * l) - target_z
    value = scipy.optimize.broyden1(F, [m], f_tol=1e-4)
    return value[0]


def get_weight(percentile, sex, agemo, country='who'):
    return _get_metric(percentile, sex, agemo, 'weight', country)


def get_height(percentile, sex, agemo, country='who'):
    return _get_metric(percentile, sex, agemo, 'height', country)

sex = 'f'
agemo = 5 * 12 + 1
get_weight(0.5, sex, agemo), get_height(0.5, sex, agemo)

# # Get weights and heights at various percentiles over all ages

percentiles = np.array([3, 15, 50, 85, 97]) / 100
rows = []

cols = ['sex', 'age']
sex_age = mp.loc[mp.country == 'who'][cols].drop_duplicates().values

rows = []
for (sex, agemo), percentile in itertools.product(sex_age, percentiles):
    try:
        weight = get_weight(percentile, sex, agemo)
    except NoConvergence:
        weight = np.nan
    try:
        height = get_height(percentile, sex, agemo)
    except NoConvergence:
        height = np.nan
    rows.append(dict(weight=weight, height=height,
                     percentile=percentile, sex=sex, agemo=agemo))

df = pd.DataFrame(rows)
df['year'] = df.agemo / 12

# # Plot

colors = sns.diverging_palette(250, 300, sep=10, n=len(percentiles), center="dark")
colors = sns.diverging_palette(170, 300, sep=10, n=len(percentiles), center="dark")
colors = sns.diverging_palette(170, 360, sep=10, n=len(percentiles), center="dark")
sns.palplot(colors)


def fmt_inset(ax, c, xlabel, ylabel, xlims, ylims):
    ax.set_facecolor('none')
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    ax.tick_params(color=c, labelcolor=c)
    for y in list(range(ylims[0], ylims[1]+1, 10))[1:-1]:
        ax.axhline(y, color=c, lw=3, ls='--', alpha=0.5)
    for x in list(range(xlims[0], xlims[1]+1, 2))[1:-1]:
        ax.axvline(x, color=c, lw=3, ls='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(c)
        spine.set_linewidth(6)
        spine.set_alpha(0.6)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.xaxis.label.set_color(c)
    ax.yaxis.label.set_color(c)
    return

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig, _ax = plt.subplots(figsize=(20, 20), facecolor='w')
img = plt.imread('./growth_chart/growth_chart_girl_canada_scan.png')

_ax.imshow(img, interpolation='bilinear', alpha=0.6)
# ax.axis('off')
# ax.set_yticks([]), ax.set_yticklabels([])
# ax.set_xticks([]), ax.set_xticklabels([])

ax = inset_axes(_ax, width="100%", height="100%",
                bbox_to_anchor=(100, 640, 290, 298), # l, t, w, h
                bbox_transform=_ax.transData,
                )
fmt_inset(ax, 'green', 'age (year)', 'weight (kg)', (0, 10), (0, 65))
    
gby = (df
       .query('sex=="f" and year<=10')
       .set_index('year')
       .sort_values('weight')
       .groupby('percentile'))

k = dict(lw=10, alpha=0.5, solid_capstyle='round')

for j, (pct, dg) in enumerate(gby):
    if pct not in [0.03, 0.5, 0.97]:
        continue
    c = colors[j]
    dg.weight.plot(ax=ax, color=c, **k)
    x = 5
    y = dg.loc[x].weight
    x1, x2 = x-1/12, x+1/12
    y1, y2 = dg.loc[x1].weight, dg.loc[x2].weight
    slope = (y2-y1)/(x2-x1)
    rot, offset = slope * 6, slope / 13  # magic
    ax.text(x, y+offset, f'{pct*100:.0f}', color=c,
            va='bottom', ha='center', size=20,
            fontdict=dict( rotation=rot, size='large', weight='bold', ))
    
ax = inset_axes(_ax, width="100%", height="100%",
                bbox_to_anchor=(70, 295, 610, 506), # l, t, w, h
                bbox_transform=_ax.transData,
                )
fmt_inset(ax, 'm', 'age (year)', 'height (cm)', (-1, 20), (70, 180))
ax.set_xticks(range(0, 20, 2))
    
gby = (df
       .query('sex=="f"')
       .set_index('year')
       .sort_values('height')
       .groupby('percentile'))

k = dict(lw=10, alpha=0.5, solid_capstyle='round')

for j, (pct, dg) in enumerate(gby):
    if pct not in [0.03, 0.5, 0.97]:
        continue
    c = colors[j]
    dg.height.plot(ax=ax, color=c, **k)
    x = 14
    y = dg.loc[x].height
    x1, x2 = x-1/12, x+1/12
    y1, y2 = dg.loc[x1].height, dg.loc[x2].height
    slope = (y2-y1)/(x2-x1)
    rot, offset = slope * 6, 0#slope / 13  # magic
    ax.text(x, y+offset, f'{pct*100:.0f}', color=c,
            va='bottom', ha='center', size=20,
            fontdict=dict( rotation=rot, size='large', weight='bold', ))
    
plt.show()

sex = 'f'
agemo = 5 * 12 + 6

fig, axes = plt.subplots(figsize=(12, 4), ncols=2)

def _do(ax, mean, sigma, color):
    x = np.linspace(mean-3*sigma, mean+3*sigma, 100)
    y = st.norm.pdf(x, mean, sigma)
    y = y / y.max()
    ax.plot(x, y, color=color, lw=4, alpha=0.8)
    ax.axvline(mean, color='grey')
    for sign in [-1, 1]:
        ax.axvline(mean+sign*sigma, color='grey', ls='--')
        ax.axvline(mean+sign*2*sigma, color='grey', ls=':')
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
y, m = agemo // 12, agemo % 12

ax = axes[0]
mean = get_weight(0.5, sex, agemo)
sigma = mean - get_weight(st.norm.cdf(-1), sex, agemo)
_do(ax, mean, sigma, 'g')
ax.set_xlabel('weight (kg)')
ax.set_title(f'weight distribution at {y}yr {m}mo')

ax = axes[1]
mean = get_height(0.5, sex, agemo)
sigma = mean - get_height(st.norm.cdf(-1), sex, agemo)
_do(ax, mean, sigma, 'm')
ax.plot([100, 100], [0, 0.2], "-", c='m', lw=2, alpha=1)
ax.plot(100, 0.2, "*", ms=20, mec='m', mfc='salmon', mew=2)
ax.set_xlabel('height (cm)')
ax.set_title(f'height distribution at {y}yr {m}mo')

plt.tight_layout()
plt.show()

path = "./growth_chart/Set-1-HFA-WFA_2-19_BOYS_EN.png"
img_boy = plt.imread(path)
path = "./growth_chart/Set-1-HFA-WFA_2-19_GIRLS_EN_Extended.png"
img_girl = plt.imread(path)

fig, axes = plt.subplots(figsize=(8, 6), ncols=2)
imgs = [img_girl, img_boy]
for i, sex in enumerate('fm'):
    ax = axes[i]
    gby = (df.loc[df.sex==sex]
           .query('year<=10')
           .set_index('year')
           .groupby('percentile')
           )
    k = dict(lw=10, alpha=0.5, solid_capstyle='round')
    for j, (pct, dg) in enumerate(gby):
        c = colors[j]
        dg.weight.sort_values().plot(ax=ax, color=c, **k)
        x = 9
        y = dg.loc[x].weight
        x1, x2 = x-1/12, x+1/12
        y1, y2 = dg.loc[x1].weight, dg.loc[x2].weight
        slope = (y2-y1)/(x2-x1)
        rot, offset = slope * 6, slope / 13  # magic
        ax.text(x, y+offset, f'{pct*100:.0f}', color=c,
                va='bottom',
                ha='center',
                fontdict=dict(
                    rotation=rot,
                    size='large',
                    weight='bold',
                ))
        
    ax.grid()
    ax.set_xlim(5, 10)
    ax.set_ylim(10, 50)
    ax.imshow(imgs[i], extent=(5, 10, 10, 50), aspect='auto')
    ax.set_title('GIRL' if sex == 'f' else 'BOY')
    if i == 0:
        ax.set_ylabel('weight (kg)')
    ax.set_xlabel('age (year)')
plt.show()

# # Get weights and heights at various percentiles over all ages

percentiles = np.array([3, 15, 50, 85, 97]) / 100
rows = []

cols = ['sex', 'age']
sex_age = mp.loc[mp.country == 'jp'][cols].drop_duplicates().values

rows = []
for (sex, agemo), percentile in itertools.product(sex_age, percentiles):
    try:
        weight = get_weight(percentile, sex, agemo, country='jp')
    except NoConvergence:
        weight = np.nan
    try:
        height = get_height(percentile, sex, agemo, country='jp')
    except NoConvergence:
        height = np.nan
    rows.append(dict(weight=weight, height=height,
                     percentile=percentile, sex=sex, agemo=agemo))

dj = pd.DataFrame(rows)
dj['year'] = dj.agemo / 12

    
fig, axes = plt.subplots(figsize=(12, 5), facecolor='w', ncols=2)

k = dict(lw=3, alpha=0.8, solid_capstyle='round')
percentiles_to_plot = [0.03, 0.5, 0.97]
def add_legend(ax):
    
    handles = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=k['alpha'], edgecolor='none')
               for c, p in zip(colors, percentiles) if p in percentiles_to_plot][::-1]
    labels = ['3rd', '50th', '97th'][::-1]
    leg1 = ax.legend(handles, labels, loc='lower right', frameon=False,
              prop=dict(family='monospace'))
    
    labels= ['WHO', 'JAPAN']
    handles = [
        plt.Line2D((0,1),(0,0), color='grey', ls='-', **k),
        plt.Line2D((0,1),(0,0), color='grey', ls='--', **k),
    ]
    ax.legend(handles, labels, loc='upper left', frameon=False,
              prop=dict(family='monospace'))
    ax.add_artist(leg1)
    return

ax = axes[0]
for i, f in enumerate([df, dj]):
    gby = (f
           .query('sex=="f" and year<=10')
           .set_index('year')
           .sort_values('weight')
           .groupby('percentile'))

    for j, (pct, dg) in enumerate(gby):
        if pct not in percentiles_to_plot:
            continue
        c = colors[j]
        ls = ['-', '--'][i]
        line = dg.weight.plot(ax=ax, color=c, ls=ls, **k)
ax.set_ylabel('weight (kg)')
ax.set_title('Weight')
add_legend(ax)

ax = axes[1]
for i, f in enumerate([df, dj]):
    gby = (f
           .query('sex=="f"')
           .set_index('year')
           .sort_values('height')
           .groupby('percentile'))

    for j, (pct, dg) in enumerate(gby):
        if pct not in percentiles_to_plot:
            continue
        c = colors[j]
        ls = ['-', '--'][i]
        dg.height.plot(ax=ax, color=c, ls=ls, **k)
ax.set_xticks(range(0, 21, 2))
ax.set_ylabel('height (cm)')
ax.set_title('Height')
add_legend(ax)
plt.show()

sex = 'f'
agemo = 5 * 12 + 6
percentile = st.norm.cdf(-1)
target_z = st.norm.ppf(percentile)

fig, axes = plt.subplots(figsize=(12, 4), ncols=2)

def _do(ax, mean, sigma, color, ls, label):
    x = np.linspace(mean-3*sigma, mean+3*sigma, 100)
    y = st.norm.pdf(x, mean, sigma)
    y = y / y.max()
    ax.plot(x, y, color=color, lw=4, alpha=0.8, ls=ls, label=label)
    ax.axvline(mean, color='grey')
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
y, m = agemo // 12, agemo % 12

ax = axes[0]
mean = get_weight(0.5, sex, agemo)
sigma = mean - get_weight(st.norm.cdf(-1), sex, agemo)
_do(ax, mean, sigma, 'g', '-', 'WHO')
mean = get_weight(0.5, sex, agemo, country='jp')
sigma = mean - get_weight(percentile, sex, agemo, country='jp')
_do(ax, mean, sigma, 'g', ':', 'JAPAN')
ax.set_xlabel('weight (kg)')
ax.set_title(f'weight distribution at {y}yr {m}mo')
ax.legend(loc='upper left', frameon=False,
          prop=dict(family='monospace'))

ax = axes[1]
mean = get_height(0.5, sex, agemo)
sigma = mean - get_height(st.norm.cdf(-1), sex, agemo)
_do(ax, mean, sigma, 'm', '-', 'WHO')
mean = get_height(0.5, sex, agemo, country='jp')
sigma = mean - get_height(percentile, sex, agemo, country='jp')
_do(ax, mean, sigma, 'm', ':', 'JAPAN')
ax.set_xlabel('weight (kg)')
ax.plot([100, 100], [0, 0.2], "-", c='m', lw=2, alpha=1)
ax.plot(100, 0.2, "*", ms=20, mec='m', mfc='salmon', mew=2)
ax.set_xlabel('height (cm)')
ax.set_title(f'height distribution at {y}yr {m}mo')
ax.legend(loc='upper left', frameon=False,
          prop=dict(family='monospace'))

plt.tight_layout()
plt.show()

mean = get_height(0.5, sex, agemo)
sigma = mean - get_height(st.norm.cdf(-1), sex, agemo)
percentile = st.norm.cdf(100, loc=mean, scale=sigma)
z_score = st.norm.ppf(percentile)
f'{percentile:.1%}', z_score

mean = get_height(0.5, sex, agemo, country='jp')
sigma = mean - get_height(st.norm.cdf(-1), sex, agemo, country='jp')
percentile = st.norm.cdf(100, loc=mean, scale=sigma)
z_score = st.norm.ppf(percentile)
f'{percentile:.1%}', z_score

