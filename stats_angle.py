# %% import depandency here
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns
# %% define function here

def print_dict_tree(d, prefix="", key_name=None, is_last=True, root_type=None):
    """
    以严格tree风格递归展示多级字典结构，展示类型信息。
    """
    # 顶层类型
    if prefix == "":
        tname = type(d).__name__ if root_type is None else root_type
        print(tname)
    # 构造当前行前缀
    branch = "└── " if is_last else "├── "
    if key_name is not None:
        print(f"{prefix}{branch}{key_name}: {type(d).__name__}")
    # 递归
    if isinstance(d, dict):
        n = len(d)
        for idx, (k, v) in enumerate(d.items()):
            last = (idx == n - 1)
            child_prefix = prefix + ("    " if is_last and key_name is not None else "│   ")
            print_dict_tree(v, child_prefix, k, last)
    elif isinstance(d, list):
        if len(d) > 0:
            print(f"{prefix}{branch}[0]: {type(d[0]).__name__}")
            child_prefix = prefix + ("    " if is_last and key_name is not None else "│   ")
            print_dict_tree(d[0], child_prefix, None, True)
        else:
            print(f"{prefix}{branch}[]: empty list")
    else:
        # 只展示类型，不递归
        pass

def find_data_folders(root: Path):
    """
    递归查找所有包含lifting_angles.pkl和lifting_angles_gmm_models.pkl的数据文件夹。
    返回这些文件夹的Path对象列表。
    """
    folders = []
    for p in root.rglob("*"):
        if p.is_dir():
            angles = p / "lifting_angles.pkl"
            gmm = p / "lifting_angles_gmm_models.pkl"
            if angles.exists() and gmm.exists():
                folders.append(p)
    return folders

def load_data_from_folders(folders, root):
    """
    读取每个数据文件夹下的pkl文件，组成多级字典。
    顶层字典的键为数据文件夹的相对路径（linux风格字符串）。
    """
    data_dict = {}
    for folder in folders:
        rel_path = str(folder.relative_to(root))
        try:
            with open(folder / "lifting_angles.pkl", "rb") as f1:
                angles = pickle.load(f1)
            with open(folder / "lifting_angles_gmm_models.pkl", "rb") as f2:
                gmm = pickle.load(f2)
            data_dict[rel_path] = {
                "lifting_angles": angles,
                "lifting_angles_gmm_models": gmm
            }
        except Exception as e:
            print(f"Failed to load data from {folder}: {e}")
    return data_dict


def calculate_angle_js(data1, data2, bin_width=1.0):
    """
    计算两组角度 [-90, 90] 分布的 JS 散度
    :param bin_width: 桶宽，默认 1度
    """
    # 1. 统一设定分桶边界，确保 P 和 Q 的维度和区间完全一致
    bins = np.arange(-90, 90 + bin_width, bin_width)
    
    # 2. 计算直方图并归一化为概率分布
    p, _ = np.histogram(data1, bins=bins, density=True)
    q, _ = np.histogram(data2, bins=bins, density=True)
    
    # 3. 计算 JS 距离 (Jensen-Shannon Distance)
    # scipy 内部会自动处理零概率问题（通过加权平均 M）
    js_dist = jensenshannon(p, q)
    
    # 4. 得到 JS 散度 (JS Divergence)
    js_div = js_dist ** 2
    
    return js_dist, js_div

# --- 点圆图绘制函数 ---
def plot_gmm_circle_matrix(param_matrix, weight_matrix, data_names, xlabel, title, colors, figsize=(10,6)):
    fig, ax = plt.subplots(figsize=figsize)
    n_data, n_comp = param_matrix.shape
    mins = param_matrix.min(axis=1)
    maxs = param_matrix.max(axis=1)
    color_handles = []
    for i in range(n_data):
        for j in range(n_comp):
            ax.scatter(param_matrix[i, j], i, s=weight_matrix[i, j]*1000, alpha=0.6, color=colors(i))
    # 为每个主成分画min/max竖线并构造图例
    for i in range(n_data):
        min_val = mins[i]
        max_val = maxs[i]
        ax.axvline(min_val, color=colors(i), linestyle='--', linewidth=0.5, alpha=0.5)
        ax.axvline(max_val, color=colors(i), linestyle='--', linewidth=0.5, alpha=0.5)
        color_handles.append(
            mpatches.Patch(color=colors(i), label=f' {data_names[i]}: min={min_val:.2f}, max={max_val:.2f}')
        )
    ax.legend(handles=color_handles, title="Component min/max", loc='best', fontsize=9)
    ax.set_yticks(range(n_data))
    ax.set_yticklabels(data_names)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Data')
    ax.set_title(title)
    plt.tight_layout()
    return fig

# %% scan and load data here
# 递归查找包含指定pkl文件的数据文件夹，并读取数据组成多级字典
data_root = Path("infer/ballbar-newCboard")
data_folders = find_data_folders(data_root)
all_data = load_data_from_folders(data_folders, data_root)
print_dict_tree(all_data)
"""
dict
│   ├── 1: dict
│   │   ├── lifting_angles: dict
│   │   │   ├── angles: ndarray
│   │   │   ├── gaussan_filtered_angles: ndarray
│   │   │   ├── gaussian_stats: dict
│   │   │   │   ├── raw_mean: float
│   │   │   │   ├── raw_std: float
│   │   │   │   ├── filtered_mean: float
│   │   │   │   ├── filtered_std: float
│   │   │   │   ├── keep_ratio: float
│   │   │   │   └── sigma: int
│   │   │   └── angle_groups: dict
│   │   │       ├── err<= 3.668: ndarray
│   │   │       ├── err<= 1.998: ndarray
│   │   │       └── err<= 1.042: ndarray
│   │   └── lifting_angles_gmm_models: dict
│   │       ├── 1: GaussianMixture
│   │       ├── 2: GaussianMixture
│   │       ├── 3: GaussianMixture
│   │       ├── 4: GaussianMixture
│   │       ├── 5: GaussianMixture
│   │       ├── 6: GaussianMixture
│   │       ├── 7: GaussianMixture
│   │       ├── 8: GaussianMixture
│   │       └── 9: GaussianMixture
... (more folders with the same structure)
"""

# %% GMM feature angle eistiamtion
# shared n_components
gmm_out_dir = data_root / "gmm_plots"
gmm_out_dir.mkdir(exist_ok=True)

data_name = list(all_data.keys())
shared_n_components = set(all_data[data_name[0]]

["lifting_angles_gmm_models"].keys())
for i in range(1, len(data_name)):
    shared_n_components = shared_n_components & set(all_data[data_name[i]]["lifting_angles_gmm_models"].keys())
shared_n_components = sorted(list(shared_n_components), key=lambda x: int(x) if str(x).isdigit() else x)

cmap = 'tab10'
gmm_sammary = {}

for n_comp in shared_n_components:
    gams = {}
    for name in data_name:
        gmm = all_data[name]["lifting_angles_gmm_models"][n_comp]
        gams[name] = {
            "weights": gmm.weights_,
            "means": gmm.means_.flatten(),
            "std": np.sqrt(gmm.covariances_.flatten())
        }
    gmm_sammary[n_comp] = gams

colors = matplotlib.colormaps.get_cmap('tab10')
for n_comp, gams in gmm_sammary.items():
    data_names = list(gams.keys())
    means = np.stack([gams[name]["means"] for name in data_names])
    stds = np.stack([gams[name]["std"] for name in data_names])
    weights = np.stack([gams[name]["weights"] for name in data_names])
    # 均值图
    fig1 = plot_gmm_circle_matrix(means, weights, data_names, xlabel='GMM Component Mean',
                                  title=f'GMM Means (n_components={n_comp})', colors=colors)
    fig1.savefig(gmm_out_dir / f'means_gmm_circles_ncomp{n_comp}.svg')
    plt.show()
    # 标准差图
    fig2 = plot_gmm_circle_matrix(stds, weights, data_names, xlabel='GMM Component Std',
                                  title=f'GMM Stds (n_components={n_comp})', colors=colors)
    fig2.savefig(gmm_out_dir / f'stds_gmm_circles_ncomp{n_comp}.svg')
    plt.show()
    
# %% KL divergence estimation
KL_out_dir = data_root / "kl_js_plots"
KL_out_dir.mkdir(exist_ok=True)
cmap = 'coolwarm'
# bin_width = 1.0
for bin_width in [0.01 ,0.5, 1.0]:
    js_dist = np.zeros((len(all_data), len(all_data)))
    js_div = np.zeros((len(all_data), len(all_data)))
    for i, (key_i, data_i) in enumerate(all_data.items()):
        angles_i = data_i["lifting_angles"]["angles"]
        for j, (key_j, data_j) in enumerate(all_data.items()):
            angles_j = data_j["lifting_angles"]["angles"]
            js_dist[i, j], js_div[i, j] = calculate_angle_js(angles_i, angles_j,bin_width=bin_width)

    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(js_div, cmap=cmap)
    ax.set_xticks(range(len(all_data)))
    ax.set_yticks(range(len(all_data)))
    ax.set_xticklabels(all_data.keys(), rotation=90)
    ax.set_yticklabels(all_data.keys())
    ax.set_title(f"JS Divergence bin {bin_width}")

    # 在每个格子上标注具体数值
    for i in range(js_div.shape[0]):
        for j in range(js_div.shape[1]):
            val = js_div[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color="w", fontsize=12)

    # 增加color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("JS Divergence")

    plt.tight_layout()
    plt.savefig(KL_out_dir / f"lifting_angle_js_divergence_bin_width_{bin_width}.svg")
    plt.show()


# %% other view
# 绘制所有角度随时间变化曲线
all_angles = {}

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

colors = matplotlib.colormaps.get_cmap('tab10')
fig, ax = plt.subplots(figsize=(20, 5))
window = 50  # 平滑窗口
for idx, (name, data) in enumerate(all_data.items()):
    angles = data["lifting_angles"]["angles"]
    all_angles[name] = angles
    color = colors(idx)
    # 绘制平滑后的曲线
    if len(angles) > window:
        smooth = moving_average(angles, window)
        ax.plot(np.arange(window//2, window//2+len(smooth)), smooth, color=color, alpha=0.9, linewidth=2, label=None)
    else:
        ax.plot(np.arange(len(angles)), angles, color=color, alpha=0.9, linewidth=2, label=None)
    # 原始曲线淡化显示
    ax.plot(np.arange(len(angles)), angles, color=color, alpha=0.15, linewidth=1, label=None)
    mean_val = np.mean(angles)
    ax.axhline(mean_val, color=color, linestyle='--', linewidth=1, alpha=0.8)
# 只在图例中标注均值线
handles = [plt.Line2D([0], [0], color=colors(i), linestyle='--', label=f"{name} mean") for i, name in enumerate(all_data.keys())]
ax.legend(handles=handles, loc='upper right', fontsize=10, ncol=2)
ax.set_xlabel("Frame/Index")
ax.set_ylabel("Angle")
ax.set_title("All Lifting Angles Over Time (Smoothed) with Mean Lines")
plt.tight_layout()
plt.savefig(data_root / "all_lifting_angles_over_time.svg")
plt.show()

# status 
# %% pair-wise t test
all_names = list(all_angles.keys())
all_pvalues = np.zeros((len(all_names), len(all_names)))
for i, name_i in enumerate(all_names):
    for j, name_j in enumerate(all_names):
        if i == j:
            all_pvalues[i, j] = 0.0
        else:
            _, pvalue = stats.ttest_ind(all_angles[name_i], all_angles[name_j], equal_var=False, nan_policy='omit')
            all_pvalues[i, j] = pvalue

print("Pairwise p-values for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_pvalues[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

# %% pair-wise Mann-Whitney U test
all_uvalues = np.zeros((len(all_names), len(all_names)))
for i, name_i in enumerate(all_names):
    for j, name_j in enumerate(all_names):
        if i == j:
            all_uvalues[i, j] = 0.0
        else:
            _, pvalue = stats.mannwhitneyu(all_angles[name_i], all_angles[name_j], alternative='two-sided')
            all_uvalues[i, j] = pvalue

print("\nPairwise Mann-Whitney U p-values for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_uvalues[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

# %% all_angles KS test
all_ksvalues = np.zeros((len(all_names), len(all_names)))
for i, name_i in enumerate(all_names):
    for j, name_j in enumerate(all_names):
        if i == j:
            all_ksvalues[i, j] = 0.0
        else:
            _, pvalue = stats.ks_2samp(all_angles[name_i], all_angles[name_j], alternative='two-sided')
            all_ksvalues[i, j] = pvalue

print("\nPairwise KS test p-values for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_ksvalues[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

# %% pair-wise permutation JS divergence test for all_angles

def permutation_test_js(data1, data2, n_permutations=1000, bin_width=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    combined = np.concatenate([data1, data2])
    n1 = len(data1)
    obs_js = calculate_angle_js(data1, data2, bin_width=bin_width)[1]
    perm_stats = np.empty(n_permutations)
    for k in range(n_permutations):
        permuted = rng.permutation(combined)
        perm1 = permuted[:n1]
        perm2 = permuted[n1:]
        perm_stats[k] = calculate_angle_js(perm1, perm2, bin_width=bin_width)[1]
    pvalue = (np.sum(perm_stats >= obs_js) + 1) / (n_permutations + 1)
    return obs_js, pvalue, perm_stats

all_perm_js = np.zeros((len(all_names), len(all_names)))
all_perm_pvalues = np.zeros((len(all_names), len(all_names)))
for i, name_i in enumerate(all_names):
    for j, name_j in enumerate(all_names):
        if i == j:
            all_perm_js[i, j] = 0.0
            all_perm_pvalues[i, j] = 0.0
        else:
            obs_js, pvalue, _ = permutation_test_js(all_angles[name_i], all_angles[name_j], n_permutations=1000, bin_width=1.0, random_state=42)
            all_perm_js[i, j] = obs_js
            all_perm_pvalues[i, j] = pvalue

print("\nPairwise permutation JS divergence for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_perm_js[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

print("\nPairwise permutation JS p-values for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_perm_pvalues[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

# %% pair-wise Anderson-Darling k-sample test
all_ad_statistics = np.zeros((len(all_names), len(all_names)))
all_ad_significance_level = np.zeros((len(all_names), len(all_names)))
for i, name_i in enumerate(all_names):
    for j, name_j in enumerate(all_names):
        if i == j:
            all_ad_statistics[i, j] = 0.0
            all_ad_significance_level[i, j] = 0.0
        else:
            result = stats.anderson_ksamp([all_angles[name_i], all_angles[name_j]])
            all_ad_statistics[i, j] = result.statistic
            all_ad_significance_level[i, j] = result.significance_level

print("\nPairwise Anderson-Darling test statistics for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_ad_statistics[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

print("\nPairwise Anderson-Darling significance levels for all_angles:")
print(all_names)
for i, name_i in enumerate(all_names):
    row = "\t".join(f"{all_ad_significance_level[i, j]:.9e}" for j in range(len(all_names)))
    print(f"{name_i}\t{row}")

# %% merge angles
merge_info = {'n':['1','2'],
              's':['3','4']}
merged_angles = {
    name: np.concatenate([all_angles[key] for key in sub_keys])
    for name, sub_keys in merge_info.items()
}
# %% merged_angles t test
stat, pvalue = stats.ttest_ind(
    merged_angles['n'],
    merged_angles['s'],
    equal_var=False,
    nan_policy='omit'
)
print("\nIndependent t-test for merged_angles (n vs s):")
print(f"statistic={stat:.6f}, pvalue={pvalue:.6e}")

# %% feature  Mann-Whitney U test
stat, pvalue = stats.mannwhitneyu(
    merged_angles['n'],
    merged_angles['s'],
    alternative='two-sided'
)
print("\nMann-Whitney U test for merged_angles (n vs s):")
print(f"statistic={stat:.6f}, pvalue={pvalue:.6e}")

# %% merged_angles violin plot

def significance_label(p):
    if p < 0.0001:
        return '****'
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

sig_text = significance_label(pvalue)
plot_df = pd.DataFrame({
    'group': ['n'] * len(merged_angles['n']) + ['s'] * len(merged_angles['s']),
    'angle': np.concatenate([merged_angles['n'], merged_angles['s']])
})

sns.set_style('whitegrid')
sns.set_palette(['#4c72b0', '#dd8452'])
fig, ax = plt.subplots(figsize=(6, 6))
sns.violinplot(
    data=plot_df,
    x='group',
    y='angle',
    inner='quartile',
    cut=0,
    ax=ax
)

ax.set_xlabel('Group')
ax.set_ylabel('Angle')
ax.set_title('angles')

# annotate p-value with standard significance bar
x1, x2 = 0, 1
y_max = max(plot_df['angle'])
y_span = max(y_max - min(plot_df['angle']), 1e-3)
bar_y = y_max + 0.02 * y_span
text_y = y_max + 0.03 * y_span
ax.plot([x1, x2], [bar_y, bar_y], color='black', linewidth=1)
ax.plot([x1, x1], [bar_y - 0.01 * y_span, bar_y], color='black', linewidth=1)
ax.plot([x2, x2], [bar_y - 0.01 * y_span, bar_y], color='black', linewidth=1)
ax.text((x1 + x2) / 2, text_y, f'{sig_text} p={pvalue:.6e}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(data_root / 'merged_angles_violin.svg')
plt.show()


# %% feature Anderson-Darling k-sample test


## view the distribution of merged angles



# %% feature

def from_angles_to_feature(angles):
    return {
        "mean": np.mean(angles),
        "std": np.std(angles),
        "min": np.min(angles),
        "max": np.max(angles),
        "median": np.median(angles),
        "25%": np.percentile(angles, 25),
        "75%": np.percentile(angles, 75)
    }

all_feature_angles = {name: from_angles_to_feature(angles) for name, angles in all_angles.items()}
all_feature_angles_df = pd.DataFrame(all_feature_angles)
print("\nFeature summary table for all_feature_angles:")
print(all_feature_angles_df)

# %% all_feature_angles statistical tests

friedman_result = stats.friedmanchisquare(
    *(all_feature_angles_df[col].values for col in all_feature_angles_df.columns)
)
print("\nFriedman test for all_feature_angles:")
print(f"statistic={friedman_result.statistic:.6f}, pvalue={friedman_result.pvalue:.6e}")

print("\nPairwise Wilcoxon signed-rank test across features:")
for i in range(len(all_feature_angles_df.columns)):
    for j in range(i + 1, len(all_feature_angles_df.columns)):
        name_i = all_feature_angles_df.columns[i]
        name_j = all_feature_angles_df.columns[j]
        stat, pvalue = stats.wilcoxon(
            all_feature_angles_df[name_i].values,
            all_feature_angles_df[name_j].values,
            alternative='two-sided'
        )
        print(f"{name_i} vs {name_j}: statistic={stat:.6f}, pvalue={pvalue:.6e}")


# %% merged feature angle data

merged_feature_angles = {name: [] for name in merge_info.keys()}
for name, sub_items_keys in merge_info.items():
    for key in sub_items_keys:
        merged_feature_angles[name].append(all_feature_angles[key])

# %% merged_feature_angles comparison
merged_feature_summary = {}
for name, feature_list in merged_feature_angles.items():
    merged_feature_summary[name] = {
        feat: np.mean([item[feat] for item in feature_list])
        for feat in feature_list[0].keys()
    }
merged_feature_angles_df = pd.DataFrame(merged_feature_summary)

# paired Wilcoxon signed-rank test for merged groups n vs s
merged_features = merged_feature_angles_df.columns.tolist()
stat, pvalue = stats.wilcoxon(
    merged_feature_angles_df['n'].values,
    merged_feature_angles_df['s'].values,
    alternative='two-sided'
)
print("\nWilcoxon signed-rank test for merged_feature_angles (n vs s):")
print(f"statistic={stat:.6f}, pvalue={pvalue:.6e}")

# optional paired t-test for reference
t_stat, t_pvalue = stats.ttest_rel(
    merged_feature_angles_df['n'].values,
    merged_feature_angles_df['s'].values
)
print("Paired t-test for merged_feature_angles (n vs s):")
print(f"statistic={t_stat:.6f}, pvalue={t_pvalue:.6e}")

# %% save results here