# Candidate copy; see README and audit change log.
from release_paths import INPUT_ROOT, output_path
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.patches import Patch
from matplotlib import rcParams
import geopandas as gpd
import seaborn as sns
from matplotlib.lines import Line2D

rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 22,
    'axes.titlesize': 24,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 20,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.alpha': 0.4
})
# 设置结果目录

output_dir = str(INPUT_ROOT / 'results')

# 加载地理数据
gdf = gpd.read_file(os.path.join(output_dir, 'all_data.shp'))

# 加载汇总结果
all_results_df = pd.read_csv(os.path.join(output_dir, 'mgwr_summary_results.csv'))

# 加载空间系数数据
with open(os.path.join(output_dir, 'spatial_coefficients.pkl'), 'rb') as f:
    coef_data = pickle.load(f)


# %%   plot coefficient distribution
# 目标变量列表
targets_list = ['Average', 'Sedan', 'Bus', 'Logistic']

# 选择目标变量（这里以Logistic为例）
target_name = os.environ.get("BEEC_VEHICLE", "Logistic")
target_data = coef_data[target_name]

# 提取数据
params = target_data['params']  # 系数矩阵 (n_valid, n_predictors+1)
predictors = target_data['predictors']  # 预测变量名称列表
valid_mask = target_data['valid_mask']  # 有效样本掩码

# 创建绘图地理数据框
plot_gdf = gdf.copy()

# 为所有系数列初始化为0（跳过截距项）
for var_name in predictors:  # predictors 只包含预测变量名称，不包括截距
    plot_gdf[f'coef_{var_name}'] = np.nan

# 将系数分配给有效样本
valid_mask = np.asarray(valid_mask, dtype=bool)
if len(valid_mask) != len(gdf) or params.shape != (int(valid_mask.sum()), len(predictors) + 1):
    raise ValueError('Coefficient rows/mask do not match the supplied geometry order.')
valid_indices = gdf.index[valid_mask]
for i, var_name in enumerate(predictors):
    # params 第一列是截距，所以预测变量从索引1开始
    plot_gdf.loc[valid_indices, f'coef_{var_name}'] = params[:, i+1]

# 定义颜色映射
color_map = {
    'Strong Negative': '#2166ac',  # 深蓝
    'Negative': '#67a9cf',        # 中蓝
    'Weak Negative': '#d1e5f0',   # 浅蓝
    'Weak Positive': '#fddbc7',   # 浅红
    'Positive': '#ef8a62',        # 中红
    'Strong Positive': '#b2182b', # 深红
    'Zero': '#ffffff',
    'Missing': 'lightgrey'        # 浅灰
}

# 创建3行4列的图形
fig, axes = plt.subplots(3, 4, figsize=(20, 15), gridspec_kw={'wspace': 0.01, 'hspace': 0.1})
axes = axes.flatten()  # 展平为1D数组以便索引

# 获取所有变量名称
variable_names = predictors  # 直接使用预测变量名称列表

# 定义变量名称映射（中文到英文）
en_variable_names = {
    'Building density': 'Building density',
    'POI diversity': 'POI diversity',
    'Land use diversity': 'Land use mix',
    'Charging station': 'Charging station',
    'Bus stop': 'Bus stop',
    'Road density': 'Road density',
    'Road curvature': 'Road curvature',
    'Arterial ratio': 'Arterial ratio',
    'Connectivity': 'Connectivity',
    'Slope': 'Slope',
    'Elevation': 'Elevation',
    'Population': 'Population'
}

# 处理每个变量
for idx, var_key in enumerate(variable_names):
    ax = axes[idx]
    
    # 获取当前变量的值
    values = plot_gdf[f'coef_{var_key}']
    valid_values = values[valid_indices]
    
    # 分离正负系数
    positive_values = valid_values[valid_values > 0]
    negative_values = valid_values[valid_values < 0]
    
    # 计算正系数的分位数
    if len(positive_values) > 0:
        q1_pos = np.quantile(positive_values, 0.25)
        q2_pos = np.quantile(positive_values, 0.75)
    else:
        q1_pos = q2_pos = 0
    
    # 计算负系数的分位数
    if len(negative_values) > 0:
        q1_neg = np.quantile(negative_values, 0.25)
        q2_neg = np.quantile(negative_values, 0.75)
    else:
        q1_neg = q2_neg = 0
    
    # 创建分类条件
    conditions = [
        values < q1_neg,  # 强负
        (values >= q1_neg) & (values < q2_neg),  # 中负
        (values >= q2_neg) & (values < 0),  # 弱负
        (values > 0) & (values <= q1_pos),  # 弱正
        (values > q1_pos) & (values <= q2_pos),  # 中正
        values > q2_pos  # 强正
    ]
    
    labels = [
        'Strong Negative',
        'Negative',
        'Weak Negative',
        'Weak Positive',
        'Positive',
        'Strong Positive'
    ]
    
    # 应用分类
    current_cat = np.select(conditions, labels, default='Missing')
    current_cat[(values == 0) & values.notna()] = 'Zero'
    
    # 绘制每个类别
    for category in labels + ['Zero', 'Missing']:
        mask = (current_cat == category)
        if mask.any():  # 仅当该类别有要素时才绘制
            subset = plot_gdf[mask]
            color = color_map[category]
            subset.plot(ax=ax, color=color, edgecolor='lightgrey', linewidth=0.1)
    
    # 使用英文变量名称设置子图标题
    ax.set_title(en_variable_names.get(var_key, var_key), fontsize=25)
    ax.set_axis_off()

# 隐藏多余的轴（如果有）
for i in range(len(variable_names), len(axes)):
    axes[i].axis('off')

# 创建共享图例
legend_elements = [
    Patch(facecolor='#2166ac', label='Strong Negative'),
    Patch(facecolor='#67a9cf', label='Negative'),
    Patch(facecolor='#d1e5f0', label='Weak Negative'),
    Patch(facecolor='#fddbc7', label='Weak Positive'),
    Patch(facecolor='#ef8a62', label='Positive'),
    Patch(facecolor='#b2182b', label='Strong Positive'),
    Patch(facecolor='white', edgecolor='grey', label='Zero'),
    Patch(facecolor='lightgrey', label='Missing')
]

# 在底部添加共享图例
fig.legend(handles=legend_elements,
           loc='upper center',
           ncol=8,
           bbox_to_anchor=(0.5, 0.1),
           fontsize=16)

# 调整布局
plt.tight_layout()
plt.subplots_adjust(bottom=0.1)  # 为标题和图例留出空间

# 保存并显示
plt.savefig(output_path(f'figures/coefficients_{target_name}.png'), dpi=600)
plt.show()
# %%
