# Candidate copy; see README and audit change log.
from release_paths import INPUT_ROOT, output_path
# %%

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
# Plot only the supplied summary CSV; no geometry input is used.

# 加载汇总结果
all_results_df = pd.read_csv(os.path.join(output_dir, 'mgwr_summary_results.csv'))

# 加载空间系数数据
# Unused coefficient pickle read removed.

# 定义目标变量顺序
targets = ['Average', 'Sedan', 'Bus', 'Logistic']

# 创建结果列表（顺序与targets一致）
all_results = []
for target in targets:
    # 从汇总结果中提取当前目标变量的结果
    target_df = all_results_df[all_results_df['Target'] == target]
    all_results.append(target_df)

# 定义变量名称映射（中文到英文缩写）
en_variable_names = {
    'Building density': 'BD',
    'POI diversity': 'POI',
    'Land use diversity': 'LUD',
    'Charging station': 'CS',
    'Bus stop': 'BS',
    'Road density': 'RD',
    'Road curvature': 'RC',
    'Arterial ratio': 'AR',
    'Connectivity': 'Conn',
    'Slope': 'Slope',
    'Elevation': 'Elev',
    'Population': 'Pop'
}

# 预测变量列表（确保顺序一致）
predictor_cols = list(en_variable_names.keys())
#%%  radius importance plot
# 设置图形
fig = plt.figure(figsize=(30, 8))  # 增加高度以容纳图例
gs = fig.add_gridspec(2, 4, height_ratios=[8, 1], wspace=0.25, hspace=0.1)  # 两行网格

# 计算雷达图角度
num_vars = len(predictor_cols)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # 闭合雷达图

# 配色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
row_labels = ['(a)', '(b)', '(c)', '(d)']
target_labels = ['Average', 'Sedan', 'Bus', 'Logistic vehicle']

# 为每个目标变量绘制雷达图
for idx, target in enumerate(targets):
    ax = fig.add_subplot(gs[0, idx], polar=True)
    results_df = all_results[idx]  # 当前目标变量的结果DataFrame
    
    # 提取变量重要性（使用平均t值的绝对值）
    importance = []
    variables = []
    
    # 遍历所有预测变量
    for var in predictor_cols:
        # 从结果中提取当前变量的平均t值
        var_row = results_df[results_df['Variable'] == var]
        if not var_row.empty:
            mean_tvalue = var_row['Mean t-value'].values[0]
            importance.append(abs(mean_tvalue))
            variables.append(en_variable_names[var])
        else:
            # 如果变量不存在，添加0值
            raise ValueError(f'Missing summary variable {var} for {target}; do not plot an invented zero.')
            variables.append(en_variable_names[var])
    
    # 归一化重要性（0-1范围）
    importance = np.array(importance)
    if np.max(importance) > 0:
        importance = importance / np.max(importance)
    
    # 准备雷达图数据（闭合）
    values = importance.tolist()
    values += values[:1]
    
    # 绘制雷达图
    ax.plot(angles, values, color=colors[idx], linewidth=2.5, 
            marker='o', markersize=8, markerfacecolor='white', 
            markeredgecolor=colors[idx], markeredgewidth=2)
    ax.fill(angles, values, color=colors[idx], alpha=0.15)
    
    
    # # 在雷达点上添加实际值
    # for i, (angle, val) in enumerate(zip(angles[:-1], importance)):
    #     value_text = f"{val:.2f}"
    #     ax.text(angle, val + 0.08, value_text, 
    #             ha='center', va='center', fontsize=18,
    #             bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor='none'))
        
    # 设置角度和标签
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(variables, fontsize=30)
    
    # 设置径向轴
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0', '0.25', '0.5', '0.75', '1.0'], fontsize=16)
    ax.set_rlabel_position(30)
    
    # 添加网格和边框
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(1.5)
    
    # 添加标题
    title = f"{row_labels[idx]} {target_labels[idx]}"
    ax.set_title(title, fontsize=40, pad=20, fontweight='bold')
    
    

# 创建图例轴
legend_ax = fig.add_subplot(gs[1, :])
legend_ax.axis('off')  # 关闭坐标轴

# 创建图例元素
legend_elements = []
for full_name, abbr in en_variable_names.items():
    # 创建图例项：缩写 + 全称
    legend_elements.append(Patch(facecolor='none', edgecolor='none', 
                                label=f"{abbr}: {full_name}"))

# 添加图例（水平排列，居中）
legend = legend_ax.legend(handles=legend_elements, 
                          loc='center', 
                          ncol=6,  # 4列布局
                          frameon=False,
                          fontsize=25,
                          handlelength=0,  # 隐藏图例句柄
                          handletextpad=0)  # 减少文本间距

fig.suptitle('Normalized absolute mean t-statistic (within each vehicle category)', fontsize=18)
# 调整布局
plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.08)  # 为图例留出空间
# 调整布局并保存
plt.savefig(output_path('figures/importance.svg'), dpi=600)
plt.show()




# %%
