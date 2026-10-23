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
from adjustText import adjust_text

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


# %% plot Bandwidth

# 定义目标变量顺序
targets_list = ['Average', 'Sedan', 'Bus', 'Logistic']

palette = sns.color_palette('tab10', n_colors=len(targets_list))
target_colors = {target: palette[i] for i, target in enumerate(targets_list)}


target_colors = {target: palette[i] for i, target in enumerate(targets_list)}
# 排除截距项
combined_df = all_results_df[all_results_df['Variable'] != 'Intercept'].copy()
valid_plot = np.isfinite(combined_df[['Bandwidth', 'Spatial Variability', 'Mean t-value']]).all(axis=1)
print(f'Rows with nonfinite plotted values omitted: {(~valid_plot).sum()}')
combined_df = combined_df.loc[valid_plot].copy()

# 计算变量重要性（使用Mean t-value的绝对值）
combined_df['Importance'] = combined_df['Mean t-value'].abs()

# 创建图形
plt.figure(figsize=(12, 10))

# 创建气泡图 - 大小表示重要性，颜色表示目标变量
scatter = sns.scatterplot(
    data=combined_df, 
    x='Bandwidth', 
    y='Spatial Variability', 
    hue='Target', 
    hue_order=targets_list,  # 设置目标变量顺序
    size='Importance',
    sizes=(500, 5000),  # 调整气泡大小范围
    alpha=0.5,
    palette='tab10',
    legend=False  # 禁用自动图例
)

# 添加参考线
median_bw = np.median(combined_df['Bandwidth'])
plt.axvline(x=median_bw, color='gray', linestyle='--', alpha=0.7)
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)

plt.text(median_bw - 20, 5, f'Median BW: {median_bw:.0f}', 
         fontsize=25, ha='left', va='top', color='black')
plt.text(40, 0, 'SV = 0.5', 
         fontsize=25, ha='right', va='bottom', color='black')

# 添加趋势线
sns.regplot(
    data=combined_df, 
    x='Bandwidth', 
    y='Spatial Variability', 
    scatter=False, 
    ci=None, 
    color='black', 
    line_kws={'alpha':0.5, 'linestyle': '--'}
)

# 为每个目标变量选择最重要的特征
top_vars_by_target = []
for target in targets_list:
    # 筛选当前目标变量的数据
    target_df = combined_df[combined_df['Target'] == target]
    
    # 如果该目标变量有数据，选择重要性最高的特征
    if not target_df.empty:
        top_var = target_df.nlargest(1, 'Importance')
        top_vars_by_target.append(top_var)

# 合并所有选出的特征
top_vars = pd.concat(top_vars_by_target)

# 添加变量标签（使用adjustText自动调整位置）
texts = []  # 存储文本对象
for i, row in top_vars.iterrows():
    # 初始位置（稍作偏移）
    offset_x = 10 if row['Bandwidth'] < median_bw else -10
    offset_y = 0.2 if row['Spatial Variability'] < 0.5 else -0.2
    
    # 获取对应目标变量的颜色
    text_color = target_colors.get(row['Target'], 'black')
    
    t = plt.text(
        row['Bandwidth'] + offset_x, 
        row['Spatial Variability'] + offset_y, 
        row['Variable'],
        fontsize=25, 
        color=text_color,
        ha='center' if offset_x < 0 else 'left',
        va='center',
        fontweight='bold',
    )
    texts.append(t)

# 自动调整文本位置以避免重叠
adjust_text(texts, 

            expand_points=(1.2, 1.5), 
            expand_text=(1.2, 1.5),
            force_text=(0.5, 0.5))

# 设置轴标签
plt.xlabel('Bandwidth', fontsize=25)
plt.ylabel('Coefficient SD / |mean coefficient|', fontsize=25)

# 添加网格
plt.grid(True, alpha=0.3)

# 创建目标变量图例


# 定义目标变量简称映射
target_short_names = {
    'Average': 'Average',
    'Sedan': 'Sedan',
    'Bus': 'Bus',
    'Logistic': 'Logistic'
}

# 创建图例元素
legend_elements = [
    Line2D([0], [0], 
           marker='o', 
           color='w', 
           markerfacecolor=target_colors[target], 
           markersize=20,
           label=target_short_names[target])
    for target in targets_list
]

# 添加图例
plt.legend(
    handles=legend_elements,
    fontsize=30,
    bbox_to_anchor=(0.65, 1),  # 将图例放在图形外部右侧
    loc='upper left',
    frameon=True,
    framealpha=0.8
)

# 设置刻度字体大小
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# 设置坐标轴范围
# Use data-driven axes: do not hide supplied values outside fixed bounds.
plt.margins(x=0.1, y=0.1)
# 调整布局
plt.tight_layout()

plt.title('Bubble area represents |mean t-statistic|; coefficient ratio is descriptive', fontsize=12)
# 保存图形
plt.savefig(output_path('figures/Bandwidth.svg'), dpi=600)
plt.show()
# %%
