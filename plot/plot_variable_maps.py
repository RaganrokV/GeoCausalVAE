
from release_paths import INPUT_ROOT, output_path
#%%
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd 
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from matplotlib import rcParams
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 10,
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
# 加载数据
town_path = str(INPUT_ROOT / 'shp2/乡镇级数据.shp')
boundry_path = str(INPUT_ROOT / 'shp2/310000.shp')
file_path = str(INPUT_ROOT / 'Data/urban_statistic.csv')

# 读取地理数据
# Unused city-wide boundary read removed.
township = gpd.read_file(town_path)

# 读取统计数据
df = pd.read_csv(file_path)

# 选择12个变量
selected_vars = [
    'urban_ec_by_vehicle_type_Total_Energy_Rate',
    'urban_building_Building Density',
    'urban_poi_shannon_entropy',
    'urban_landuse_landuse_entropy',
    'urban_charge_total_charging_stations',
    'urban_bus_bus_stop_count',
    'urban_road_road_density',
    'urban_road_avg_road_curvature',
    'urban_road_arterial_ratio',
    'urban_road_connectivity_index',
    'urban_slopes_avg_slope',   
    'urban_population_population2020',
]

# 定义变量映射（英文标签）
variable_labels = {
    'urban_ec_by_vehicle_type_Total_Energy_Rate': ' Energy consumption rate',
    'urban_building_Building Density': 'Building density',
    'urban_poi_shannon_entropy': 'POI diversity',
    'urban_landuse_landuse_entropy': 'Land use diversity',
    'urban_charge_total_charging_stations': 'Charging station',
    'urban_bus_bus_stop_count': 'Bus stop',
    'urban_road_road_density': 'Road density',
    'urban_road_avg_road_curvature':'Road curvature',
    'urban_road_arterial_ratio': 'Arterial ratio',
    'urban_road_connectivity_index': 'Connectivity',
    'urban_slopes_avg_slope': 'Slope',
    'urban_population_population2020': 'Population'
}


#%%
# 假设df中有一个'community_id'列与township中的'id'列匹配
merged_gdf = township.merge(df, left_on='NAME', right_on='NAME', how='inner')

# 创建3x3子图
fig, axes = plt.subplots(3, 4, figsize=(18, 16))
axes = axes.flatten()  # 将2D轴数组展平为1D
plt.rcParams['font.family'] = 'Times New Roman'
# 设置红蓝渐变色彩方案
cmap = plt.get_cmap('RdBu_r')  # 反转的RdBu，使红色表示高值RdBu_r

# 为每个变量创建可视化
for i, var in enumerate(selected_vars):
    if var not in merged_gdf.columns:
        print(f"Warning: Variable {var} not found in merged data. Skipping.")
        continue
    
    ax = axes[i]
    
    # 设置标题
    title = variable_labels[var]
    
    # 计算数据范围（使用分位数避免极端值）
    data = merged_gdf[var].replace([float('inf'), -float('inf')], float('nan')).dropna()
    if data.empty:
        print(f"Warning: No valid data for {var}. Skipping.")
        continue
    
    # 使用分位数设置范围
    vmin = data.quantile(0.05)
    vmax = data.quantile(0.95)
    
    # 特殊处理某些变量
    if 'Density' in title or 'Ratio' in title or 'Index' in title:
        vmin = max(0, data.min())
        vmax = data.max()
    # 绘制地图
    plot = merged_gdf.plot(
        column=var,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor='none',
        linewidth=0.1,
        ax=ax,
        legend=False
    )
    
    # 添加上海市边界
    # gpd.GeoSeries([shanghai_boundary]).boundary.plot(
    #     ax=ax, color='black', linewidth=0.5, alpha=0.7
    # )
    
    # 设置图标题
    ax.set_title(title, fontsize=22, fontweight='bold', pad=5)
    
    # 移除坐标轴
    ax.set_axis_off()
    
    # # 添加指北针
    ax.annotate('N', xy=(0.95, 0.95), xycoords='axes fraction',
                fontsize=15, fontweight='bold', ha='center', va='center')
    
    # 添加颜色条
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []  # 空数组用于颜色条
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', extend='both')


    # 设置颜色条刻度格式
    if var == 'urban_population_population2020':
        cbar.formatter = ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}K' if x >= 1000 else f'{x:.0f}')
        cbar.update_ticks()
    elif var in ['urban_charge_charging_station_count', 'urban_bus_bus_stop_count']:
        cbar.formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.0f}')
        cbar.update_ticks()


# 调整布局
# plt.tight_layout(h_pad=0.01, w_pad=0.01)  # 为标题和脚注留出空间
plt.subplots_adjust(wspace=0.1, hspace=0.12)  # 大幅减小间距

plt.savefig(output_path('figures/variable_maps.png'), dpi=600)
plt.show()

# %%
