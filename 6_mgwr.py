# %%
import pandas as pd
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mgwr.gwr import MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import warnings
import os
import pickle

# 忽略警告
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-whitegrid')
sns.set(style="whitegrid", palette="pastel", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'  # 确保中文字符显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ======================
# 1. 数据加载与预处理
# ======================
file_path = "/home/ps/haichao/6-BEEC/Data/urban_statistic.csv"
shp_path = "/home/ps/haichao/6-BEEC/shp2/乡镇级数据.shp"

# 列名映射（与OLS保持一致）
COLUMN_MAPPING = {
    'urban_ec_by_vehicle_type_Total_Energy_Rate': 'Average',
    'urban_ec_by_vehicle_type_Sedan_Energy_Rate': 'Sedan',
    'urban_ec_by_vehicle_type_Bus_Energy_Rate': 'Bus',
    'urban_ec_by_vehicle_type_Logistic_Energy_Rate': 'Logistic',
    'urban_building_Building Density': 'Building density',
    'urban_poi_shannon_entropy': 'POI diversity',
    'urban_landuse_landuse_entropy': 'Land use diversity',
    'urban_charge_total_charging_stations': 'Charging station',
    'urban_bus_bus_stop_count': 'Bus stop',
    'urban_road_road_density': 'Road density',
    'urban_road_intersection_density': 'Intersection density',
    'urban_road_avg_road_curvature': 'Road curvature',
    'urban_road_arterial_ratio': 'Arterial ratio',
    'urban_road_connectivity_index': 'Connectivity',
    'urban_slopes_avg_slope': 'Slope',
    'urban_elevation_Elevation_Range': 'Elevation',
    'urban_population_population2020': 'Population'
}

# 一次性读取并处理数据
df = pd.read_csv(file_path).rename(columns=COLUMN_MAPPING)
gdf = gpd.read_file(shp_path).merge(df, on='NAME', how='inner')

# 定义变量（与OLS保持一致）
targets = ['Average', 'Sedan', 'Bus', 'Logistic']
predictors = [
    'Building density', 'POI diversity', 'Land use diversity',
    'Charging station', 'Bus stop', 'Road density',
    'Road curvature', 'Arterial ratio',
    'Connectivity', 'Slope', 'Elevation', 'Population'
]

# 坐标系转换（确保使用投影坐标系）
if gdf.crs is None or gdf.crs.to_epsg() == 4326:
    # 转换为UTM坐标系（根据你的研究区域调整）
    gdf = gdf.to_crs(epsg=32650)  # 中国东部使用32650
    print("坐标系已转换为UTM (EPSG:32650)")

# 计算质心坐标
gdf['x'] = gdf.geometry.centroid.x
gdf['y'] = gdf.geometry.centroid.y
#%%
# 创建字典存储所有结果
mgwr_results = {}
all_results = []

# ======================
# 2. 缺失值处理
# ======================
print("\n缺失值分析:")
missing_analysis = []
for target in targets:
    # 将0值替换为NaN（假设0代表缺失）
    gdf.loc[gdf[target] == 0, target] = np.nan
    
    # 计算缺失值统计
    missing_count = gdf[target].isna().sum()
    missing_percent = missing_count / len(gdf) * 100
    missing_analysis.append({
        'Target': target,
        'Missing_Regions': missing_count,
        'Missing_Percent': missing_percent
    })
    print(f"{target}: {missing_count} 个缺失区域 ({missing_percent:.1f}%)")

# 保存缺失值分析结果
missing_df = pd.DataFrame(missing_analysis)
#%%
# ======================
# 3. 模型拟合循环
# ======================
for target in targets:
    print(f"\n{'='*80}")
    print(f"MGWR模型拟合: {target}")
    print(f"{'='*80}")
    
    # 创建有效样本掩码
    valid_mask = ~gdf[target].isna()
    valid_gdf = gdf[valid_mask].copy()
    n_valid = len(valid_gdf)
    print(f"有效样本量: {n_valid} (总样本量: {len(gdf)}, 缺失: {len(gdf) - n_valid})")
    
    # 提取有效样本的坐标
    valid_coords = np.array(list(zip(valid_gdf['x'], valid_gdf['y'])))
    
    # 提取目标变量
    y = valid_gdf[target].values.reshape(-1, 1)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(valid_gdf[predictors])
    
    # ======================
    # 4. 创建空间权重矩阵
    # ======================
    nbrs = NearestNeighbors(n_neighbors=2).fit(valid_coords)
    distances, _ = nbrs.kneighbors(valid_coords)
    avg_min_dist = np.median(distances[:, 1])
    threshold = 1 * avg_min_dist
    
    print(f"平均最近邻距离: {avg_min_dist:.2f} 米")
    print(f"使用距离阈值: {threshold:.2f} 米")
    
    # ======================
    # 5. MGWR模型拟合
    # ======================
    # 选择最优带宽
    print("\n开始带宽选择... (可能需要较长时间)")
    selector = Sel_BW(valid_coords, y, X_scaled, multi=True)
    
    # 设置最小带宽（每个变量至少需要2个邻居）
    min_bw = [2] * (X_scaled.shape[1] + 1)  # +1 包括截距项
    
    # 搜索最优带宽
    try:
        bw = selector.search(multi_bw_min=min_bw, verbose=True)
        print(f"最优带宽向量: {bw}")
    except Exception as e:
        print(f"带宽选择错误: {e}")
        print("使用默认带宽选择方法")
        bw = selector.search(verbose=True)
    
    # 拟合MGWR模型
    print("\n拟合MGWR模型...")
    mgwr_model = MGWR(valid_coords, y, X_scaled, selector=selector)
    result = mgwr_model.fit()
    mgwr_results[target] = result
    
    # 打印模型摘要
    print(f"\n{target} MGWR模型摘要:")
    # print(result.summary())
    
    # ======================
    # 6. 结果提取
    # ======================
    # 提取结果
    params = result.params  # 系数
    bse = result.bse        # 标准误
    t_vals = result.tvalues # t值
    
    # 安全计算p值
    def safe_pvalue_calculation(t_vals, n, k):
        """安全计算p值，处理极端t值"""
        p_vals = np.zeros_like(t_vals)
        # 计算全局自由度 (n - k - 1)
        df_global = max(1, n - k - 1)  # 确保自由度至少为1
        for i in range(t_vals.shape[0]):
            for j in range(t_vals.shape[1]):
                t_val = t_vals[i, j]
                # 处理极端t值
                if np.isnan(t_val) or np.isinf(t_val):
                    p_vals[i, j] = np.nan
                elif abs(t_val) > 100:  # 非常大的t值
                    p_vals[i, j] = 0.0 if t_val != 0 else 1.0
                else:
                    # 双尾检验
                    p_vals[i, j] = 2 * (1 - stats.t.cdf(abs(t_val), df_global))
        return p_vals
    
    # 手动计算R²和调整R²
    def calculate_r2_adjr2(result):
        """手动计算R²和调整R²"""
        # 获取必要数据
        y_true = result.y.reshape(-1,1)  # 实际值
        y_pred = result.predy.reshape(-1,1)  # 预测值
        n = result.n  # 样本量
        k = result.k  # 变量数（包括截距）
        
        # 计算残差平方和 (SS_res)
        ss_res = np.sum((y_true - y_pred) ** 2)
        
        # 计算总平方和 (SS_tot)
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        
        # 计算R²
        r2 = 1 - (ss_res / ss_tot)
        
        # 计算调整R²
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k)
        
        return r2, adj_r2
    
    # 使用手动计算（使用有效样本量）
    k = result.k  # 变量数（包括截距）
    p_vals = safe_pvalue_calculation(t_vals, n_valid, k)
    # 保存回结果对象
    result.pvalues = p_vals
    
    # 计算显著比例（使用有效样本量）
    percent_significant = np.sum(p_vals < 0.05, axis=0) / n_valid * 100
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Variable': ['Intercept'] + predictors,
        'Bandwidth': bw,
        'Mean Coefficient': np.mean(params, axis=0),
        'Coefficient Std': np.std(params, axis=0),
        'Min Coefficient': np.min(params, axis=0),
        'Max Coefficient': np.max(params, axis=0),
        'Mean Std Error': np.mean(bse, axis=0),
        'Mean t-value': np.mean(t_vals, axis=0),
        'Significant (%)': percent_significant,
        'Spatial Variability': np.std(params, axis=0) / np.abs(np.mean(params, axis=0)),
        # 添加样本量信息
        'Valid Regions': n_valid,
        'Total Regions': len(gdf),
        'Missing Regions': len(gdf) - n_valid
    })
    
    # 添加模型评估指标
    results_df['AICc'] = result.aicc
    r2, adj_r2 = calculate_r2_adjr2(result)
    results_df['R²'] = r2
    results_df['Adjusted R²'] = adj_r2
    results_df['Target'] = target
    
    # 存储结果
    all_results.append(results_df)
    print(f"已完成 {target} 模型拟合")
#%%
# ======================
# 7. 结果保存
# ======================
# 创建结果目录
output_dir = "/home/ps/haichao/6-BEEC/results"
os.makedirs(output_dir, exist_ok=True)

# 1. 保存模型结果
with open(os.path.join(output_dir, 'mgwr_results.pkl'), 'wb') as f:
    pickle.dump(mgwr_results, f)

# 2. 保存汇总结果
all_results_df = pd.concat(all_results, ignore_index=True)
all_results_df.to_csv(os.path.join(output_dir, 'mgwr_summary_results.csv'), index=False, encoding='utf-8-sig')

# 3. 保存空间系数数据
coef_data = {}
for target in targets:
    result = mgwr_results[target]
    params = result.params  # 系数矩阵
    
    # 创建用于绘图的数据结构
    coef_data[target] = {
        'params': params,
        'predictors': predictors,
        'valid_mask': ~gdf[target].isna()  # 有效样本掩码
    }

# # 保存空间系数数据
with open(os.path.join(output_dir, 'spatial_coefficients.pkl'), 'wb') as f:
    pickle.dump(coef_data, f)

# 4. 保存地理数据（包含几何信息）
gdf.to_file(os.path.join(output_dir, 'all_data.shp'), encoding='utf-8')

print(f"\n所有结果数据已保存到 {output_dir}")
