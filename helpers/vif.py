#%%
import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

#%% ols

# 1. 数据加载与预处理
file_path = "/home/ps/haichao/6-BEEC/Data/urban_statistic.csv"
shp_path = "/home/ps/haichao/6-BEEC/shp2/乡镇级数据.shp"

# 读取数据
df = pd.read_csv(file_path)
gdf = gpd.read_file(shp_path).merge(df, on='NAME')


# 读取数据并重命名列
df = pd.read_csv(file_path).rename(columns={
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
})

gdf = gpd.read_file(shp_path).merge(df, on='NAME')

# 定义目标变量和预测变量
targets = {
    'Average': 'Average',
    'Sedan': 'Sedan',
    'Bus': 'Bus',
    'Logistic': 'Logistic'
}

predictors = {
    # Built Environment
    'Building density': 'Building density',
    'POI diversity': 'POI diversity',
    'Land use diversity': 'Land use diversity',
    'Charging station': 'Charging station',
    'Bus stop': 'Bus stop',
    # Road network design
    'Road density': 'Road density',
    'Intersection density': 'Intersection density',
    'Road curvature': 'Road curvature',
    'Arterial ratio': 'Arterial ratio',
    'Connectivity': 'Connectivity',
    'Slope': 'Slope',
    # Socio-spatial characteristics
    'Elevation': 'Elevation',
    'Population': 'Population'
}


# 数据清洗
data = gdf[['geometry'] + list(targets.keys()) + list(predictors.keys())].dropna()

# 坐标系转换（如果未投影）
if data.crs is None or data.crs.to_epsg() == 4326:
    data = data.to_crs(epsg=32650)
    print("转换后CRS:", data.crs)


#%% VIF test
# 提取预测变量数据
X = data[list(predictors.keys())]

# 标准化预测变量（VIF计算不需要标准化，但有助于解释）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# 添加常数项用于VIF计算
X_scaled_const = sm.add_constant(X_scaled)

# 计算每个预测变量的VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_scaled_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled_const.values, i) 
                   for i in range(X_scaled_const.shape[1])]

# 移除常数项的行
vif_data = vif_data[vif_data["Variable"] != "const"]

# 按VIF值降序排序
vif_data = vif_data.sort_values(by="VIF", ascending=False)

# 打印结果
print("多重共线性分析结果 (VIF值):")
print("--------------------------------")
print(vif_data)
print("\nVIF解释:")
print("VIF < 5: 可接受的共线性")
print("5 ≤ VIF < 10: 中等共线性")
print("VIF ≥ 10: 严重共线性")
#%%
# 1. 创建新的预测变量字典（移除Intersection_density）
predictors_reduced = predictors.copy()
del predictors_reduced['Intersection density']  # 移除VIF最高的变量

print("\n移除高VIF变量(Intersection density)后重新计算VIF...")
print(f"剩余变量数: {len(predictors_reduced)}")

# 2. 提取新的预测变量集
X_reduced = data[list(predictors_reduced.keys())]

# 3. 标准化变量
scaler = StandardScaler()
X_reduced_scaled = pd.DataFrame(scaler.fit_transform(X_reduced), columns=X_reduced.columns)

# 4. 添加常数项
X_reduced_scaled_const = sm.add_constant(X_reduced_scaled)

# 5. 计算新VIF
vif_reduced = pd.DataFrame()
vif_reduced["Variable"] = X_reduced_scaled_const.columns
vif_reduced["VIF"] = [variance_inflation_factor(X_reduced_scaled_const.values, i) 
                      for i in range(X_reduced_scaled_const.shape[1])]

# 6. 移除常数项并排序
vif_reduced = vif_reduced[vif_reduced["Variable"] != "const"]
vif_reduced = vif_reduced.sort_values(by="VIF", ascending=False)

# 7. 打印结果
print("\n移除交叉口密度后的多重共线性分析结果:")
print("--------------------------------------")
print(vif_reduced)

# 8. 检查是否仍有高VIF变量
high_vif_reduced = vif_reduced[vif_reduced["VIF"] >= 5]
if not high_vif_reduced.empty:
    print("\n警告: 以下变量仍存在中等到严重的多重共线性:")
    for _, row in high_vif_reduced.iterrows():
        print(f"- {row['Variable']}: VIF = {row['VIF']:.2f}")
else:
    print("\n所有变量的VIF均低于5，多重共线性问题已解决")
    
# 9. 比较处理前后的VIF变化
vif_comparison = pd.merge(vif_data, vif_reduced, on="Variable", suffixes=('_原始', '_处理后'), how='right')
vif_comparison['VIF变化'] = vif_comparison['VIF_处理后'] - vif_comparison['VIF_原始']
vif_comparison['VIF变化百分比'] = (vif_comparison['VIF变化'] / vif_comparison['VIF_原始']) * 100

print("\nVIF变化对比分析:")
print("-----------------")
print(vif_comparison[['Variable', 'VIF_原始', 'VIF_处理后', 'VIF变化', 'VIF变化百分比']])
# %%
