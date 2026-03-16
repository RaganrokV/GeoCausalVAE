#%%
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
from func_getdata import prepare_spatial_data
from func_model import SpatialUrbanCausalVAE
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 创建保存结果的目录
model_dir = "/home/ps/haichao/6-BEEC/models/"
results_dir = "/home/ps/haichao/6-BEEC/results/"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

#%%
# ===================================================
# 主程序
# ===================================================
# 定义文件路径
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

# 选择目标变量

target_col = targets[3]
print(target_col)
# 关键修改：处理缺失值并创建有效样本掩码
# 将所有目标变量中的0值替换为NaN
for target in targets:
    gdf.loc[gdf[target] == 0, target] = np.nan

# 创建有效样本掩码（目标变量非缺失）
valid_mask = ~gdf[target_col].isna()
valid_gdf = gdf[valid_mask].copy()
# 打印有效样本数量
print(f"总样本数: {len(gdf)}")
print(f"有效样本数 (非缺失): {len(valid_gdf)}")
print(f"缺失样本数: {len(gdf) - len(valid_gdf)}")
print(f"缺失比例: {(len(gdf) - len(valid_gdf)) / len(gdf) * 100:.2f}%")

# 准备空间数据 - 只使用有效样本
X_tensor, spatial_lags_tensor, spatial_weight_matrix, lisa_cluster, w, coords, scaler = prepare_spatial_data(
    valid_gdf, predictors, target_col
)
#%%
# 保存预处理信息
preprocess_info = {
    'scaler_mean': scaler.mean_,
    'scaler_scale': scaler.scale_,
    'predictors': predictors,
    'target_col': target_col,
    'spatial_weight_matrix': spatial_weight_matrix,
    'lisa_cluster': lisa_cluster,
    'coords': coords
}
with open(os.path.join(results_dir, f"preprocess_info_{target_col}.pkl"), 'wb') as f:
    pickle.dump(preprocess_info, f)

# 模型参数
input_dim = X_tensor.shape[1]  # 13 (12个特征 + 1个目标变量)
z1_dim = 13  # 因果因子维度
z2_dim = 1   # 每个因果因子的维度
# 初始化模型

model = SpatialUrbanCausalVAE(
    input_dim, 
    spatial_weight_matrix, 
    lisa_cluster,
    coords,  # 新增坐标参数
    z1_dim=z1_dim, 
    z2_dim=z2_dim,
    spatial_reg_weight=0.01,
    lag_reg_weight=0.001,
    dag_weight = 0.5
).to(device)
# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# 动态学习率调度器 - 带热重启的余弦退火
scheduler = CosineAnnealingWarmRestarts(
    optimizer, 
    T_0=1000,        # 初始重启周期
    T_mult=4,       # 每次重启周期加倍
    eta_min=1e-5    # 最小学习率
)
# 训练参数
epochs = 60000

# 准备数据加载器
train_dataset = TensorDataset(X_tensor, spatial_lags_tensor)
train_loader = DataLoader(train_dataset, batch_size=len(X_tensor), shuffle=False)
# 训练循环
loss_history = {
    'total_loss': [], 'recon_loss': [], 'kl_loss': [], 
    'dag_loss': [], 'spatial_reg': [], 'lag_reg': [], 'lr': []
}
# 保存因果结构演化
causal_evolution = []

best_loss = float('inf')
best_model = None

for epoch in range(epochs):
    model.train()
    # 获取整个数据集作为单个batch
    batch = next(iter(train_loader))
    x = batch[0].to(device)
    spatial_lags = batch[1].to(device)
    optimizer.zero_grad()
    # 计算损失
    loss, summaries = model.loss(x, spatial_lags)
    current_loss = summaries['loss/total']

    # 检查并保存最佳模型
    # if current_loss < best_loss:
    #     best_loss = current_loss
    #     best_model = model  # 保存模型状态
    #     # print(f"best Epoch {epoch}/{epochs} | Total Loss: {summaries['loss/total']:.4f} | ")

    # 反向传播
    loss.backward()
    optimizer.step()
    # 更新学习率
    scheduler.step()
    # 记录损失
    # 每10个epoch记录一次损失值和学习率
    if epoch % 10 == 0:
        # 记录损失
        loss_history['total_loss'].append(summaries['loss/total'])
        loss_history['recon_loss'].append(summaries['loss/reconstruction'])
        loss_history['kl_loss'].append(summaries['loss/kl'])
        loss_history['dag_loss'].append(summaries['loss/dag'])
        loss_history['spatial_reg'].append(summaries['loss/spatial_reg'])
        loss_history['lag_reg'].append(summaries['loss/lag_reg'])
        # 记录学习率
        loss_history['lr'].append(optimizer.param_groups[0]['lr'])
    
    # 每1000个epoch保存一次因果结构图
    if epoch % 1000 == 0:
        model.eval()
        causal_structure = model.get_causal_structure(threshold=0.3)
        raw_matrix = model.get_raw_causal_structure()
        causal_evolution.append({
            'epoch': epoch,
            'causal_structure': causal_structure,
            'raw_matrix': raw_matrix
        })
        model.train()
        
    if epoch % 1000 == 0:
    # 打印训练进度
          print(f"Epoch {epoch}/{epochs} | Total Loss: {summaries['loss/total']:.4f} | "
          f"Recon: {summaries['loss/reconstruction']:.4f} | KL: {summaries['loss/kl']:.4f} | "
          f"DAG: {summaries['loss/dag']:.4f} | "
          f"Spatial Reg: {summaries['loss/spatial_reg']:.9f} | "
          f"Lag Reg: {summaries['loss/lag_reg']:.4f} | "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")   
        

#%%
# 训练完成后保存最终模型和结果
final_model_path = os.path.join(model_dir, f"SCVAE_{target_col}.pth")
torch.save(model.state_dict(), final_model_path)

# 保存模型参数信息（用于后续加载）
model_params = {
    'input_dim': input_dim,
    'z1_dim': z1_dim,
    'z2_dim': z2_dim,
    'spatial_reg_weight': 0.01,
    'lag_reg_weight': 0.001,
    'dag_weight': 0.5
}
with open(os.path.join(model_dir, f"model_params_{target_col}.pkl"), 'wb') as f:
    pickle.dump(model_params, f)

# 保存损失历史
with open(os.path.join(results_dir, f"loss_history_{target_col}.pkl"), 'wb') as f:
    pickle.dump(loss_history, f)

# 保存因果结构演化
with open(os.path.join(results_dir, f"causal_evolution_{target_col}.pkl"), 'wb') as f:
    pickle.dump(causal_evolution, f)


print("训练完成！所有结果已保存。")

#%%
