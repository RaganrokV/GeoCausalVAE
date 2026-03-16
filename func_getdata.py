import torch
import numpy as np
import os
os.environ['USE_PYGEOS'] = '0'
from sklearn.preprocessing import StandardScaler
from esda.moran import Moran, Moran_Local
from libpysal.weights import KNN, lag_spatial  # 修改这里

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ===================================================
# 数据准备与预处理
# ===================================================
def create_spatial_weights(gdf, target_name=None):
    """
    创建空间权重矩阵，直接使用KNN方法
    根据目标变量名称指定不同的k值
    """
    # 定义目标变量到k值的映射
    k_mapping = {
        'Average': 4,
        'Sedan': 11,
        'Bus': 15,
        'Logistic': 4
    }
    
    # 计算质心坐标
    coords = np.array([(pt.x, pt.y) for pt in gdf.centroid])
    
    # 根据目标变量确定k值
    if target_name and target_name in k_mapping:
        k = k_mapping[target_name]
        print(f"为 {target_name} 使用指定k值: {k}")
    else:
        # 默认k值逻辑
        k = min(5, len(coords)-1)
        print(f"使用默认k值: {k}")
    
    # 创建K近邻权重矩阵
    knn = KNN.from_dataframe(gdf, k=k)
    knn.transform = 'r'  # 行标准化
    
    # 检查是否有岛屿区域
    if knn.islands:
        print(f"警告: 仍有 {len(knn.islands)} 个岛屿区域未连接")
    
    return knn

# 在prepare_spatial_data函数中替换权重创建部分
def prepare_spatial_data(gdf, predictor_cols, target_col):
    """
    准备空间数据
    """
    # 坐标系转换
    if gdf.crs is None or gdf.crs.to_epsg() == 4326:
        gdf = gdf.to_crs(epsg=32650)
        print("转换后CRS:", gdf.crs)
    
    # 创建空间权重矩阵（使用改进方法）
    w = create_spatial_weights(gdf, target_col)
    
    # 计算全局莫兰指数
    moran_y = Moran(gdf[target_col].values, w)
    print(f"全局莫兰指数: {moran_y.I:.4f}, p-value: {moran_y.p_sim:.4f}")
    
    # 计算局部莫兰指数
    lisa = Moran_Local(gdf[target_col].values, w)
    
    # 创建分类变量
    lisa_cluster = np.zeros(len(gdf), dtype=int)
    lisa_cluster[(lisa.p_sim < 0.05) & (lisa.q == 1)] = 1  # HH
    lisa_cluster[(lisa.p_sim < 0.05) & (lisa.q == 2)] = 4  # HL
    lisa_cluster[(lisa.p_sim < 0.05) & (lisa.q == 3)] = 3  # LL
    lisa_cluster[(lisa.p_sim < 0.05) & (lisa.q == 4)] = 2  # LH
    
    # 计算空间滞后特征
    spatial_lags = np.zeros_like(gdf[predictor_cols + [target_col]].values)
    for i, col in enumerate(predictor_cols + [target_col]):
        spatial_lags[:, i] = lag_spatial(w, gdf[col].values)
    
    # 标准化数据
    # 使用MinMaxScaler替代StandardScaler
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # full_data = scaler.fit_transform(gdf[predictor_cols + [target_col]].values)
    # spatial_lags = scaler.transform(spatial_lags)

    scaler = StandardScaler()
    full_data = scaler.fit_transform(gdf[predictor_cols + [target_col]].values)
    spatial_lags = scaler.transform(spatial_lags)
    
    # 转换为PyTorch张量
    X_tensor = torch.tensor(full_data, dtype=torch.float32)
    spatial_lags_tensor = torch.tensor(spatial_lags, dtype=torch.float32)
    
    # 转换为稀疏张量表示空间权重矩阵 - 修复部分
    sparse_weights = w.sparse.tocoo()  # 转换为COO格式
    indices = np.vstack((sparse_weights.row, sparse_weights.col))
    values = sparse_weights.data
    spatial_weight_matrix = torch.sparse_coo_tensor(
        indices, values, (len(gdf), len(gdf)), dtype=torch.float32
    ).to(device)
    
    # 计算质心坐标并存储
    coords = torch.tensor(np.array([(pt.x, pt.y) for pt in gdf.centroid]), dtype=torch.float32).to(device)
    
    return X_tensor, spatial_lags_tensor, spatial_weight_matrix, lisa_cluster, w, coords, scaler
