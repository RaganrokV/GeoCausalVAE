import run_options as options
#%%
import torch
from torch.utils.data import TensorDataset, DataLoader
import os

os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import pandas as pd
import numpy as np

import time
import random
import json
import pickle

from func_getdata import prepare_spatial_data
from func_GeoCausalVAE import SpatialUrbanCausalVAE as SpatialUrbanVAE_NoDAG

import torch.optim as optim

from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts
)

from func_metric import (
    evaluate_reconstruction,
    benchmark_model
)

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# =====================================================
# Folder
# =====================================================

model_dir = "/home/ps/haichao/6-BEEC/R2_comparison_models/"
results_dir = "/home/ps/haichao/6-BEEC/R2_comparison_results/"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
options.save_snapshot(results_dir, __file__)

# =====================================================
# Data
# =====================================================

file_path = "/home/ps/haichao/6-BEEC/Data/urban_statistic.csv"

shp_path = "/home/ps/haichao/6-BEEC/shp2/乡镇级数据.shp"

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

df = pd.read_csv(file_path).rename(columns=COLUMN_MAPPING)

gdf = gpd.read_file(shp_path).merge(
    df,
    on='NAME',
    how='inner'
)

targets = [
    'Average',
    'Sedan',
    'Bus',
    'Logistic'
]

predictors = [
    'Building density',
    'POI diversity',
    'Land use diversity',
    'Charging station',
    'Bus stop',
    'Road density',
    'Road curvature',
    'Arterial ratio',
    'Connectivity',
    'Slope',
    'Elevation',
    'Population'
]

for target in targets:
    gdf.loc[gdf[target] == 0, target] = np.nan

# =====================================================
# Results Container
# =====================================================

all_results = []

# =====================================================
# Loop Vehicle Types
# =====================================================

for target_col in targets:

    random.seed(1000)
    np.random.seed(1000)
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1000)

    print("\n")
    print("=" * 60)
    print(f"Running GeoCausalVAE_wo_DAG : {target_col}")
    print("=" * 60)

    valid_mask = gdf[predictors + [target_col]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)

    valid_gdf = gdf[
        valid_mask
    ].copy()

    print(
        f"Valid Samples: {len(valid_gdf)}"
    )

    (
        X_tensor,
        spatial_lags_tensor,
        spatial_weight_matrix,
        lisa_cluster,
        w,
        coords,
        scaler

    ) = prepare_spatial_data(
        valid_gdf,
        predictors,
        target_col
    )

    # =====================================
    # Model
    # =====================================

    input_dim = X_tensor.shape[1]

    z1_dim = 13
    z2_dim = 1

    model = SpatialUrbanVAE_NoDAG(
        input_dim,
        spatial_weight_matrix,
        lisa_cluster,
        coords,
        z1_dim=z1_dim,
        z2_dim=z2_dim,
        spatial_reg_weight=0.01,
        lag_reg_weight=0.0,
        prior_dag_weight=0.0,
        dag_weight=0.0,
        use_dag=False
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-3
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,
        T_mult=4,
        eta_min=1e-5
    )

    epochs = 85000

    train_dataset = TensorDataset(
        X_tensor,
        spatial_lags_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=len(X_tensor),
        shuffle=False
    )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    with open(os.path.join(results_dir, f"preprocess_28_wo_dag_{target_col}.pkl"), 'wb') as f:
        pickle.dump({'scaler_mean': scaler.mean_, 'scaler_scale': scaler.scale_,
                     'predictors': predictors, 'target_col': target_col,
                     'sample_names': valid_gdf['NAME'].astype(str).tolist(),
                     'random_seed': 1000, 'model_version': 'reviewer1_r2'}, f)

    training_start_time = time.time()

    # =====================================
    # Training
    # =====================================

    for epoch in range(epochs):

        model.train()

        batch = next(iter(train_loader))

        x = batch[0].to(device)

        spatial_lags = batch[1].to(device)

        optimizer.zero_grad()

        loss, summaries = model.loss(
            x,
            spatial_lags
        )

        loss.backward()

        optimizer.step()

        scheduler.step()

        if epoch % 1000 == 0:

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Loss: {summaries['loss/total']:.4f} | "
                f"Recon: {summaries['loss/reconstruction']:.4f} | "
                f"KL: {summaries['loss/kl']:.4f} | "
                f"Spatial Reg: {summaries['loss/spatial_reg']:.6f} | "
                f"Lag Reg: {summaries['loss/lag_reg']:.6f}"
            )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_time = (
        time.time()
        - training_start_time
    )

    # =====================================
    # Save Model
    # =====================================

    torch.save(

        model.state_dict(),

        os.path.join(
            model_dir,
            f"GeoCausalVAE_wo_DAG_{target_col}.pth"
        )
    )

    # =====================================
    # Evaluation
    # =====================================

    metrics = evaluate_reconstruction(
        model=model,
        X_tensor=X_tensor,
        spatial_lags_tensor=spatial_lags_tensor,
        device=device,
        scaler=scaler,
        target_idx=input_dim - 1
    )

    runtime = benchmark_model(
        model=model,
        X_tensor=X_tensor,
        spatial_lags_tensor=spatial_lags_tensor,
        device=device
    )

    runtime["Training_Time_s"] = training_time
    runtime["Training_Time_min"] = training_time / 60

    result = {
        "Vehicle": target_col,
        "Model": "GeoCausalVAE_wo_DAG"
    }

    result.update(metrics)
    result.update(runtime)

    all_results.append(result)

# =====================================================
# Final Table
# =====================================================

final_df = pd.DataFrame(
    all_results
)

final_results_path = os.path.join(
    results_dir,
    "wo_dag.csv"
)

final_df.to_csv(
    final_results_path,
    index=False
)
print("\n")
print("=" * 60)
print("GeoCausalVAE_wo_DAG Results")
print("=" * 60)

final_df

