import run_options as options
#%%
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
os.environ['USE_PYGEOS'] = '0'

import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
import time
import random
import json

from func_getdata import prepare_spatial_data
from func_GeoCausalVAE import SpatialUrbanCausalVAE as SpatialUrbanCausalVAE_NoSpatial

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from func_metric import (
    evaluate_reconstruction,
    benchmark_model
)

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

def zero_spatial_weight_matrix(W):
    """
    Set spatial weight matrix to zero.
    Supports both sparse and dense torch tensors.
    """

    if W.is_sparse:

        indices = torch.empty(
            (2, 0),
            dtype=torch.long,
            device=W.device
        )

        values = torch.empty(
            (0,),
            dtype=W.dtype,
            device=W.device
        )

        W_zero = torch.sparse_coo_tensor(
            indices,
            values,
            size=W.shape,
            device=W.device
        ).coalesce()

    else:

        W_zero = torch.zeros_like(W)

    return W_zero

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
    print(f"Running GeoCausalVAE_wo_SpatialReg : {target_col}")
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

    
    # =====================================================
    # Remove spatial-lag information for w/o spatial input
    # =====================================================

    
    spatial_lags_tensor = torch.zeros_like(spatial_lags_tensor)
    spatial_weight_matrix = zero_spatial_weight_matrix(spatial_weight_matrix)


    # =====================================
    # Model
    # =====================================

    input_dim = X_tensor.shape[1]

    z1_dim = 13
    z2_dim = 1

    variable_order = predictors + [target_col]
    prior_path = f"/home/ps/haichao/6-BEEC/Results_R2/binary_table_{target_col}.csv"
    prior_table = pd.read_csv(prior_path, index_col=0)
    if not prior_table.index.is_unique or not prior_table.columns.is_unique:
        raise ValueError("Duplicate empirical-prior labels.")
    if set(prior_table.index) != set(variable_order) or set(prior_table.columns) != set(variable_order):
        raise ValueError("Empirical-prior labels do not match model variables.")
    prior_dag = prior_table.loc[variable_order, variable_order].to_numpy(dtype=float).T.copy()
    if not np.isfinite(prior_dag).all():
        raise ValueError("Expected finite prior weights.")
    if not np.isin(prior_dag, [0, 1]).all():
        if options.REQUIRE_BINARY_PRIOR:
            raise ValueError("Expected a binary prior; disable REQUIRE_BINARY_PRIOR for weighted priors.")
        print("WARNING: using a weighted prior rather than the binary prior in Eq. (6).")
    np.fill_diagonal(prior_dag, 0)

    model = SpatialUrbanCausalVAE_NoSpatial(
        input_dim,
        spatial_weight_matrix,
        lisa_cluster,
        coords,
        z1_dim=z1_dim,
        z2_dim=z2_dim,
        dag_weight=0.5,
        spatial_reg_weight=0.0,
        lag_reg_weight=0.0,
        prior_dag=prior_dag,
        prior_dag_weight=1.0
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
    with open(os.path.join(results_dir, f"preprocess_27_wo_spatial_{target_col}.pkl"), 'wb') as f:
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
                f"DAG: {summaries['loss/dag']:.4f}"
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

    graph_check = model.dag.graph_diagnostics()
    with open(os.path.join(results_dir, f"wo_spatial_graph_check_{target_col}.json"), 'w', encoding='utf-8') as f:
        json.dump(graph_check, f, indent=2)
    # 无环性仅作诊断。
    if not graph_check['finite']:
        raise FloatingPointError("Non-finite adjacency weights; inspect numerical training failure.")
    if not graph_check['is_dag']:
        if options.REQUIRE_EXACT_DAG:
            raise RuntimeError('Exact DAG check requested; disable REQUIRE_EXACT_DAG to continue with cycles.')
        print(
            "WARNING: The learned graph contains cycles. Continuing model saving and evaluation. "
            "NOTEARS is a soft penalty; this graph is not a verified DAG. "
            f"h(A)={graph_check['notears_h']}"
        )

    torch.save(

        model.state_dict(),

        os.path.join(
            model_dir,
            f"GeoCausalVAE_wo_SpatialReg_{target_col}.pth"
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
        "Model": "GeoCausalVAE_wo_SpatialReg"
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
    "wo_spatial.csv"
)

final_df.to_csv(
    final_results_path,
    index=False
)

print("\n")
print("=" * 60)
print("Final Results")
print("=" * 60)


final_df
