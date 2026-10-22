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
import hashlib
from pathlib import Path

from func_metric import (
    evaluate_reconstruction,
    benchmark_model
)

from func_getdata import prepare_spatial_data
from func_GeoCausalVAE import SpatialUrbanCausalVAE

import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# =====================================================
# Device
# =====================================================

device = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# =====================================================
# Folders
# =====================================================

model_dir = "/home/ps/haichao/6-BEEC/GeoCausalVAE_R2_models/"
results_dir = "/home/ps/haichao/6-BEEC/GeoCausalVAE_R2_results/"

os.makedirs(
    model_dir,
    exist_ok=True
)

os.makedirs(
    results_dir,
    exist_ok=True
)
options.save_snapshot(results_dir, __file__)


# =====================================================
# DAG prior directory
# =====================================================

output_dir = "/home/ps/haichao/6-BEEC/Results_R2/"


# =====================================================
# Load binary DAG prior tables
# =====================================================

p_value_tables = {}

for target in [
    'Average',
    'Sedan',
    'Bus',
    'Logistic'
]:

    csv_path = os.path.join(
        output_dir,
        f"binary_table_{target}.csv"
    )

    if os.path.exists(csv_path):

        df_prior = pd.read_csv(
            csv_path,
            index_col=0
        )

        p_value_tables[target] = df_prior

        print(
            f"已读取 {target} 的CSV文件: {csv_path}"
        )

        print(
            f"数据维度: {df_prior.shape}"
        )

    else:

        print(
            f"警告: 未找到 {target} 的CSV文件: {csv_path}"
        )


# =====================================================
# Data paths
# =====================================================

file_path = "/home/ps/haichao/6-BEEC/Data/urban_statistic.csv"
shp_path = "/home/ps/haichao/6-BEEC/shp2/乡镇级数据.shp"


# =====================================================
# Column mapping
# =====================================================

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


# =====================================================
# Read data
# =====================================================

df = pd.read_csv(
    file_path
).rename(
    columns=COLUMN_MAPPING
)

gdf = gpd.read_file(
    shp_path
).merge(
    df,
    on='NAME',
    how='inner'
)


# =====================================================
# Variables
# =====================================================

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


# =====================================================

# =====================================================

for target in targets:

    gdf.loc[
        gdf[target] == 0,
        target
    ] = np.nan


# =====================================================
# Global settings
# =====================================================

epochs = 85000
random_seed = 1000
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)

spatial_reg_weight = 0.01
lag_reg_weight = 0.0  
dag_weight = 0.5
prior_dag_weight = 1

learning_rate = 0.001
weight_decay = 1e-3

z1_dim = 13
z2_dim = 1


# =====================================================
# Results container
# =====================================================

all_results = []


# =====================================================
# Main loop over vehicle categories
# =====================================================

for target_col in targets:

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    print("\n")
    print("=" * 80)
    print(f"Running GeoCausalVAE: {target_col}")
    print("=" * 80)

    # =================================================
    # Load and transpose prior DAG
    # =================================================

    if target_col not in p_value_tables:
        raise FileNotFoundError(f"Missing empirical prior for {target_col}; run 10_get_dag.py first.")

    variable_order = predictors + [target_col]
    prior_table = p_value_tables[target_col]
    if not prior_table.index.is_unique or not prior_table.columns.is_unique:
        raise ValueError("Duplicate labels in empirical prior.")
    if set(prior_table.index) != set(variable_order) or set(prior_table.columns) != set(variable_order):
        raise ValueError(f"Prior labels do not match model variables for {target_col}.")
    # table[dependent, explanatory] -> A[source, target]
    prior_dag = prior_table.loc[variable_order, variable_order].to_numpy(dtype=float).T.copy()
    if not np.isfinite(prior_dag).all():
        raise ValueError("Expected finite prior weights.")
    if not np.isin(prior_dag, [0, 1]).all():
        if options.REQUIRE_BINARY_PRIOR:
            raise ValueError("Expected a binary prior; disable REQUIRE_BINARY_PRIOR for weighted priors.")
        print("WARNING: using a weighted prior rather than the binary prior in Eq. (6).")
    np.fill_diagonal(prior_dag, 0)

    print(
        f"先验DAG矩阵形状: {prior_dag.shape}"
    )

    print(
        "注意: prior_dag 已转置为 A[source, target] 格式"
    )

    # =================================================
    # Valid samples
    # =================================================

    finite_data = gdf[variable_order].replace([np.inf, -np.inf], np.nan)
    valid_mask = finite_data.notna().all(axis=1)
    valid_gdf = gdf[valid_mask].copy()

    print(
        f"总样本数: {len(gdf)}"
    )

    print(
        f"有效样本数 (非缺失): {len(valid_gdf)}"
    )

    print(
        f"缺失样本数: {len(gdf) - len(valid_gdf)}"
    )

    print(
        f"缺失比例: {(len(gdf) - len(valid_gdf)) / len(gdf) * 100:.2f}%"
    )

    # =================================================
    # Prepare spatial data
    # =================================================

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

    input_dim = X_tensor.shape[1]

    # =================================================
    # Save preprocess info
    # =================================================

    preprocess_info = {
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'predictors': predictors,
        'target_col': target_col,
        'spatial_weight_matrix': spatial_weight_matrix,
        'lisa_cluster': lisa_cluster,
        'coords': coords,
        'prior_dag': prior_dag,
        'prior_dag_transposed': True,
        'variable_order': variable_order,
        'random_seed': random_seed,
        'model_version': 'reviewer1_r2',
        'u_columns': predictors
    }

    with open(
        os.path.join(
            results_dir,
            f"preprocess_info_{target_col}.pkl"
        ),
        'wb'
    ) as f:

        pickle.dump(
            preprocess_info,
            f
        )

    # =================================================
    # Model
    # =================================================

    model = SpatialUrbanCausalVAE(
        input_dim,
        spatial_weight_matrix,
        lisa_cluster,
        coords,
        z1_dim=z1_dim,
        z2_dim=z2_dim,
        spatial_reg_weight=spatial_reg_weight,
        lag_reg_weight=lag_reg_weight,
        dag_weight=dag_weight,
        prior_dag=prior_dag,
        prior_dag_weight=prior_dag_weight
    ).to(device)

    raw_matrix_initial = model.get_raw_causal_structure()

    # =================================================
    # Optimizer and scheduler
    # =================================================

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,
        T_mult=4,
        eta_min=1e-5
    )

    # =================================================
    # DataLoader
    # =================================================

    train_dataset = TensorDataset(
        X_tensor,
        spatial_lags_tensor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=len(X_tensor),
        shuffle=False
    )

    # =================================================
    # Training history
    # =================================================

    loss_history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'dag_loss': [],
        'spatial_reg': [],
        'lag_reg': [],
        'prior_dag_loss': [],
        'lr': []
    }

    causal_evolution = []

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_start_time = time.time()

    # =================================================
    # Training loop
    # =================================================

    for epoch in range(epochs):

        model.train()

        batch = next(
            iter(train_loader)
        )

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

        if epoch % 10 == 0:

            loss_history[
                'total_loss'
            ].append(
                summaries['loss/total']
            )

            loss_history[
                'recon_loss'
            ].append(
                summaries['loss/reconstruction']
            )

            loss_history[
                'kl_loss'
            ].append(
                summaries['loss/kl']
            )

            loss_history[
                'dag_loss'
            ].append(
                summaries['loss/dag']
            )

            loss_history[
                'spatial_reg'
            ].append(
                summaries['loss/spatial_reg']
            )

            loss_history[
                'lag_reg'
            ].append(
                summaries['loss/lag_reg']
            )

            loss_history[
                'prior_dag_loss'
            ].append(
                summaries['loss/prior_dag']
            )

            loss_history[
                'lr'
            ].append(
                optimizer.param_groups[0]['lr']
            )

        # Save causal graph every 1000 epochs
        if epoch % 1000 == 0:

            model.eval()

            causal_structure = model.get_causal_structure(
                threshold=0.3
            )

            raw_matrix = model.get_raw_causal_structure()

            causal_evolution.append(
                {
                    'epoch': epoch,
                    'causal_structure': causal_structure,
                    'raw_matrix': raw_matrix
                }
            )

            model.train()

        if epoch % 1000 == 0:

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Total Loss: {summaries['loss/total']:.4f} | "
                f"Recon: {summaries['loss/reconstruction']:.4f} | "
                f"KL: {summaries['loss/kl']:.4f} | "
                f"DAG: {summaries['loss/dag']:.4f} | "
                f"Prior DAG: {summaries['loss/prior_dag']:.4f} | "
                f"Spatial Reg: {summaries['loss/spatial_reg']:.9f} | "
                f"Lag Reg: {summaries['loss/lag_reg']:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    training_time = time.time() - training_start_time

    # 
    graph_check = model.dag.graph_diagnostics()
    with open(os.path.join(results_dir, f"graph_check_{target_col}.json"), 'w', encoding='utf-8') as f:
        json.dump(graph_check, f, indent=2)
    # 。
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

    model.eval()

    final_causal_structure = model.get_causal_structure(
        threshold=0.3
    )

    final_raw_matrix = model.get_raw_causal_structure()

    causal_evolution.append(
        {
            'epoch': epochs,
            'causal_structure': final_causal_structure,
            'raw_matrix': final_raw_matrix
        }
    )

    model.train()

    print(
        f"\nTraining Time for {target_col}: {training_time:.2f} seconds"
    )

    print(
        f"Training Time for {target_col}: {training_time / 60:.2f} minutes"
    )

    # =================================================
    # Save model
    # =================================================

    final_model_path = os.path.join(
        model_dir,
        f"GeoCausalVAE_{target_col}.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )

    # =================================================
    # Save model params
    # =================================================

    model_params = {
        'input_dim': input_dim,
        'z1_dim': z1_dim,
        'z2_dim': z2_dim,
        'spatial_reg_weight': spatial_reg_weight,
        'lag_reg_weight': lag_reg_weight,
        'dag_weight': dag_weight,
        'prior_dag_weight': prior_dag_weight,
        'prior_dag_path': os.path.join(
            output_dir,
            f"binary_table_{target_col}.csv"
        ),
        'prior_dag_transposed': True,
        'variable_order': variable_order,
        'model_version': 'reviewer1_r2',
        'u_columns': predictors,
        'epochs_completed': epochs
    }

    with open(
        os.path.join(
            model_dir,
            f"model_params_{target_col}.pkl"
        ),
        'wb'
    ) as f:

        pickle.dump(
            model_params,
            f
        )

    # =================================================
    # Save loss history
    # =================================================

    with open(
        os.path.join(
            results_dir,
            f"loss_history_{target_col}.pkl"
        ),
        'wb'
    ) as f:

        pickle.dump(
            loss_history,
            f
        )

    # =================================================
    # Save causal evolution
    # =================================================

    with open(
        os.path.join(
            results_dir,
            f"causal_evolution_{target_col}.pkl"
        ),
        'wb'
    ) as f:

        pickle.dump(
            causal_evolution,
            f
        )

    print(
        f"训练完成: {target_col}，模型和训练过程已保存。"
    )

    # =================================================
    # Evaluation
    # =================================================

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

    runtime[
        "Training_Time_s"
    ] = training_time

    runtime[
        "Training_Time_min"
    ] = training_time / 60

    print(
        "\n========== Metrics =========="
    )

    print(
        metrics
    )

    print(
        "\n========== Runtime =========="
    )

    print(
        runtime
    )

    evaluation_result = {
        "Vehicle": target_col,
        "Model": "GeoCausalVAE"
    }

    evaluation_result.update(
        metrics
    )

    evaluation_result.update(
        runtime
    )

    evaluation_df = pd.DataFrame(
        [
            evaluation_result
        ]
    )

    evaluation_path = os.path.join(
        results_dir,
        f"GeoCausalVAE_evaluation_{target_col}.csv"
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False
    )

    print(
        f"Saved evaluation result to: {evaluation_path}"
    )

    # 
    evidence_paths = [file_path, os.path.join(output_dir, f"binary_table_{target_col}.csv"), final_model_path]
    evidence_paths += [str(p) for p in Path(shp_path).parent.glob(Path(shp_path).stem + '.*')]
    evidence_paths += [str(Path(__file__).resolve().parent / name) for name in
                       ['11_GeoCausalVAE.py', 'func_GeoCausalVAE.py', 'func_getdata.py', 'func_metric.py', 'run_options.py']]
    evidence = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in evidence_paths}
    with open(os.path.join(results_dir, f"run_manifest_{target_col}.json"), 'w', encoding='utf-8') as f:
        json.dump({'model_version': 'reviewer1_r2', 'vehicle': target_col,
                   'epochs_completed': epochs, 'random_seed': random_seed,
                   'sample_names': valid_gdf['NAME'].astype(str).tolist(),
                   'variable_order': variable_order, 'torch_version': torch.__version__,
                   'device': str(device), 'metrics': metrics, 'source_hashes': evidence,
                   'graph_check': graph_check,
                   'graph_postprocessing': 'none', 'runtime_options': options.snapshot()}, f, indent=2)
    all_results.append(evaluation_result)


# =====================================================
# Save all results
# =====================================================

final_df = pd.DataFrame(
    all_results
)

final_results_path = os.path.join(
    results_dir,
    "GeoCausalVAE_all_results.csv"
)

final_df.to_csv(
    final_results_path,
    index=False
)

print("\n")
print("=" * 80)
print("Final GeoCausalVAE Results")
print("=" * 80)

print(
    final_df
)

print(
    f"Saved all results to: {final_results_path}"
)

