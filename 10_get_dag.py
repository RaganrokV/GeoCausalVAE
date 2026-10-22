import run_options as options
#%%
import os
import json
import pandas as pd
import geopandas as gpd
import numpy as np

from libpysal.weights import KNN
from spreg import ML_Lag, ML_Error
from sklearn.preprocessing import StandardScaler

# =====================================================
# Paths
# =====================================================

file_path = "/home/ps/haichao/6-BEEC/Data/urban_statistic.csv"
shp_path = "/home/ps/haichao/6-BEEC/shp2/乡镇级数据.shp"

output_dir = "/home/ps/haichao/6-BEEC/Results_R2/"
os.makedirs(output_dir, exist_ok=True)
options.save_snapshot(output_dir, __file__)

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
# Variables
# =====================================================

base_predictors = [
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

targets = [
    'Average',
    'Sedan',
    'Bus',
    'Logistic'
]

# 与 prepare_spatial_data 保持一致
k_dict = {
    'Average': 4,
    'Sedan': 11,
    'Bus': 15,
    'Logistic': 4
}

# =====================================================
# Read data
# =====================================================

df = pd.read_csv(file_path).rename(columns=COLUMN_MAPPING)

gdf = gpd.read_file(shp_path).merge(
    df,
    on='NAME',
    how='inner'
)

# 
for target in targets:
    gdf.loc[gdf[target] == 0, target] = np.nan

# CRS 转换
gdf = options.resolve_crs(gdf)
if gdf.crs.is_geographic:
    gdf = gdf.to_crs(epsg=32650)

print("CRS:", gdf.crs)

# =====================================================
# Construct p-value tables and binary DAG priors
# =====================================================

p_value_tables = {}
binary_tables = {}

for target in targets:

    print("\n")
    print("=" * 80)
    print(f"Processing vehicle: {target}")
    print("=" * 80)

    predictors = base_predictors + [target]

    # 当前车型有效样本
    data = gdf[
        ['geometry'] + predictors
    ].replace([np.inf, -np.inf], np.nan).dropna().copy()

    print(f"Valid samples: {len(data)}")

    if len(data) == 0:
        print(f"Warning: no valid samples for {target}, skipped.")
        continue

    # 使用车型对应的 KNN k
    k_value = options.effective_knn_k(len(data), k_dict[target])

    print(f"Using KNN k = {k_value}")

    w = KNN.from_dataframe(
        data,
        k=k_value,
        use_index=False
    )

    w.transform = 'r'

    # 创建 13 x 13 p-value matrix
    # row = dependent variable
    # column = explanatory variable
    table = pd.DataFrame(
        np.nan,
        index=predictors,
        columns=predictors
    )

    # =================================================
    # Run spatial regressions
    # =================================================

    regression_models = {}
    for dep_var in predictors:

        print(f"  Dependent variable: {dep_var}")

        y = data[dep_var].values

        indep_vars = [
            var for var in predictors
            if var != dep_var
        ]

        X = data[indep_vars].values

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        model_name, model = options.fit_prior_regression(
            y, X_scaled, w, indep_vars, ML_Lag, ML_Error
        )
        regression_models[dep_var] = model_name
        if model is None:
            continue

        # Extract p-values
        # z_stat order:
        # index 0 = constant
        # index 1...len(indep_vars) = explanatory variables
        for i, var in enumerate(indep_vars):

            idx = i + 1

            try:
                p_value = model.z_stat[idx][1]

                table.loc[
                    dep_var,
                    var
                ] = p_value

            except Exception as e:
                print(
                    f"    Failed to extract p-value for {var}: {str(e)}"
                )

    off_diagonal = ~np.eye(len(predictors), dtype=bool)
    missing = off_diagonal & ~np.isfinite(table.values)
    if missing.any() and options.REQUIRE_COMPLETE_PRIOR:
        raise ValueError(f"Incomplete regression p-values for {target}; disable REQUIRE_COMPLETE_PRIOR to continue.")
    if missing.any():
        print(f"WARNING: {int(missing.sum())} unknown prior entries for {target}; binary entries set to 0, not evidence of absent associations.")
        table.values[missing] = np.nan
    with open(os.path.join(output_dir, f"prior_diagnostics_{target}.json"), 'w', encoding='utf-8') as f:
        json.dump({'regression_models': regression_models, 'effective_k': int(k_value),
                   'unknown_edges': [[predictors[i], predictors[j]] for i, j in zip(*np.where(missing))],
                   'unknown_edge_order': 'dependent, explanatory',
                   'unknown_binary_fill': 0, 'runtime_options': options.snapshot()}, f, indent=2)

    p_value_tables[target] = table

    # =================================================
    # Save p-value table
    # =================================================

    pvalue_path = os.path.join(
        output_dir,
        f"p_value_table_{target}.csv"
    )

    table.to_csv(pvalue_path)

    print(f"Saved p-value table to: {pvalue_path}")

    # =================================================
 
    # =================================================

    binary_df = table.applymap(
        lambda x: 1 if pd.notna(x) and x < 0.10 else 0
    )

    # No self-loop
    np.fill_diagonal(
        binary_df.values,
        0
    )

    binary_tables[target] = binary_df

    binary_path = os.path.join(
        output_dir,
        f"binary_table_{target}.csv"
    )

    binary_df.to_csv(binary_path)

    print(f"Saved binary DAG prior to: {binary_path}")


# =====================================================
# Summary
# =====================================================

print("\n")
print("=" * 80)
print("Finished generating p-value tables and binary DAG priors.")
print("=" * 80)

