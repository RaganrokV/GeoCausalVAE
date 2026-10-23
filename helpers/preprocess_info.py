
from release_paths import INPUT_ROOT, output_path
#%%
import os
import pickle
import pandas as pd

# =====================================
# Path
# =====================================

results_dir = str(INPUT_ROOT / 'results/')

targets = [
    "Average",
    "Sedan",
    "Bus",
    "Logistic"
]

all_results = []

# =====================================
# Read pkl
# =====================================

for target_col in targets:

    pkl_path = os.path.join(
        results_dir,
        f"preprocess_info_{target_col}.pkl"
    )

    with open(pkl_path, "rb") as f:

        preprocess_info = pickle.load(f)

    target_mean = preprocess_info["scaler_mean"][-1]
    target_std = preprocess_info["scaler_scale"][-1]

    all_results.append({

        "Vehicle": target_col,

        "Target_Mean": target_mean,

        "Target_Std": target_std

    })

# =====================================
# Export
# =====================================

scaler_df = pd.DataFrame(all_results)

scaler_df



print(scaler_df.to_string(index=False))
scaler_df.to_csv(output_path("scaler/target_scalers.csv"), index=False)
