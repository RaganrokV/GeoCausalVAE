import run_options as options
import numpy as np
import torch

import time

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def evaluate_reconstruction(
        model,
        X_tensor,
        spatial_lags_tensor=None,
        device="cpu",
        scaler=None,
        target_idx=-1,
        n_samples=100,
        seed=1000
):

    if scaler is None and options.REQUIRE_ORIGINAL_SCALE_METRICS:
        raise ValueError("Provide the fitted scaler to report original-scale energy metrics.")
    if scaler is None:
        print("WARNING: no scaler supplied; metrics are on standardized input scale, not original units.")
    if n_samples < 1 or not -X_tensor.shape[1] <= target_idx < X_tensor.shape[1]:
        raise ValueError("Invalid evaluation samples or target index.")
    was_training = model.training
    model.eval()
    X = X_tensor.to(device)
    spatial_lags = spatial_lags_tensor.to(device) if spatial_lags_tensor is not None else None
    cuda_devices = [X.device.index] if X.is_cuda else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            torch.manual_seed(seed)
            predictions = torch.zeros_like(X[:, target_idx])
            for _ in range(n_samples):
                outputs = model(X, spatial_lags) if spatial_lags is not None else model(X)
                recon_x = outputs[0] if isinstance(outputs, tuple) else outputs
                predictions += recon_x[:, target_idx] / n_samples
        scale = float(scaler.scale_[target_idx]) if scaler is not None else 1.0
        mean = float(scaler.mean_[target_idx]) if scaler is not None else 0.0
        y_true = X[:, target_idx].cpu().numpy() * scale + mean
        y_pred = predictions.cpu().numpy() * scale + mean
        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "R2": float(r2_score(y_true, y_pred)),
            "Metric_Target": "energy_only",
            "Metric_Scale": "original" if scaler is not None else "standardized",
            "Evaluation": "in_sample_reconstruction",
            "Evaluation_MC_samples": int(n_samples),
            "Evaluation_Seed": int(seed)
        }
    finally:
        model.train(was_training)


    


def benchmark_model(
        model,
        X_tensor,
        spatial_lags_tensor=None,
        device="cpu",
        repeats=100
):
    """报告整批推断均值；CUDA计时前后同步。"""
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    was_training = model.training
    model.eval()
    X = X_tensor.to(device)
    spatial_lags = spatial_lags_tensor.to(device) if spatial_lags_tensor is not None else None
    cuda_devices = [X.device.index] if X.is_cuda else []
    try:
        with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
            for _ in range(10):
                _ = model(X, spatial_lags) if spatial_lags is not None else model(X)
            if X.is_cuda:
                torch.cuda.synchronize(X.device)
            start = time.perf_counter()
            for _ in range(repeats):
                _ = model(X, spatial_lags) if spatial_lags is not None else model(X)
            if X.is_cuda:
                torch.cuda.synchronize(X.device)
            elapsed = (time.perf_counter() - start) * 1000 / repeats
        return {
            "Parameters": int(parameter_count),
            "Inference_Time_ms": elapsed,
            "Inference_Per_Unit_ms": elapsed / len(X),
            "Inference_Batch_Size": int(len(X)),
            "Inference_Repeats": int(repeats)
        }
    finally:
        model.train(was_training)
