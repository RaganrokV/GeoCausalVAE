#%%
"""运行开关：修改后重新启动脚本生效。默认放松非致命限制。"""
import json
import sys

# 避免日志打印中断计算。
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(errors="backslashreplace")
from pathlib import Path

# False: 有环仅提示，继续保存与评估；。
REQUIRE_EXACT_DAG = False
# False: 接受有限的加权先验；。
REQUIRE_BINARY_PRIOR = False
# False: 失败回归保留缺失p值。
REQUIRE_COMPLETE_PRIOR = False
# SLM/SEM最低AIC选择。
PRIOR_REGRESSION_MODE = 'slm'
# False: strengthen/weaken幅度不限制在[0,1]。
LIMIT_INTERVENTION_STRENGTH = False
# True: 调用者可以设置非零lag_reg_weight；。
ALLOW_EXTRA_LAG_PENALTY = True
# True: 小样本时自动把k降为n-1并提示；至少需要2个空间单元。
ADAPT_KNN_TO_SAMPLE_SIZE = True
# False: 无scaler时报告标准化尺度指标并标记。
REQUIRE_ORIGINAL_SCALE_METRICS = False
# 仅在源文件缺少CRS时使用。
SOURCE_CRS = None


def snapshot():
    return {name: value for name, value in globals().items() if name.isupper()}


def save_snapshot(directory, script):
    path = Path(directory) / (Path(script).stem + '_runtime_options.json')
    path.write_text(json.dumps(snapshot(), indent=2), encoding='utf-8')


def effective_knn_k(n, k):
    if n < 2:
        raise ValueError('KNN requires at least two valid spatial units.')
    if k >= n:
        if not ADAPT_KNN_TO_SAMPLE_SIZE:
            raise ValueError(f'Need more than {k} units, or enable ADAPT_KNN_TO_SAMPLE_SIZE.')
        print(f'WARNING: KNN k reduced from {k} to {n-1} for {n} units.')
        return n - 1
    return k


def resolve_crs(gdf):
    if gdf.crs is None:
        if SOURCE_CRS is None:
            raise ValueError('Missing CRS: set SOURCE_CRS to the confirmed source CRS in run_options.py.')
        return gdf.set_crs(SOURCE_CRS)
    return gdf


def fit_prior_regression(y, X, w, names, slm_constructor, sem_constructor):
    if PRIOR_REGRESSION_MODE not in ('slm', 'slm_sem_aic'):
        raise ValueError('PRIOR_REGRESSION_MODE must be slm or slm_sem_aic.')
    models = []
    constructors = [('SLM', slm_constructor)]
    if PRIOR_REGRESSION_MODE == 'slm_sem_aic':
        constructors.append(('SEM', sem_constructor))
    for label, constructor in constructors:
        try:
            model = constructor(y, X, w, name_x=names)
            models.append((label, model))
        except Exception as e:
            print(f'WARNING: {label} regression failed: {e}')
    if not models:
        if REQUIRE_COMPLETE_PRIOR:
            raise RuntimeError('All requested prior regressions failed.')
        print('WARNING: continuing with missing p-values for this dependent variable.')
        return None, None
    if len(models) == 1:
        return models[0]
    return min(models, key=lambda item: item[1].aic)
