"""Shared local paths and map preparation; no network access or embedded tokens."""
import os
import warnings
from pathlib import Path

INPUT_ROOT = Path(os.environ.get('BEEC_INPUT_ROOT', Path(__file__).parent / 'input'))
OUTPUT_ROOT = Path(os.environ.get('BEEC_OUTPUT_ROOT', Path(__file__).parent / 'generated_outputs'))

class MapInputUnavailable(ValueError):
    pass

def output_path(name):
    path = OUTPUT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def read_map(path, **kwargs):
    import geopandas as gpd
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing map input: {path}')
    data = gpd.read_file(path, **kwargs)
    if data.crs is None:
        raise MapInputUnavailable(f'CRS is missing: {path}; supply the actual source CRS, do not guess.')
    data = data.loc[data.geometry.notna() & ~data.geometry.is_empty].copy()
    invalid = ~data.geometry.is_valid
    if invalid.any():
        warnings.warn(f'{path.name}: repairing {int(invalid.sum())} invalid geometries.')
        data.loc[invalid, 'geometry'] = data.loc[invalid, 'geometry'].make_valid()
    return data.to_crs(epsg=4326)

def clip_map(data, bbox, points=False):
    result = data.clip(bbox).copy()
    if points and not result.empty:
        nonpoints = result.geometry.geom_type != 'Point'
        if nonpoints.any():
            warnings.warn(f'Using representative points for {int(nonpoints.sum())} non-point map features.')
            result.loc[nonpoints, 'geometry'] = result.loc[nonpoints, 'geometry'].representative_point()
    if result.empty:
        raise MapInputUnavailable('No features in the displayed bounds; map skipped, not fabricated.')
    return result

def save_figure(fig, name):
    import matplotlib.pyplot as plt
    path = output_path(name)
    fig.savefig(path, dpi=int(os.environ.get('BEEC_DPI', '200')), bbox_inches='tight')
    if os.environ.get('BEEC_SHOW', '0') == '1':
        plt.show()
    plt.close(fig)
    print(f'Saved: {path}')
    return path
