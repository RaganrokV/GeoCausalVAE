#%%
"""Plot trajectory points and administrative boundaries offline."""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from release_utils import INPUT_ROOT, read_map, save_figure

model_mapping = {123115361:'Sedans', 123114912:'Sedans', 417:'Buses', 123115042:'Buses', 361:'Buses', 123115443:'Logistics vehicles', 123115789:'Logistics vehicles'}
color_mapping = {'Logistics vehicles':'#377EB8', 'Sedans':'#4DAF4A', 'Buses':'#E41A1C'}
bounds = [120.84, 30.65, 122.1, 31.88]

def prepare_points(trip_dfs):
    frames = []
    for day in trip_dfs.values():
        if isinstance(day, pd.DataFrame):
            frames.append(day)
        else:
            frames.extend(frame for frame in day if not frame.empty)
    required = ['vmodelseq','vehicleposition_latitude','vehicleposition_longitude']
    if not frames:
        return pd.DataFrame(columns=required+['vehicle_type'])
    data = pd.concat(frames, ignore_index=True)
    missing = set(required)-set(data.columns)
    if missing:
        raise ValueError(f'Missing trajectory columns: {sorted(missing)}')
    for col in required:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data['vehicle_type'] = data['vmodelseq'].map(model_mapping)
    good = data['vehicleposition_latitude'].between(bounds[1],bounds[3]) & data['vehicleposition_longitude'].between(bounds[0],bounds[2]) & data['vehicle_type'].notna()
    print(f'Trajectory rows: {len(data)}; displayed valid mapped rows: {int(good.sum())}; excluded: {int((~good).sum())}')
    return data.loc[good].copy()

def plot_trajectories(trip_dfs, township):
    valid_df = prepare_points(trip_dfs)
    if township.crs is None:
        raise ValueError('Township CRS is missing; supply actual source CRS.')
    township = township.to_crs(epsg=4326)
    fig, ax = plt.subplots(figsize=(10,10))
    township.plot(ax=ax, color='none', edgecolor='black', linewidth=1, alpha=.5)
    for vehicle_type, color in color_mapping.items():
        plot_df = valid_df.loc[valid_df['vehicle_type']==vehicle_type]
        if plot_df.empty:
            continue
        sample_frac = .2 if len(plot_df)>1000000 else .1
        if len(plot_df)>50000:
            plot_df = plot_df.sample(frac=sample_frac, random_state=42)
        ax.scatter(plot_df['vehicleposition_longitude'], plot_df['vehicleposition_latitude'], c=color, s=.6, alpha=.1, edgecolors='none', label=vehicle_type)
    if valid_df.empty:
        ax.text(.5,.5,'No valid trajectory points',transform=ax.transAxes,ha='center')
    else:
        legend = ax.legend(markerscale=15, fontsize=14, loc='upper right')
        handles = getattr(legend,'legend_handles',None)
        if handles is None:
            handles = legend.legendHandles
        for handle in handles:
            handle.set_alpha(.7)
    ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[1],bounds[3]),xlabel='Longitude (°E)',ylabel='Latitude (°N)')
    ax.annotate('N',xy=(.05,.94),xytext=(.05,.83),xycoords='axes fraction',ha='center',arrowprops={'arrowstyle':'->'})
    ax.grid(True,alpha=.1)
    fig.tight_layout()
    return fig

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trips', default=str(INPUT_ROOT/'Data/trip_dfs.pkl'))
    parser.add_argument('--township', default=str(INPUT_ROOT/'shp2/乡镇级数据.shp'))
    args=parser.parse_args()
    with open(args.trips,'rb') as f:
        trip_dfs=pickle.load(f)
    save_figure(plot_trajectories(trip_dfs,read_map(args.township)), 'map_trajectory.png')

if __name__=='__main__':
    main()
