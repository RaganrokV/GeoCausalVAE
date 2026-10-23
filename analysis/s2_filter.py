#%%
"""Per-vehicle trip extraction and original feature-range filtering."""
import argparse
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from release_utils import INPUT_ROOT, output_path

filter_rules = {
    '行程能耗': {'min': 0, 'max': 304},
    '行程距离': {'min': 1, 'max': 320},
    '行程时间': {'min': 1, 'max': 720},
    '消耗SOC': {'min': 0, 'max': 100}
}

def trip_extract(df, vehicle_column=None, single_vehicle=False, timestamp_unit='ms', stats=None):
    stats = stats if stats is not None else {}
    def count(name, value):
        stats[name] = stats.get(name, 0) + int(value)
    if df.empty:
        return []
    required = ['collectiontime', 'vehicledata_vehiclestatus', 'vehicledata_runmodel']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing trip extraction columns: {sorted(missing)}')
    if vehicle_column is None:
        vehicle_column = next((c for c in ['vehicle_id', 'vin', 'VIN'] if c in df.columns), None)
    if vehicle_column is not None and vehicle_column not in df.columns:
        raise ValueError(f'Vehicle ID column not found: {vehicle_column}')
    if vehicle_column is None and not single_vehicle:
        raise ValueError('No vehicle ID column. Set --vehicle-column, or --single-vehicle only when each daily frame contains one vehicle.')
    data = df.copy()
    count('input_rows', len(data))
    if pd.api.types.is_datetime64_any_dtype(data['collectiontime']):
        times = pd.to_datetime(data['collectiontime'], errors='coerce', utc=True)
    elif pd.api.types.is_numeric_dtype(data['collectiontime']):
        times = pd.to_datetime(data['collectiontime'], unit=timestamp_unit, errors='coerce', utc=True)
    else:
        numeric = pd.to_numeric(data['collectiontime'], errors='coerce')
        if numeric.notna().sum() == data['collectiontime'].notna().sum():
            times = pd.to_datetime(numeric, unit=timestamp_unit, errors='coerce', utc=True)
        else:
            times = pd.to_datetime(data['collectiontime'], errors='coerce', utc=True)
    data['collectiontime'] = times
    count('invalid_timestamp_rows', times.isna().sum())
    data = data.loc[times.notna()].copy()
    if vehicle_column:
        bad_id = data[vehicle_column].isna() | data[vehicle_column].astype(str).str.strip().eq('')
        count('missing_vehicle_id_rows', bad_id.sum())
        data = data.loc[~bad_id].copy()
        groups = data.groupby(vehicle_column, sort=False)
    else:
        groups = [('single_vehicle', data)]
    trips = []
    for _, group in groups:
        group = group.sort_values('collectiontime', kind='stable')
        # Same vehicle/time: retain first record consistently; count discarded records.
        count('duplicate_timestamp_rows', group.duplicated('collectiontime').sum())
        group = group.drop_duplicates('collectiontime', keep='first')
        active = (pd.to_numeric(group['vehicledata_vehiclestatus'], errors='coerce') == 1) & (pd.to_numeric(group['vehicledata_runmodel'], errors='coerce') == 1)
        trip_whole = group.loc[active].reset_index(drop=True)
        if trip_whole.empty:
            continue
        # Original definition retained: filter active/electric records, then split gaps >5 minutes.
        segments = trip_whole['collectiontime'].diff().gt(pd.Timedelta(minutes=5)).cumsum()
        for _, trip in trip_whole.groupby(segments, sort=False):
            minutes = (trip['collectiontime'].iloc[-1] - trip['collectiontime'].iloc[0]).total_seconds()/60
            if minutes >= 10:
                trips.append(trip.reset_index(drop=True))
            else:
                count('segments_under_10_minutes', 1)
    return trips

def trip_features(trip_df, interval_mode='capped'):
    required = ['vehicledata_sumcurrent', 'vehicledata_sumvoltage', 'vehicledata_summileage', 'vehicledata_soc']
    missing = set(required) - set(trip_df.columns)
    if missing:
        raise ValueError(f'Missing energy/filter columns: {sorted(missing)}')
    values = trip_df[required].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(values.to_numpy()).all():
        return None
    interval = trip_df['collectiontime'].diff().iloc[1:].dt.total_seconds().to_numpy()
    if len(interval) == 0 or not np.isfinite(interval).all() or (interval <= 0).any():
        return None
    if interval_mode == 'capped':
        mode_collect_time = pd.Series(interval).mode().iloc[0]
        interval = np.minimum(interval, mode_collect_time)
    elif interval_mode != 'actual':
        raise ValueError('interval_mode must be capped or actual')
    # Preserve original signed-current, left-endpoint integration; inputs in A and V.
    energy = np.sum(values['vehicledata_sumcurrent'].to_numpy()[:-1] * values['vehicledata_sumvoltage'].to_numpy()[:-1] * interval)/3600000
    return {
        '行程能耗': float(energy),
        '行程距离': float(values['vehicledata_summileage'].iloc[-1] - values['vehicledata_summileage'].iloc[0]),
        '行程时间': (trip_df['collectiontime'].iloc[-1] - trip_df['collectiontime'].iloc[0]).total_seconds()/60,
        '消耗SOC': float(values['vehicledata_soc'].iloc[0] - values['vehicledata_soc'].iloc[-1])
    }

def filter_daily_data(daily_data, vehicle_column=None, single_vehicle=False, timestamp_unit='ms', interval_mode='capped'):
    if not isinstance(daily_data, dict):
        raise ValueError('Expected dictionary: date -> DataFrame.')
    cleaned, details, stats = {}, [], {}
    before = 0
    for date, frame in daily_data.items():
        trips = trip_extract(frame, vehicle_column, single_vehicle, timestamp_unit, stats)
        before += len(trips)
        valid = []
        for index, trip in enumerate(trips):
            features = trip_features(trip, interval_mode)
            reasons = ['Missing/nonfinite telemetry or invalid time intervals'] if features is None else [name for name, limits in filter_rules.items() if not limits['min'] <= features[name] <= limits['max']]
            if reasons:
                details.append({'date': str(date), 'trip_index': index, 'reasons': '; '.join(reasons), **(features or {})})
            else:
                valid.append(trip)
        cleaned[date] = valid
    after = sum(map(len, cleaned.values()))
    report = {'input_qc': stats, 'trips_before_filter': before, 'trips_after_filter': after, 'removed_trips': before-after, 'removed_percent': 100*(before-after)/before if before else 0., 'interval_mode': interval_mode, 'timestamp_unit': timestamp_unit, 'vehicle_column': vehicle_column or 'auto: vehicle_id/vin/VIN', 'single_vehicle_explicit': single_vehicle, 'filter_rules': filter_rules}
    return cleaned, report, details

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(INPUT_ROOT/'Data/daily_data.pkl'))
    parser.add_argument('--vehicle-column')
    parser.add_argument('--single-vehicle', action='store_true')
    parser.add_argument('--timestamp-unit', default='ms', choices=['s','ms','us','ns'])
    parser.add_argument('--interval-mode', default='capped', choices=['capped','actual'])
    args = parser.parse_args()
    with open(args.input, 'rb') as f:
        daily_data = pickle.load(f)
    cleaned, report, details = filter_daily_data(daily_data, args.vehicle_column, args.single_vehicle, args.timestamp_unit, args.interval_mode)
    with output_path('trip_dfs_cleaned.pkl').open('wb') as f:
        pickle.dump(cleaned, f)
    output_path('filter_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.DataFrame(details, columns=['date','trip_index','reasons',*filter_rules]).to_csv(output_path('removed_trips.csv'), index=False, encoding='utf-8-sig')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['input_qc'].get('duplicate_timestamp_rows', 0):
        warnings.warn('Duplicate vehicle/timestamp records retained first; inspect input_qc before using changed sample results.')

if __name__ == '__main__':
    main()
