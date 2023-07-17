#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

"""Calculate planting dates
"""

LOOKUP = lambda lut, crop: f'./data/{crop}_rainfed_{lut.lower()}_lookup_3.2.csv'
RUNS = lambda lut, crop: f'./data/{lut.lower()}_{scenario}_{crop}_runs.csv'

TMP_BASE = 6.0
TMP_REF = 10.0

TMP_MAXS = {
    'maize': '-999',
    'springwheat': '-999',
    'winterwheat': '15.0',
}
TMP_MINS = {
    'maize': '15.0',
    'springWheat': '5.0',
    'winterWheat': '-999',
}
CROPS = {
    'maize',
    'springwheat',
    'winterwheat',
}
HYBRIDS = {
    'maize': {
        'CornRM.115': 2425.0,
        'CornRM.110': 2300.0,
        'CornRM.105': 2175.0,
        'CornRM.100': 2050.0,
        'CornRM.95': 1925.0,
        'CornRM.90': 1800.0,
        'CornRM.85': 1675.0,
        'CornRM.80': 1550.0,
        'CornRM.75': 1425.0,
        'CornRM.70': 1300.0,
    }
}
MONTHS = {
    '01': [1, 31],
    '02': [32, 59],
    '03': [60, 90],
    '04': [91, 120],
    '05': [121, 151],
    '06': [152, 181],
    '07': [182, 212],
    '08': [213, 243],
    '09': [244, 273],
    '10': [274, 304],
    '11': [305, 334],
    '12': [335, 365],
}

TMP_HALF_WINDOW = 45
TMP_SLOPE_WINDOW = 7

PRCP_LEFT_WINDOW = 60
PRCP_RIGHT_WINDOW = 30
PRCP_SLOPE_WINDOW = 7


def read_cycles_weather(f):
    cols = {
        'YEAR': int,
        'DOY': int,
        'PP': float,
        'TX': float,
        'TN': float,
        'SOLAR': float,
        'RHX': float,
        'RHN': float,
        'WIND': float,
    }
    df = pd.read_csv(
        f,
        names=cols.keys(),
        comment='#',
        sep='\s+',
        na_values=[-999],
    )
    df = df.iloc[4:, :]
    df = df.astype(cols)

    return df


def adjust_doy(df, ref_doy):
    df['DOY'] = df.index
    df['DOY'] = df['DOY'].map(lambda x: x + 365 if x < ref_doy else x)
    df = df.set_index('DOY').sort_index()

    return df


def find_month(doy):
    for key, value in MONTHS.items():
        if value[0] <= doy <= value[1]: return key


def moving_avg(arr, left_window, right_window):
    avg = []
    length = len(arr)
    for i in range(right_window):
        arr.append(arr[i])
    arr = np.array(arr)
    for i in range(length):
        avg.append(sum(arr[np.r_[i-left_window:i+right_window+1]]) / float(left_window + right_window + 1))

    return avg


def cal_slope(arr, left_window):
    slope = []
    arr = np.array(arr)
    for i in range(len(arr)):
        if left_window == 0:
            slope.append(arr[i] - arr[i - 1])
        else:
            slope.append(np.polyfit(list(range(left_window + 1)), arr[np.r_[i - left_window:i+1]],1)[0])

    return slope


def select_cultivar(crop, monthly_df):
    if any(monthly_df['tmp'] > float(TMP_MINS[crop])):
        rm, _ = min(HYBRIDS[crop].items(), key=lambda x: abs(monthly_df['thermal_time'].mean() * 365 * 0.85 - x[1]))
        if any(monthly_df['PP'] * 30.0 < 100.0):
            limit = 'tmp' if any(monthly_df['tmp'] < TMP_REF) else 'prcp'
        else:
            limit = 'tmp'
    else:
        rm = ''
        limit = ''

    return rm, limit


def find_doy(df, crop, limit):
    if limit == 'tmp':
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop])) &
                (df['tmp_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop]) - 3.0) &
                (df['tmp_slope'] > 0)
            ].index[0]
        except:
            return np.nan
    else:
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop])) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0) &
                (df['prcp_ma'] > 100.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop])) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0) &
                (df['prcp_ma'] > 80.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop] - 3.0)) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0) &
                (df['prcp_ma'] > 100.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop] - 3.0)) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0) &
                (df['prcp_ma'] > 80.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop])) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop] - 3.0)) &
                (df['tmp_slope'] > 0) &
                (df['prcp_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop])) &
                (df['tmp_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['tmp_ma'] >= float(TMP_MINS[crop] - 3.0)) &
                (df['tmp_slope'] > 0)
            ].index[0]
        except:
            return np.nan


def main(params):
    crop = params['crop']
    lut = params['lut']
    scenario = 'nw_cntrl_03' if lut == 'EOW' else ''
    start_year = params['start']
    end_year = params['end']

    # Read in look-up table or run table
    lookup_df = pd.read_csv(
        LOOKUP(lut, crop),
        index_col=0,
    )

    weathers = lookup_df['Weather'].unique()

    dict = {}
    counter = 0
    for grid in weathers:
        print(len(weathers), counter:= counter + 1, grid)
        dict[grid] = {}

        # Open weather file
        f = f'./input/weather/{scenario}/{scenario}_{grid}.weather' if lut == 'EOW' else f'input/weather/{grid}'
        df = read_cycles_weather(f)

        # Calculate daily average temperature and thermal time
        df['tmp'] = 0.5 * (df['TX'] + df['TN'])  # average temperature
        df['thermal_time'] = df['tmp'].map(lambda x: 0.0 if x < TMP_BASE else x - TMP_BASE)

        # Average to DOY
        df = df.groupby('DOY').mean()
        df['month'] = df.index.map(lambda x: find_month(x))

        # Selecte cultivar and determin if area is temperature or precipitation limited
        rm, limit = select_cultivar(crop, df.groupby('month').mean())
        if len(rm) == 0: continue

        dict[grid]['Crop'] = rm
        dict[grid]['Control'] = limit

        # Calculate moving averages of temperature and precipitation, and their slopes
        df['tmp_smoothed'] = moving_avg(df['tmp'].tolist(), TMP_HALF_WINDOW, TMP_HALF_WINDOW)
        df['tmp_ma'] = moving_avg(df['tmp'].tolist(), 7, 0)
        df['tmp_slope'] = cal_slope(df['tmp_smoothed'].tolist(), TMP_SLOPE_WINDOW)
        df['tmp_slope'] = moving_avg(df['tmp_slope'].tolist(), TMP_HALF_WINDOW, TMP_HALF_WINDOW)

        df['prcp_ma'] = moving_avg(df['PP'].tolist(), 0, 30)
        df['prcp_ma'] = moving_avg(df['prcp_ma'].tolist(), 45, 45)
        df['prcp_slope'] = cal_slope(df['prcp_ma'].tolist(), PRCP_SLOPE_WINDOW)
        df['prcp_slope'] = moving_avg(df['prcp_slope'].tolist(), TMP_HALF_WINDOW, TMP_HALF_WINDOW)

        # Find start of rising temperature
        tmin_doy = df['tmp_smoothed'].idxmin()

        # Adjust DOY to start with lowest temperature
        df_adjusted = adjust_doy(df, tmin_doy)

        # Find doy
        doy = find_doy(df_adjusted, crop, limit)

        if np.isnan(doy):
            dict[grid]['Crop'] = np.nan
            continue

        doy = doy - 365 if doy > 365 else doy
        dict[grid]['pd'] = doy

    output_df = pd.DataFrame(dict).T    #.dropna(how='all')

    df = lookup_df.join(output_df, on='Weather')
    df = df[df['Crop'].notna()]
    df.drop(columns=[
        'NAME_0',
        'NAME_1',
        'NAME_2',
        'Lat',
        'Lon',
        'CropLat',
        'CropLon',
    ]).to_csv(f'./data/{crop}_{lut.lower()}_runs.csv')


def _main():
    parser = argparse.ArgumentParser(description='Cycles execution for a crop')
    parser.add_argument(
        '--crop',
        default='maize',
        help='Crop to be simulated',
    )
    parser.add_argument(
        '--lut',
        default='global',
        choices=['global', 'CONUS', 'EOW', 'test'],
        help='Look-up table to be used',
    )
    parser.add_argument(
        '--start',
        required=True,
        type=int,
        help='Start year of simulation (use 0001 for EOW simulations)',
    )
    parser.add_argument(
        '--end',
        required=True,
        type=int,
        help='End year of simulation (use 0019 for EOW simulations)',
    )
    args = parser.parse_args()

    main(vars(args))


if __name__ == '__main__':
    _main()
