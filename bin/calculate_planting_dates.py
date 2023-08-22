#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from setting import LOOKUP_TABLE, CROPS, MONTHS, MOVING_AVERAGE_HALF_WINDOW, SLOPE_WINDOW, DAYS_IN_MONTH, DAYS_IN_WEEK

"""Calculate planting dates and selecting hybrids
"""
def read_cycles_weather(f):
    columns = {
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
        names=columns.keys(),
        comment='#',
        sep='\s+',
        na_values=[-999],
    )
    df = df.iloc[4:, :]
    df = df.astype(columns)

    return df


def adjust_doy(df, ref_doy):
    df['DOY'] = df.index
    df['DOY'] = df['DOY'].map(lambda x: x + 365 if x < ref_doy else x)
    df = df.set_index('DOY').sort_index()

    return df


def find_month(doy):
    for key, value in MONTHS.items():
        if value[0] <= doy <= value[1]: return key


def moving_average(arr, left_window, right_window):
    avg = []
    length = len(arr)
    for i in range(right_window):
        arr.append(arr[i])
    arr = np.array(arr)
    for i in range(length):
        avg.append(sum(arr[np.r_[i-left_window:i+right_window+1]]) / float(left_window + right_window + 1))

    return avg


def calculate_slope(arr, left_window):
    slope = []
    arr = np.array(arr)
    for i in range(len(arr)):
        if left_window == 0:
            slope.append(arr[i] - arr[i - 1])
        else:
            slope.append(np.polyfit(list(range(left_window + 1)), arr[np.r_[i - left_window:i+1]],1)[0])

    return slope


def select_hybrid(crop, monthly_df):
    if any(monthly_df['temperature'] > float(CROPS[crop]['minimum_temperature'])):
        hybrid, _ = min(CROPS[crop]['hybrids'].items(), key=lambda x: abs(monthly_df['thermal_time'].mean() * 365 * 0.85 - x[1]))
        if any(monthly_df['PP'] * 30.0 < 100.0):
            limit = 'temperature' if any(monthly_df['temperature'] < CROPS[crop]['reference_temperature']) else 'precipitation'
        else:
            limit = 'temperature'
    else:
        hybrid = ''
        limit = ''

    return hybrid, limit


def find_doy(df, crop, limit):
    minimum_temperature = float(CROPS[crop]['minimum_temperature'])
    if limit == 'temperature':
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature - 3.0) &
                (df['temperature_slope'] > 0)
            ].index[0]
        except:
            return np.nan
    else:
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0) &
                (df['precipitation_moving_average'] > 100.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0) &
                (df['precipitation_moving_average'] > 80.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0) &
                (df['precipitation_moving_average'] > 100.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0) &
                (df['precipitation_moving_average'] > 80.0 / 31.0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0) &
                (df['precipitation_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0)
            ].index[0]
        except:
            pass
        try:
            return df[
                (df['temperature_moving_average'] >= minimum_temperature) &
                (df['temperature_slope'] > 0)
            ].index[0]
        except:
            return np.nan


def main(params):
    crop = params['crop']
    lookup_table = params['lut']
    scenario = 'nw_cntrl_03' if lookup_table == 'EOW' else ''
    start_year = params['start']
    end_year = params['end']

    # Read in look-up table or run table
    lookup_df = pd.read_csv(
        LOOKUP_TABLE(lookup_table, crop),
        index_col=0,
    )

    weathers = lookup_df['Weather'].unique()

    dict = {}
    counter = 0
    for grid in weathers:
        print(len(weathers), counter:= counter + 1, grid)
        dict[grid] = {}

        # Open weather file
        f = f'./input/weather/{scenario}/{scenario}_{grid}.weather' if lookup_table == 'EOW' else f'input/weather/{grid}'
        df = read_cycles_weather(f)

        # Calculate daily average temperature and thermal time
        df['temperature'] = 0.5 * (df['TX'] + df['TN'])     # average temperature
        df['thermal_time'] = df['temperature'].map(lambda x: 0.0 if x < CROPS[crop]['base_temperature'] else x - CROPS[crop]['base_temperature'])   # thermal time

        # Average to DOY
        df = df.groupby('DOY').mean()
        df['month'] = df.index.map(lambda x: find_month(x))

        # Select hybrid and determine if area is temperature or precipitation limited
        hybrid, limit = select_hybrid(crop, df.groupby('month').mean())
        if len(hybrid) == 0: continue

        dict[grid]['Crop'] = hybrid
        dict[grid]['Control'] = limit

        # Calculate moving averages of temperature and precipitation, and their slopes
        df['temperature_smoothed'] = moving_average(df['temperature'].tolist(), MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)
        df['temperature_moving_average'] = moving_average(df['temperature'].tolist(), DAYS_IN_WEEK, 0)
        df['temperature_slope'] = calculate_slope(df['temperature_smoothed'].tolist(), SLOPE_WINDOW)
        df['temperature_slope'] = moving_average(df['temperature_slope'].tolist(), MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)

        df['precipitation_moving_average'] = moving_average(df['PP'].tolist(), 0, DAYS_IN_MONTH)
        df['precipitation_moving_average'] = moving_average(df['precipitation_moving_average'].tolist(), MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)
        df['precipitation_slope'] = calculate_slope(df['precipitation_moving_average'].tolist(), SLOPE_WINDOW)
        df['precipitation_slope'] = moving_average(df['precipitation_slope'].tolist(), MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW)

        # Adjust DOY to start with lowest temperature
        df_adjusted = adjust_doy(df, df['temperature_smoothed'].idxmin())

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
    ]).to_csv(f'./data/{crop}_{lookup_table.lower()}_runs.csv')


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
