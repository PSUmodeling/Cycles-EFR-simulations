#!/usr/bin/env python3

"""Calculate planting dates and selecting hybrids
"""
import argparse
import itertools
import numpy as np
import pandas as pd
from setting import LOOKUP_TABLE
from setting import RUN_FILE
from setting import SCENARIOS
from setting import CONTROL_SCENARIO
from setting import INJECTION_YEAR
from setting import CROPS
from setting import MONTHS
from setting import MOVING_AVERAGE_HALF_WINDOW
from setting import SLOPE_WINDOW
from setting import DAYS_IN_MONTH
from setting import DAYS_IN_WEEK

def read_cycles_weather(f, start_year=0, end_year=9999):
    NUM_HEADER_LINES = 4
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
    df = df.iloc[NUM_HEADER_LINES:, :]
    df = df.astype(columns)
    df = df[(df['YEAR'] <= end_year) & (df['YEAR'] >= start_year)]

    return df


def adjust_doy(df, ref_doy):
    df['DOY'] = df.index
    df['DOY'] = df['DOY'].map(lambda x: x + 365 if x < ref_doy else x)
    df = df.set_index('DOY').sort_index()

    return df


def find_month(doy):
    for key, value in MONTHS.items():
        if value[0] <= doy <= value[1]: return key


def moving_average(array, left_window, right_window):
    avg_array = []
    length = len(array)
    for i in range(right_window):
        array.append(array[i])
    array = np.array(array)
    for i in range(length):
        avg_array.append(sum(array[np.r_[i-left_window : i+right_window+1]]) / float(left_window + right_window + 1))

    return avg_array


def calculate_slope(array, left_window):
    slope = []
    array = np.array(array)
    for i in range(len(array)):
        if left_window == 0:
            slope.append(array[i] - array[i - 1])
        else:
            slope.append(np.polyfit(list(range(left_window + 1)), array[np.r_[i - left_window:i+1]],1)[0])

    return slope


def select_hybrid(crop, thermal_time):
    hybrid, _ = min(CROPS[crop]['hybrids'].items(), key=lambda x: abs(thermal_time * 0.85 - x[1]))

    return hybrid


def calculate_planting_date(crop, limit, weather_df, temperature_levels, precipitation_conditions):
    # Calculate moving averages of temperature and precipitation, and their slopes
    df = weather_df.groupby('DOY').mean()
    df['temperature_smoothed'] = moving_average(
        df['temperature'].tolist(),
        MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW,
    )
    df['temperature_moving_average'] = moving_average(
        df['temperature'].tolist(),
        DAYS_IN_WEEK, 0,
    )
    df['temperature_slope'] = calculate_slope(
        df['temperature_smoothed'].tolist(),
        SLOPE_WINDOW,
    )
    df['temperature_slope'] = moving_average(
        df['temperature_slope'].tolist(),
        MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW,
    )

    df['precipitation_moving_average'] = moving_average(
        df['PP'].tolist(),
        0, DAYS_IN_MONTH,
    )
    df['precipitation_moving_average'] = moving_average(
        df['precipitation_moving_average'].tolist(),
        MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW,
    )
    df['precipitation_slope'] = calculate_slope(
        df['precipitation_moving_average'].tolist(),
        SLOPE_WINDOW,
    )
    df['precipitation_slope'] = moving_average(
        df['precipitation_slope'].tolist(),
        MOVING_AVERAGE_HALF_WINDOW, MOVING_AVERAGE_HALF_WINDOW,
    )

    # Adjust DOY to start with coldest day
    df = adjust_doy(df, df['temperature_smoothed'].idxmin())

    # Calculate planting date
    ## Temperature limited planting date
    if limit == 'temperature':
        for temperature in temperature_levels:
            try:
                return df[
                    (df['temperature_moving_average'] >= temperature) &
                    (df['temperature_slope'] > 0)
                ].index[0]
            except:
                pass
        else:
            return np.nan
    ## Precipitation limited planting date
    else:
        for temperature, precipitation in precipitation_conditions:
            try:
                return df[
                    (df['temperature_moving_average'] >= temperature) &
                    (df['temperature_slope'] > 0) &
                    (df['precipitation_slope'] > 0) &
                    (df['precipitation_moving_average'] > precipitation)
                ].index[0]
            except:
                pass
        else:
            for temperature in temperature_levels:
                try:
                    return df[
                        (df['temperature_moving_average'] >= temperature) &
                        (df['temperature_slope'] > 0)
                    ].index[0]
                except:
                    pass
            else:
                return np.nan


def main(params):
    crop = params['crop']
    lookup_table = params['lut']
    scenario = CONTROL_SCENARIO if lookup_table == 'EOW' else ''
    start_year = params['start']
    end_year = params['end']

    # Read in look-up table or run table
    lookup_df = pd.read_csv(
        LOOKUP_TABLE(lookup_table, crop),
        index_col=0,
    )

    weathers = lookup_df['Weather'].unique()

    # Create a list of temperature and precipitation conditions for planting date calculation
    minimum_temperature = float(CROPS[crop]['minimum_temperature'])
    temperature_levels = [
        minimum_temperature + 3.0,
        minimum_temperature + 2.0,
        minimum_temperature + 1.0,
        minimum_temperature,
    ]

    precipitation_conditions = list(itertools.product(temperature_levels, [100 / 30.0, 80 / 30.0]))
    precipitation_conditions += list(itertools.product(temperature_levels, [60 / 30.0, 40 / 30.0]))
    precipitation_conditions += list(itertools.product(temperature_levels, [0]))

    dict = {}
    counter = 0
    for grid in weathers:
        print(len(weathers), counter:= counter + 1, grid)

        # Open weather file
        f = f'./input/weather/{scenario}/{scenario}_{grid}.weather' if lookup_table == 'EOW' else f'input/weather/{grid}'
        weather_df = read_cycles_weather(f, start_year, end_year)

        # Calculate daily average temperature and thermal time
        weather_df['temperature'] = 0.5 * (weather_df['TX'] + weather_df['TN'])
        weather_df['thermal_time'] = weather_df['temperature'].map(lambda x: 0.0 if x < CROPS[crop]['base_temperature'] else x - CROPS[crop]['base_temperature'])
        weather_df['month'] = weather_df['DOY'].map(lambda x: find_month(x))

        monthly_df = weather_df.groupby('month').mean()
        if all(monthly_df['temperature'] < float(CROPS[crop]['minimum_temperature'])):
            continue

        if any(monthly_df['PP'] * 30.0 < 100.0):
            limit = 'temperature' if any(monthly_df['temperature'] < CROPS[crop]['reference_temperature']) else 'precipitation'
        else:
            limit = 'temperature'

        doy = calculate_planting_date(crop, limit, weather_df, temperature_levels, precipitation_conditions)
        if np.isnan(doy):
            continue

        dict[grid] = {
            'Control': limit,
        }

        doy = doy - 365 if doy > 365 else doy
        dict[grid]['pd'] = doy

        # Select hybrid
        hybrid = select_hybrid(crop, monthly_df['thermal_time'].mean() * 365)

        if lookup_table == 'EOW':
            for s in SCENARIOS:
                if s == CONTROL_SCENARIO:
                    for y in range(start_year, end_year + 1):
                        dict[grid][f'{s}_{"%4.4d" % y}'] = hybrid
                else:
                    f = f'./input/weather/{s}/{s}_{grid}.weather'
                    weather_df = read_cycles_weather(f, start_year, end_year)
                    weather_df['temperature'] = 0.5 * (weather_df['TX'] + weather_df['TN'])
                    weather_df['thermal_time'] = weather_df['temperature'].map(lambda x: 0.0 if x < CROPS[crop]['base_temperature'] else x - CROPS[crop]['base_temperature'])

                    for y in range(start_year, end_year + 1):
                        dict[grid][f'{s}_{"%4.4d" % y}'] = hybrid if y <= INJECTION_YEAR else select_hybrid(crop, weather_df[weather_df['YEAR'] == y - 1]['thermal_time'].mean() * 365)
        else:
            dict[grid]['crop'] = hybrid

    output_df = lookup_df.join(pd.DataFrame(dict).T, on='Weather')
    output_df = output_df[output_df['Control'].notna()]

    output_df.drop(columns=[
        'NAME_0',
        'NAME_1',
        'NAME_2',
        'Lat',
        'Lon',
        'CropLat',
        'CropLon',
    ]).to_csv(RUN_FILE(lookup_table, crop))


def _main():
    parser = argparse.ArgumentParser(description='Cycles execution for a crop')
    parser.add_argument(
        '--crop',
        default='maize',
        help='Crop to be simulated',
    )
    parser.add_argument(
        '--lut',
        default='EOW',
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
