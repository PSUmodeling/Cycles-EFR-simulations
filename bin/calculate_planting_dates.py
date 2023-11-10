#!/usr/bin/env python3

"""Calculate planting dates and selecting hybrids
"""
import argparse
import itertools
import math
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

def pet(doy, lat, pres, screen_height, t_max, t_min, sol_rad, rh_max, rh_min, wind):
    """Calculates potential ET using Penman Monteith
    """
    RC = 0.0        # surface resistance to vapor flow (day/m) (0.0081 for 350-400 ppm)
    SCP = 0.001013  # specific heat of dry air (J/kg/C)
    RGAS = 0.28704  # kg/degree C) (note that pres is in kPa)

    t_avg = (t_max + t_min) / 2.0
    es_tavg = saturation_vapor_pressure(t_avg)
    es_tmax = saturation_vapor_pressure(t_max)
    es_tmin = saturation_vapor_pressure(t_min)
    ea = 0.5 * (es_tmin * rh_max + es_tmax * rh_min) / 100.0
    vpd = (es_tmax + es_tmin) / 2.0 - ea
    pot_rad = potential_radiation(doy, lat)
    net_rad = net_radiation(pot_rad, sol_rad, ea, t_max, t_min)

    # Aerodynamic resistance to vapor flow (day/m)
    ra = aerodynamic_resistance(wind, screen_height)

    # Slope of saturated vapor pressure vs temperature function (kPa/C)
    delta = 4098.0 * es_tavg / ((t_avg + 237.3) * (t_avg + 237.3))

    # Latent heat of vaporization (MJ/kg)
    l = 2.501 - 0.002361 * t_avg

    # Psychrometric constant (kPaC)
    gamma = SCP * pres / (0.622 * l)

    # Approximates virtual temperature (K)
    tkv = 1.01 * (t_avg + 273.15)

    # SCP * AirDensity (J/kg * kg/m3)
    vol_cp = SCP * pres / (RGAS * tkv)

    # Aerodynamic term (MJ/m2)
    aero_term = (vol_cp * vpd / ra) / (delta + gamma * (1.0 + RC / ra))

    # Radiation term (MJ/m2)
    rad_term = delta * net_rad / (delta + gamma * (1.0 + RC / ra))

    # Potential ET (kg water/m2 or mm) (water density = 1 Mg/m3)
    pmet = (aero_term + rad_term) / l
    #pmet = MAX(pmet, 0.001)     # Preventing a negative value usually small and indicative of condensation

    #// CO2 correction factor
    #*co2_adjust_transp = (delta + gamma * (1.0 + RC / ra)) /
    #    (delta + gamma * (1.0 + RC / (0.32 * (1.0 + 4.9 * exp(-0.0024 * co2))) / ra));

    return pmet


def saturation_vapor_pressure(t):
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))


def potential_radiation(doy, lat):
    SOLAR_CONST = 118.08

    lat_rad = lat * math.pi / 180.0
    dr = 1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0)
    sol_dec = 0.409 * math.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    sunset_hour_angle = math.acos(-math.tan(lat_rad) * math.tan(sol_dec))
    term = sunset_hour_angle * math.sin(lat_rad) * math.sin(sol_dec) + math.cos(lat_rad) * math.cos(sol_dec) * math.sin(sunset_hour_angle)

    return SOLAR_CONST * dr * term / math.pi


def net_radiation(pot_rad, solar_rad, vpa, t_max, t_min):
    ALBEDO = 0.23

    # Calculate shortwave net radiation
    rns = (1.0 - ALBEDO) * solar_rad

    # Calculate cloud factor
    f_cloud = 1.35 * (solar_rad / (pot_rad * 0.75)) - 0.35

    # Calculate humidity factor
    f_hum = (0.34 - 0.14 * math.sqrt(vpa))

    # Calculate isothermal LW net radiation
    lwr = 86400.0 * 5.67E-8 * 0.5 * (pow(t_max + 273.15, 4.0) + pow(t_min + 273.15, 4.0)) / 1.0E6

    rnl = lwr * f_cloud * f_hum

    return rns - rnl


def aerodynamic_resistance(uz, z):
    VK = 0.41   # von Karman's constant

    uz = 1.0E-5 if uz == 0.0 else uz
    u2 = uz if z == 2.0 else uz * (4.87 / (math.log(67.8 * z - 5.42)))
    u2 *= 86400.0   # Convert to m/day
    d = 0.08
    zom = 0.01476
    zoh = 0.001476
    zm = 2.0
    zh = 2.0

    r_a = math.log((zm - d) / zom) * math.log((zh - d) / zoh) / (VK * VK * u2)

    return r_a


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

    with open(f) as file:
        lines = [line.strip() for line in file.readlines() if not line.startswith('#')][:3]

    for line in lines:
        df[line.split()[0]] = float(line.split()[1])

    df['PRESSURE'] = df['ALTITUDE'].map(lambda x: 101.325 * math.exp(-x / 8200.0))

    df = df.reset_index()

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
        if lookup_table == 'EOW':
            weather_df = pd.DataFrame()
            for s in CONTROL_SCENARIO:
                _df = read_cycles_weather(f'./input/weather/{s}/{s}_{grid}.weather', start_year, end_year)
                weather_df = pd.concat([weather_df, _df])
            weather_df = weather_df.groupby(level=0).mean()
            print(weather_df)
        else:
            weather_df = read_cycles_weather(f'input/weather/{grid}', start_year, end_year)

        # Calculate daily average temperature and thermal time
        weather_df['temperature'] = 0.5 * (weather_df['TX'] + weather_df['TN'])
        weather_df['pet'] = weather_df.apply(lambda x: pet(x['DOY'], x['LATITUDE'], x['PRESSURE'], x['SCREENING_HEIGHT'], x['TX'], x['TN'], x['SOLAR'], x['RHX'], x['RHN'], x['WIND']), axis=1)
        print(weather_df)
        exit()
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
                for m in s:
                    if m in CONTROL_SCENARIO:
                        for y in range(start_year, end_year + 1):
                            dict[grid][f'{m}_{"%4.4d" % y}'] = hybrid
                    else:
                        f = f'./input/weather/{m}/{m}_{grid}.weather'
                        weather_df = read_cycles_weather(f, start_year, end_year)
                        weather_df['temperature'] = 0.5 * (weather_df['TX'] + weather_df['TN'])
                        weather_df['thermal_time'] = weather_df['temperature'].map(lambda x: 0.0 if x < CROPS[crop]['base_temperature'] else x - CROPS[crop]['base_temperature'])

                        for y in range(start_year, end_year + 1):
                            dict[grid][f'{m}_{"%4.4d" % y}'] = hybrid if y <= INJECTION_YEAR else select_hybrid(crop, weather_df[weather_df['YEAR'] == y - 1]['thermal_time'].mean() * 365)
        else:
            dict[grid]['crop'] = hybrid

    output_df = lookup_df.join(pd.DataFrame(dict).T, on='Weather')
    output_df = output_df[output_df['Control'].notna()]
    output_df['pd'] = output_df['pd'].astype(int)

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
