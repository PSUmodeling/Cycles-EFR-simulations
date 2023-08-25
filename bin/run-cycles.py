#!/usr/bin/env python3

"""Run Cycles simulations for different crops under different nuclear war scenarios

Run Cycles simulations
"""
import argparse
import csv
import os
import pandas as pd
import subprocess
from string import Template
from setting import RUN_FILE
from setting import SUMMARY_FILE
from setting import YEARS
from setting import INJECTION_YEAR
from setting import SCENARIOS
from setting import CROPS
from setting import CYCLES
from setting import RM_CYCLES_IO


def generate_operation_file(gid, hybrids, end_year, max_temperature, min_temperature, planting_date):
    with open(f'data/template.operation') as f:
        operation_file_template = Template(f.read())

    result = ''
    for y in range(end_year):
        operation_data = {
            'year': y + 1,
            'doy_start': max(1, int(planting_date) - 7),
            'doy_end': min(365, int(planting_date) + 90),
            'max_tmp': max_temperature,
            'min_tmp': min_temperature,
            'crop': hybrids[y],
        }
        result += operation_file_template.substitute(operation_data) + '\n'

    with open(f'./input/{gid}.operation', 'w') as f: f.write(result)


def generate_control_file(gid, soil, weather, start_year, end_year):
    with open(f'data/template.ctrl') as f:
        control_file_template = Template(f.read())

    control_data = {
        'start': f'{start_year:04}',
        'end': f'{end_year:04}',
        'rotation_size': end_year - start_year + 1,
        'operation': f'{gid}.operation',
        'soil': soil,
        'weather': f'weather/{weather}',
    }
    result = control_file_template.substitute(control_data)
    with open(f'./input/{gid}.ctrl', 'w') as f: f.write(result + '\n')


def run_cycles(simulation, spin_up=False):
    cmd = [CYCLES, '-s', simulation] if spin_up else [CYCLES, simulation]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode


def main(params):
    lookup_table = params['lut']
    scenario = params['scenario']
    crop = params['crop']
    start_year = 1
    end_year = YEARS

    os.makedirs('summary', exist_ok=True)

    max_temperature = CROPS[crop]['maximum_temperature']
    min_temperature = CROPS[crop]['minimum_temperature']

    fn = SUMMARY_FILE(lookup_table, scenario, crop)

    # Read in look-up table or run table
    with open(RUN_FILE(lookup_table, crop)) as f:
        reader = csv.reader(f, delimiter=',')

        headers = next(reader)
        data = [{h:x for (h,x) in zip(headers,row)} for row in reader]

    first = True

    counter = 0
    with open(fn, 'w') as output_fp:
        # Run each region
        for row in data:
            if not row: continue    # Skip empty lines

            gid = row['GID']
            weather = f'{scenario}/{scenario}_{row["Weather"]}.weather' if lookup_table == 'EOW' else row['Weather']
            soil = row['Soil']

            print(
                f'{gid} - [{weather}, {soil}] - ',
                end=''
            )

            planting_date = row['pd']
            hybrids = [row[f'{scenario}_{y:04}'] for y in range(start_year, end_year + 1)]

            # Run Cycles spin-up
            generate_operation_file(gid, hybrids, end_year, max_temperature, min_temperature, planting_date)
            generate_control_file(gid, f'soil/{soil}', weather, start_year, INJECTION_YEAR - 1)
            run_cycles(gid, spin_up=True)

            generate_control_file(gid, f'{gid}_ss.soil', weather, start_year, end_year)
            run_cycles(gid)

            try:
                df = pd.read_csv(
                    f'output/{gid}/season.txt',
                    sep='\t',
                    header=0,
                    skiprows=[1],
                    skipinitialspace=True,
                )
                df = df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'))
                df['crop'] = df['crop'].str.strip()
                df.insert(0, 'gid', gid)
                df.insert(1, 'area_km2', row['AreaKm2'])
                df.insert(2, 'area_fraction', row['AreaFraction'])

                print('Success')

                if first:
                    strs = df.to_csv(index=False)
                    first = False
                else:
                    strs = df.to_csv(header=False, index=False)

                output_fp.write(''.join(strs))
            except:
                print('Cycles errors')

            # Remove generated input/output files
            subprocess.run(
                RM_CYCLES_IO,
                shell='True',
            )

            counter += 1
            #if counter == 3: break


def _main():
    parser = argparse.ArgumentParser(description='Cycles execution for a crop')
    parser.add_argument(
        '--crop',
        default='maize',
        choices=CROPS,
        help='Crop to be simulated',
    )
    parser.add_argument(
        '--lut',
        default='global',
        choices=['global', 'CONUS', 'EOW', 'test'],
        help='Look-up table to be used',
    )
    parser.add_argument(
        '--scenario',
        default='nw_cntrl_03',
        choices=SCENARIOS,
        help='EOW NW scenario',
    )
    args = parser.parse_args()

    main(vars(args))


if __name__ == '__main__':
    _main()
