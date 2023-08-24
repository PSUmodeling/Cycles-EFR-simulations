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
from setting import SCENARIOS
from setting import CROPS
from setting import CYCLES
from setting import RM_CYCLES_IO


def generate_cycles_input(gid, crop, soil, weather, start_year, end_year, tmp_max, tmp_min, planting_date):
    with open(f'data/template.operation') as op_file:
        op_src = Template(op_file.read())

    with open(f'data/template.ctrl') as ctrl_file:
        ctrl_src = Template(ctrl_file.read())

    op_data = {
        'doy_start': max(1, int(planting_date) - 7),
        'doy_end': min(365, int(planting_date) + 21),
        'max_tmp': tmp_max,
        'min_tmp': tmp_min,
        'crop': crop,
    }
    result = op_src.substitute(op_data)

    with open(f'./input/{gid}.operation', 'w') as f: f.write(result + '\n')

    ### Create control file
    ctrl_data = {
        'start': f'{start_year:04}',
        'end': f'{end_year:04}',
        'operation': f'{gid}.operation',
        'soil': f'soil/{soil}',
        'weather': f'weather/{weather}',
    }
    result = ctrl_src.substitute(ctrl_data)
    with open(f'./input/{gid}.ctrl', 'w') as f: f.write(result + '\n')


def run_cycles(simulation):
    cmd = [
        CYCLES,
        '-s',
        simulation,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    return result.returncode


def main(params):
    lut = params['lut']
    scenario = params['scenario']
    crop = params['crop']
    start_year = params['start']
    end_year = params['end']

    os.makedirs('summary', exist_ok=True)

    tmp_max = CROPS[crop]['maximum_temperature']
    tmp_min = CROPS[crop]['minimum_temperature']

    fn = SUMMARY_FILE(lut, scenario, crop)

    # Read in look-up table or run table
    with open(RUN_FILE(lut, crop)) as f:
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
            weather = f'{scenario}/{scenario}_{row["Weather"]}.weather' if lut == 'EOW' else row['Weather']
            soil = row['Soil']

            print(
                f'{gid} - [{weather}, {soil}] - ',
                end=''
            )

            planting_date = row['pd']
            crop_rm = row['Crop']

            # Run Cycles with spin-up
            generate_cycles_input(gid, crop_rm, soil, weather, start_year, end_year, tmp_max, tmp_min, planting_date)
            run_cycles(gid)

            try:
                # Return season file with best yield
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
