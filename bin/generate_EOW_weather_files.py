#!/usr/bin/env python3

import argparse
import math
import numpy as np
import os
from datetime import datetime
from netCDF4 import Dataset

DATA_DIR = '/storage/home/yzs123/scratch/Toon_group_CLM_output'
CNTRL = 'nw_cntrl_03'
YEARS = {
    'nw_cntrl_03': (1, 21),
    'nw_cntrl_03m02': (1, 20),
    'nw_cntrl_03m03': (1, 19),
    'nw_targets_01': (5, 19),
    'nw_targets_02': (5, 19),
    'nw_targets_03': (5, 19),
    'nw_targets_04': (5, 19),
    'nw_targets_05': (5, 19),
    'nw_ur_150_07': (5, 19),
}


def read_grids():
    # Open netCDF topography file to read in the grid
    with Dataset(f'{DATA_DIR}/consistent-topo-fv1.9x2.5_c130424.nc') as nc:
        lats, lons = np.meshgrid(nc['lat'][:], nc['lon'][:], indexing='ij')
        lats = lats.flatten()
        lons = lons.flatten()
        sgp = nc['PHIS'][:][:].flatten()
        landfrac = nc['LANDFRAC'][:][:].flatten()

    grids = []

    for k in range(landfrac.size):
        if landfrac[k] > 0.0: grids.append(k)

    return [lats, lons], grids, sgp, landfrac


def get_file_names(scenario, coord, grids):
    lats = coord[0]
    lons = coord[1]

    fns = []
    strs = []
    for k in grids:
        # Get lat/lon and elevation
        grid_lat = lats[k]
        grid_lon = lons[k]

        _grid = '%.1f%sx%.1f%s' % (
            abs(grid_lat),
            'S' if grid_lat < 0.0 else 'N',
            360.0 - grid_lon if grid_lon > 180.0 else grid_lon,
            'W' if grid_lon > 180.0 else 'E',
        )

        fns.append(f'{scenario}/{scenario}_{_grid}.weather')
        strs.append([])

    return fns, strs


def elevation(sgp, phi):
    '''Calculate grid elevation from surface geopotential
    http://stcorp.github.io/harp/doc/html/algorithms/derivations/geopotential_height.html
    '''
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))

    g = 9.78032533591 * (1.0 + 0.00193185265241 * sin_phi**2) / math.sqrt(1.0 - 0.00669437999013 * sin_phi**2)
    r = 1.0 / math.sqrt((cos_phi / 6356752.0)**2 + (sin_phi / 6378137.0)**2)

    return sgp * r / (g * r - sgp)


def init_weather_files(scenario, coord, landfrac, strs, grids, sgp):

    os.makedirs(scenario, exist_ok=True)

    lats = coord[0]

    for kgrid in range(len(grids)):
        grid_lat = lats[grids[kgrid]]

        strs[kgrid].append('# Land fraction %.2f\n' % (landfrac[grids[kgrid]]))
        strs[kgrid].append('%-23s\t%.2f\n' % ('LATITUDE', grid_lat))
        strs[kgrid].append('%-23s\t%.2f\n' % ('ALTITUDE', elevation(sgp[grids[kgrid]], grid_lat)))
        strs[kgrid].append('%-23s\t%.1f\n' % ('SCREENING_HEIGHT', 2.0))
        strs[kgrid].append("%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n" %
            ("YEAR", "DOY", "PP", "TX", "TN", "SOLAR", "RHX", "RHN", "WIND"))
        strs[kgrid].append("%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%-7s\t%s\n" %
        ("####", "###", "mm", "degC", "degC", "MJ/m2", "%", "%", "m/s"))


def read_var(t, grids, nc):
    prcp = (
        nc['a2x3h_Faxa_rainc'][t].flatten()[grids] +
        nc['a2x3h_Faxa_rainl'][t].flatten()[grids] +
        nc['a2x3h_Faxa_snowc'][t].flatten()[grids] +
        nc['a2x3h_Faxa_snowl'][t].flatten()[grids]
    )
    tmp  = nc['a2x3h_Sa_tbot'][t].flatten()[grids]
    uwind = nc['a2x3h_Sa_u'][t].flatten()[grids]
    vwind = nc['a2x3h_Sa_v'][t].flatten()[grids]
    solar = (
        nc['a2x3h_Faxa_swndr'][t].flatten()[grids] +
        nc['a2x3h_Faxa_swvdr'][t].flatten()[grids] +
        nc['a2x3h_Faxa_swndf'][t].flatten()[grids] +
        nc['a2x3h_Faxa_swvdf'][t].flatten()[grids]
    )
    pres  = nc['a2x3h_Sa_pbot'][t].flatten()[grids]
    spfh  = nc['a2x3h_Sa_shum'][t].flatten()[grids]

    es = 611.2 * np.exp(17.67 * (tmp - 273.15) / (tmp - 273.15 + 243.5))
    ws = 0.622 * es / (pres - es)
    w = spfh / (1.0 - spfh)
    rh = w / ws
    rh = np.clip(rh, 0.01, 1.0)
    wind = np.sqrt(uwind ** 2 + vwind **2)

    var = {
        'PRCP': np.array(prcp),     # kg m-2 s-1
        'TMP': np.array(tmp),       # K
        'WIND': np.array(wind),     # m s-1
        'SOLAR': np.array(solar),   # W m-2
        'RH': np.array(rh),         # -
    }

    return var


def satvp(temp):
    return 0.6108 * math.exp(17.27 * temp / (temp + 237.3))


def ea(patm, q):
    return patm * q / (0.622 * (1.0 - q) + q)


def process_day(nc, year, grids, strs):
    '''Process one day of data and convert it to Cycles input
    '''
    var = {
        'PRCP': [],
        'TMP': [],
        'WIND': [],
        'SOLAR': [],
        'RH': [],
    }

    for t in range(8): # 3-hourly
        _var = read_var(t, grids, nc)

        for key in var:
            var[key].append(_var[key])

    doy = math.floor(nc['time'][0]) % 365 + 1

    prcp = np.array(var['PRCP']).mean(axis=0) * 86400.0
    tx = np.array(var['TMP']).max(axis=0) - 273.15
    tn = np.array(var['TMP']).min(axis=0) - 273.15
    wind = np.array(var['WIND']).mean(axis=0)
    solar = np.array(var['SOLAR']).mean(axis=0) * 86400.0 / 1.0E6
    rhx = np.array(var['RH']).max(axis=0) * 100.0
    rhn = np.array(var['RH']).min(axis=0) * 100.0

    for kgrid in range(len(grids)):
        strs[kgrid].append(f'%-7.4d\t%-7.3d\t{"%-#.5g" if prcp[kgrid] >= 1.0 else "%-.4f"}\t%-7.2f\t%-7.2f\t%-7.3f\t%-7.2f\t%-7.2f\t%-8.2f\n' % (
            year,
            doy,
            prcp[kgrid],
            tx[kgrid],
            tn[kgrid],
            solar[kgrid],
            rhx[kgrid],
            rhn[kgrid],
            wind[kgrid],
        ))


def write_to_files(fns, strs):
    for k in range(len(fns)):
        with open(fns[k], 'w') as f:
            f.writelines(strs[k])


def main(scenario):

    coord, grids, sgp, landfrac = read_grids()

    fns, strs = get_file_names(scenario, coord, grids)

    init_weather_files(scenario, coord, landfrac, strs, grids, sgp)

    # Generate a list of netCDF files
    for y in range(1, YEARS[scenario][1] + 1):
        s = scenario if y >= YEARS[scenario][0] else CNTRL
        for d in range(1, 366):
            print(y, d)
            f = f'{DATA_DIR}/%s/%s.cpl.ha2x3h.%4.4d-%s.nc' % (
                s,
                s,
                y,
                datetime.strptime('2001-%d' % (d), '%Y-%j').strftime('%m-%d'),
            )


            with Dataset(f, 'r') as nc:
                process_day(nc, y, grids, strs)

    write_to_files(fns, strs)


def _main():
    parser = argparse.ArgumentParser(description='Generate Cycles weather files from Toon group CLM output')
    parser.add_argument(
        "--scenario",
        choices=list(YEARS.keys()),
        required=True,
        help='Scenario',
    )
    args = parser.parse_args()

    main(vars(args)['scenario'])


if __name__ == '__main__':
    _main()
