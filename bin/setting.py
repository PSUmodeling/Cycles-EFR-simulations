VERSION = "1.0.post"

DATABASE_VERSION = 3.2
LOOKUP_TABLE = lambda lut, crop: f'./data/{crop}_rainfed_{lut.lower()}_lookup_{DATABASE_VERSION}.csv'

RUN_FILE = lambda lut, crop: f'./data/{crop}_{lut.lower()}_runs.csv'
SUMMARY_FILE = lambda lut, scenario, crop, adaption: f'summary/{lut.lower()}_{scenario}_{crop}{"_adaption" if adaption else ""}.csv' if lut == 'EOW' else f'summary/{lut.lower()}_{crop}.csv'
WATER_SUMMARY_FILE = lambda lut, scenario, crop, adaption: f'summary/{lut.lower()}_{scenario}_{crop}_water{"_adaption" if adaption else ""}.csv' if lut == 'EOW' else f'summary/{lut.lower()}_{crop}.csv'

SCENARIOS = [
    'nw_cntrl_03',
    'nw_targets_01',
    'nw_targets_02',
    'nw_targets_03',
    'nw_targets_04',
    'nw_targets_05',
    'nw_ur_150_07',
]

CONTROL_SCENARIO = 'nw_cntrl_03'

CROPS = {
    'maize': {
        'maximum_temperature': '-999',
        'minimum_temperature': '12.0',
        'base_temperature': 6.0,
        'reference_temperature': 10.0,
        'hybrids': {
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
        },
    },
    'springwheat': {
        'maximum_temperature': '-999',
        'minimum_temperature': '5.0',
    },
    'winterwheat': {
        'maximum_temperature': '15.0',
        'minimum_temperature': '-999',
    }
}

INJECTION_YEAR = 5

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
    '12': [335, 366],
}

MOVING_AVERAGE_HALF_WINDOW = 45
SLOPE_WINDOW = 7
DAYS_IN_MONTH = 30
DAYS_IN_WEEK = 7

CYCLES = './bin/Cycles'

RM_CYCLES_IO = 'rm -fr input/*.ctrl input/*.operation input/*.soil output/*'
