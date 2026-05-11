import os

# Simulation runtime
SIMULATION_DURATION = 60
TIME_STEP = 10
OUTPUT_DIRECTORY = 'output/'
LOGGING_LEVEL = 'INFO'
MACRO_PERIOD_SECONDS = 30
ENABLE_ROUTE_OPTIMIZATION = False

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAFFIC_GRID_FILE = os.path.join(DATA_DIR, 'Internet_area.xlsx')
GROUND_STATION_FILE = os.path.join(DATA_DIR, 'GroundStation.xls')

# Physical constants
EARTH_RADIUS_KM = 6371.0
MAX_SLANT_RANGE_KM = 1200.0
SPEED_OF_LIGHT_KM_S = 299792.458
SUN_DISTANCE_KM = 149597870.7

# Satellite energy
CHARGE_RATE_1HZ = 0.00022
DISCHARGE_RATE_1HZ = 0.00044
MIN_ENERGY_LIMIT = 0.2

# Inter-satellite links
MAX_LINK_SPEED_GBPS = 10.0
MAX_LINKS_PER_SATELLITE = 4
LINK_STABILITY = 0.95

# Satellite baseline capability
DEFAULT_STORAGE = 1000.0
DEFAULT_COMPUTING_POWER = 1e12

# Background traffic
TRAFFIC_SCALE_FACTOR = 0.195
TRAFFIC_GRID_ROWS = 12
TRAFFIC_GRID_COLS = 24

# Time-of-day traffic factor
DAYTIME_START_HOUR = 6.0
DAYTIME_PEAK_START_HOUR = 10.0
DAYTIME_PEAK_END_HOUR = 22.0
DAYTIME_END_HOUR = 24.0
MIN_TIME_FACTOR = 0.05
MAX_TIME_FACTOR = 1.0

# Constellations: tle file, height(km), inclination, planes, sats/plane, phase factor.
CONSTELLATION_CONFIGS = {
    '3600': ('Com_qx4_60p60_TLE.txt', 500, 58, 60, 60, 23),
    '432': ('Delta_18p24_50_1150_5.txt', 1150, 50, 24, 18, 5),
}
SELECTED_CONSTELLATION_ID = '3600'


def apply_constellation_config(constellation_id: str):
    global SELECTED_CONSTELLATION_ID
    global CONSTELLATION_ID, TLE_FILE_PATH, ORBIT_HEIGHT, ORBIT_HEIGHT_KM
    global ORBIT_INCLINATION, NUM_ORBIT_PLANES, SATS_PER_PLANE, PHASE_FACTOR

    constellation_id = str(constellation_id)
    if constellation_id not in CONSTELLATION_CONFIGS:
        raise ValueError(f"Unknown constellation_id={constellation_id}")

    tle_name, height, inclination, planes, sats_per_plane, phase = CONSTELLATION_CONFIGS[
        constellation_id
    ]
    SELECTED_CONSTELLATION_ID = constellation_id
    CONSTELLATION_ID = constellation_id
    TLE_FILE_PATH = os.path.join(DATA_DIR, tle_name)
    ORBIT_HEIGHT = height
    ORBIT_HEIGHT_KM = height
    ORBIT_INCLINATION = inclination
    NUM_ORBIT_PLANES = planes
    SATS_PER_PLANE = sats_per_plane
    PHASE_FACTOR = phase


apply_constellation_config(SELECTED_CONSTELLATION_ID)
