import os

# Simulation runtime
SIMULATION_DURATION = 60
TIME_STEP = 5
OUTPUT_DIRECTORY = 'output/'
LOGGING_LEVEL = 'INFO'
MICRO_PERIOD_SECONDS = 5
MACRO_PERIOD_SECONDS = 30
ENABLE_ROUTE_OPTIMIZATION = False

LINK_AVAILABILITY_THRESHOLD = 0.9
LINK_PREDICTION_BATCH_SIZE = 4096
LINK_PREDICTION_TIMING_LOG = True

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

TRAFFIC_GRID_FILE = os.path.join(DATA_DIR, 'Internet_area.xlsx')
GROUND_STATION_FILE = os.path.join(DATA_DIR, 'GroundStation.xls')

# Physical constants
EARTH_RADIUS_KM = 6371.0
SPEED_OF_LIGHT_KM_S = 299792.458
SUN_DISTANCE_KM = 149597870.7

# Ground-station visibility
min_elevation_deg = 5

# Satellite energy
ENERGY_RATE_V_1HZ = 0.00022
DISCHARGE_RATE_1HZ = ENERGY_RATE_V_1HZ
CHARGE_RATE_1HZ = 2 * ENERGY_RATE_V_1HZ
MIN_ENERGY_LIMIT = 0.2

# Inter-satellite links
MAX_LINK_SPEED_GBPS = 100.0
MAX_LINKS_PER_SATELLITE = 4
LINK_STABILITY = 0.95

# Source-Destination pair constraint (based on lat/lon difference)
MIN_SRC_DST_LAT_DIFF_DEG = 20.0   # 源-目的卫星对的最小纬度差阈值（度）
MIN_SRC_DST_LON_DIFF_DEG = 20.0   # 源-目的卫星对的最小经度差阈值（度）

# Satellite node load
MAX_NODE_FLOW_GBPS = 400.0

# Satellite baseline capability
DEFAULT_STORAGE = 1000.0
DEFAULT_COMPUTING_POWER = 1e12

TRAFFIC_GRID_ROWS = 12
TRAFFIC_GRID_COLS = 24

# Time-of-day traffic factor
DAYTIME_START_HOUR = 5.0
DAYTIME_PEAK_START_HOUR = 10.0
DAYTIME_PEAK_END_HOUR = 22.0
DAYTIME_END_HOUR = 24.0
MIN_TIME_FACTOR = 0.2
MAX_TIME_FACTOR = 1.0

# Constellations: tle file, height(km), inclination, planes, sats/plane, phase factor,
# max_slant_range_km, traffic_scale_factor, min_src_dst_lat_diff_deg, min_src_dst_lon_diff_deg, link_availability_threshold
CONSTELLATION_CONFIGS = {
    '3600': ('Delta_60_60_58_500_23.txt', 500, 58, 60, 60, 23, 2000.0, 5, 20.0, 20.0, 0.9),
    '432': ('Delta_24_18_50_1150_5.txt', 1150, 50, 24, 18, 5, 4000.0, 5, 40.0, 40.0, 0.8),
}
SELECTED_CONSTELLATION_ID = '3600'


def apply_constellation_config(constellation_id: str):
    global SELECTED_CONSTELLATION_ID
    global CONSTELLATION_ID, TLE_FILE_PATH, ORBIT_HEIGHT, ORBIT_HEIGHT_KM
    global ORBIT_INCLINATION, NUM_ORBIT_PLANES, SATS_PER_PLANE, PHASE_FACTOR
    global MAX_SLANT_RANGE_KM, TRAFFIC_SCALE_FACTOR
    global MIN_SRC_DST_LAT_DIFF_DEG, MIN_SRC_DST_LON_DIFF_DEG
    global LINK_AVAILABILITY_THRESHOLD

    constellation_id = str(constellation_id)
    if constellation_id not in CONSTELLATION_CONFIGS:
        raise ValueError(f"Unknown constellation_id={constellation_id}")

    (
        tle_name,
        height,
        inclination,
        planes,
        sats_per_plane,
        phase,
        max_slant_range,
        traffic_scale,
        min_src_dst_lat_diff,
        min_src_dst_lon_diff,
        link_availability_threshold,
    ) = CONSTELLATION_CONFIGS[constellation_id]
    SELECTED_CONSTELLATION_ID = constellation_id
    CONSTELLATION_ID = constellation_id
    TLE_FILE_PATH = os.path.join(DATA_DIR, tle_name)
    ORBIT_HEIGHT = height
    ORBIT_HEIGHT_KM = height
    ORBIT_INCLINATION = inclination
    NUM_ORBIT_PLANES = planes
    SATS_PER_PLANE = sats_per_plane
    PHASE_FACTOR = phase
    MAX_SLANT_RANGE_KM = max_slant_range
    TRAFFIC_SCALE_FACTOR = traffic_scale
    MIN_SRC_DST_LAT_DIFF_DEG = min_src_dst_lat_diff
    MIN_SRC_DST_LON_DIFF_DEG = min_src_dst_lon_diff
    LINK_AVAILABILITY_THRESHOLD = link_availability_threshold

apply_constellation_config(SELECTED_CONSTELLATION_ID)
