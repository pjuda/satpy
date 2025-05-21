# Set configuration
import numpy
import os

#TODO should all this dictionaries be written in the form of objects instead?
# variable of interest for each product

MTGDIR= '/home/jup/work/projects/satpy/examples/parallax/mtgdir/'
OUTDIR = '/home/jup/work/projects/satpy/examples/parallax/outdir/'

MTG_VARS = {
    'LGR': ['group_id', 'group_time', 'latitude', 'longitude', 'number_of_events', 'flash_id'],
    'LFL': [ 'flash_id', 'flash_time', 'flash_duration', 'flash_footprint', 'latitude', 'longitude', 'number_of_groups'],
    'AF': ['flash_accumulation', 'accumulation_start_times', 'accumulation_offsets','x', 'y']
            }

MTG_PROD_PATTERN = {
    "LGR": "*MTI1+LI-2-LGR*",
    "LFL": "*MTI1+LI-2-LFL*",
    "AF": "*MTI1+LI-2-AF-*" ,
    "AFA": "*MTI1+LI-2-AFA*"
}

#Useful for reading the multiple files and to concatenate based on their delivery frequency
#Unit: minutes
MTG_PROD_DELIVERY_FREQ = {
    'LGR': 10,
    'LFL': 10,
    'AF': 10
}

# Equidistant cylindrical projection. Coordinates are along parallels and meridians.
# Maybe not the best when comparing surfaces, are there alternatives?
wgs84_PROJ4 = "+proj=eqc +lat_0=46.5 +lon_0=8.25 +lat_ts=46.0 +ellps=WGS84 +units=m +no_defs"

# lon-lat domain by Météorage (see pdf documentation)
MTRG=dict(lon_min=0.16, lon_max=16.75, lat_min=42.67, lat_max=49.73)

RADIUS_OF_INFLUENCE = 5000

PATH_CTH_PAL = "/home/zjequier/Documents/MTG_LI/mtg-li-meteorage-comparison/palette.pkl"
CTH_valid_range = numpy.array([    -2000, 25000], dtype = numpy.float32)
#--------------- Resolution for the grid --------------------------------------
TIME_RES_S = 60
T_MIN = numpy.datetime64('2024-09-01T16:20:00')
T_MAX = numpy.datetime64('2024-09-01T16:30:00')
SHAPE_T = (T_MAX - T_MIN)//numpy.timedelta64(TIME_RES_S,'s')

# 5 km grid
RES_M = 5000
X_MIN=-625000
X_MAX=655000
Y_MIN=-425000
Y_MAX=355000

# 1km grid
#RES_M = 1000
#X_MIN = -625000
#X_MAX = 657000
#Y_MIN = -426000
#Y_MAX = 359000

SHAPE_X = (X_MAX - X_MIN)//RES_M
SHAPE_Y = (Y_MAX - Y_MIN)//RES_M

# Example of coordinates to use in LGR and LFL
_MTG_TIME_COORD =  {
    "LGR": "group_time",
    "LFL": "flash_time",
    "AF": "accumulated_flash_times"
}
