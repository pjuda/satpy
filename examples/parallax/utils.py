import pickle
import sys
import time
from datetime import datetime, timedelta
from glob import glob
from typing import List, Tuple

import matplotlib
import numpy
import pandas
import pyproj
import xarray
from config import _MTG_TIME_COORD, MTG_PROD_PATTERN, MTG_VARS, CTH_valid_range, wgs84_PROJ4
from decorators import timer
from pyorbital.orbital import A as EARTH_RADIUS
from pyorbital.orbital import get_observer_look

from satpy import Scene
from satpy.utils import lonlat2xyz, xyz2lonlat


def process_datetime_to_read_file(start_dt:str, end_dt:str) -> tuple:
    '''
    Process the times to match the 10 minutes resolution of MTG folders.

    Args:
        start_dt: start time of the data to be extracted
        end_dt: end time of the data to be extracted

    returns: (datetime, datetime)
        start_dt_floored: the start time floored to 10 minutes before
        end_dt_time: the end time ceiled  to 10 minutes above
     '''
    start_dt = datetime.strptime(start_dt, '%Y-%m%d-%H:%M:%S')
    end_dt = datetime.strptime(end_dt, '%Y-%m%d-%H:%M:%S')

    #seconds don't matter for reading files
    start_dt = start_dt.replace(second= 0)
    end_dt = end_dt.replace(second=0)
    print(f'time of interest {start_dt} - {end_dt}')

    #floor
    minute = (start_dt.minute // 10) * 10
    start_dt_floored = start_dt.replace(minute=minute)

    #ceil
    minute = (end_dt.minute // 10)* 10
    end_dt_ceiled = end_dt.replace(minute = minute)+ timedelta(minutes = 10)

    return (start_dt_floored, end_dt_ceiled)

def get_MTG_filenames(path_to_folder_li: str, start_time:str, end_time:str, MTG_product:str, return_initial_time_range = False) -> list:
    '''
    Find all the filenames corresponding to a given period over which to extract data

    Args:
        path_to_folder_li: path to MTG data
        start_time: time from which extracting data ('%Y-%m%d-%H:%M:%S')
        end_time : time to which extracting data ('%Y-%m%d-%H:%M:%S')
        return_initial_time_range (Optional): to retrieve the hours where data is missing

    Returns:
        time_range : to be given to open_dataset function of utils
        f_li : str of file names, can be given to SatPY or open_dataset function of utils
    '''
    #Round times to match filenames:
    start_read_f, end_read_f = process_datetime_to_read_file(start_dt=start_time, end_dt=end_time)

    time_range = pandas.date_range(start_read_f, end_read_f, freq = timedelta(minutes = 10))

    print(f'Getting file names for the period:  {start_read_f} to {end_read_f}')
    print('---- Product: ', MTG_product)

    #Buffer for file names
    f_li = ['' for i in range(len(time_range)) ]
    filtered_time_range = time_range

    for file_idx, nc_time in enumerate(time_range):

        time_pattern = 'EUMT_' + datetime.strftime(nc_time, '%Y%m%d%H%M') + '*'

        file_id = MTG_PROD_PATTERN[MTG_product] + time_pattern
        list_of_file = glob(str(path_to_folder_li) + file_id)


        try: f_li[file_idx] = list_of_file[0] #the list contains one filename (expected):

        except(IndexError): #missing data
            print(time_pattern)
            print(f'WARNING:  no data available for end time {nc_time}')
            filtered_time_range = filtered_time_range[filtered_time_range != nc_time]

    #Filter out missing timesteps
    f_li = [mtg_filename for mtg_filename in f_li if mtg_filename != '']
    if return_initial_time_range:
        return (filtered_time_range, f_li, time_range)
    else:
        return(filtered_time_range, f_li)

@timer
def open_dataset(path_to_folder_li: str, start_time: str, end_time: str, MTG_product: str) -> pandas.DataFrame:

    """
    Search for the files containing the MTG_product of interest for the given time period.
    Open the files and merge the .nc files to one pandas dataframe covering the entire period.

    Args:
        path_to_folder: path to the location where to look for the EUMETSAT data folders
        start_time: start of the time period in the form '%Y-%m%d-%H%M' : only 10 minutes invrements valid
        end_time: end of the time period in the form '%Y-%m%d-%H%M'
        MTG_product: specify which lightning product ('LGR', 'LFL', ...)

    Returns:
        df_concat (pandas dataframe): Lightning product stored in one table. Each row being one group or flash etc.

    """
    time_range, f_li = get_MTG_filenames(path_to_folder_li, start_time, end_time, MTG_product)

    #Buffer for nc files converted to dataframes
    nc_dataframes = [xarray.Dataset()]*(len(time_range))

    start_datetime = datetime.strptime(start_time, '%Y-%m%d-%H:%M:%S')
    end_datetime = datetime.strptime(end_time, '%Y-%m%d-%H:%M:%S')

    for file_idx, nc_time in enumerate(time_range): #Reads up to a file of the next ten minutes to avoid missing data

        #open file and store in buffer nc_datasets:
        nc_dataset = xarray.open_dataset(f_li[file_idx])
        #Store each dataframe in nc_dataframe
        nc_dataframes[file_idx] = nc_dataset[MTG_VARS[MTG_product]].to_dataframe()

    # -- 3 Merge the datasets in one panda dataframe
    df_concat = pandas.concat(nc_dataframes)
    df_concat = df_concat[(df_concat[_MTG_TIME_COORD[MTG_product]] >= start_datetime) & (df_concat[_MTG_TIME_COORD[MTG_product]] < end_datetime)]
    print(' Data merged into a pandas dataframe: ')
    print(df_concat)

    return df_concat

def remove_timesteps_with_missing_groups(LFL_data:pandas.DataFrame, path_to_folder_li:str, start_time:str, end_time:str) -> pandas.DataFrame:
    ''' Remove from LFL data timesteps that are missing in LGR data.  '''

    #Read LGR filenames to store a list of existing groups
    MTG_product = 'LGR'
    existing_timesteps, lgr_filenames, lgr_initial_time_range = get_MTG_filenames(  path_to_folder_li, start_time, end_time,
                                                                                    MTG_product, return_initial_time_range=True)
    existing_datetime = pandas.to_datetime(existing_timesteps)
    lgr_initial_datetime = pandas.to_datetime(lgr_initial_time_range)

    #initialize cropped data
    LFL_data_cropped = LFL_data
    #LFL_data_cropped[_MTG_TIME_COORD['LFL']] = pandas.to_datetime(LFL_data_cropped[_MTG_TIME_COORD['LFL']])

    for i, ts in enumerate(lgr_initial_time_range[1:-1]):

        if lgr_initial_time_range[i] not in (existing_timesteps):
            print('Deleting data between: ')
            print(f'{lgr_initial_datetime[i-1] } to {lgr_initial_datetime[i]}')

            LFL_data_cropped = LFL_data_cropped[~(  (LFL_data_cropped[_MTG_TIME_COORD['LFL']]>= lgr_initial_time_range[i-1]) &
                                                (LFL_data_cropped[_MTG_TIME_COORD['LFL']]<= lgr_initial_time_range[i])      )]
    print('Initial number of flashes : ', LFL_data.size)
    print('Final number of flashes: ', LFL_data_cropped.size)
    print(' ---- Number of flashes deleted: ', LFL_data.size-LFL_data_cropped.size)
    return LFL_data_cropped


def get_last_ctth_time(start_time:str) -> str :
    '''Retrieve the nearest 5 minutes timestep preceding mtg li start time of interest to open CTTH data
    Args:
        start_time: date string in the form '%Y-%m%d-%H:%M:%S'
    Return:
        start_time_floored
    '''
    start_dt = datetime.strptime(start_time, '%Y-%m%d-%H:%M:%S')
    minutes_floored = start_dt.minute - (start_dt.minute % 5)
    start_dt_floored = datetime(start_dt.year, start_dt.month, start_dt.day, start_dt.hour, minutes_floored, 0)
    CTTH_time = datetime.strftime(start_dt_floored, '%H%M%S')

    return CTTH_time




def lonlat_to_proj(lon: numpy.ndarray, lat: numpy.ndarray, target_proj: str) -> numpy.ndarray:
    """
    Transform lon, lat to target projection.

    Args:
        lon (numpy.ndarray): 1-D array of longitudes
        lat (numpy.ndarray): 1-D array latitudes
        target_proj (str): proj4-formatted definition of the target projection.

    Returns:
        x (numpy.ndarray): 1-D array of coordinates along x-direction
        y (numpy.ndarray): 1-D array of coordinates along y-direction
    """
    p = pyproj.Proj(target_proj)
    x, y = p(lon, lat)
    return x, y


def proj_to_lonlat(x: numpy.ndarray, y: numpy.ndarray, target_proj: str) -> numpy.ndarray:
    """
    Transform lon, lat to target projection.

    Args:
        lon (numpy.ndarray): 1-D array of x coordinates
        lat (numpy.ndarray): 1-D array y coordinates
        target_proj (str): proj4-formatted definition of the target projection.

    Returns:
        x (numpy.ndarray): 1-D array of longitudes
        y (numpy.ndarray): 1-D array of latitudes
    """
    p = pyproj.Proj(target_proj)
    lon, lat = p(x, y, inverse=True)
    return lon, lat

# Define a formatter function to convert meters to kilometers
def ticks_meter_to_km(x: float, pos) -> str:
    '''Function to be given as argument to FuncFormatter. Defines how the ticks labels should be formatted'''
    return f'{x / 1000:.0f}'  # Convert meters to kilometers

def date_in_plots_name(start_time:datetime, end_time:datetime)-> str:
    '''
    Apply a systematic convention for date and time in output files' names

    Args:
        start_time : start time of the plot
        end_time: end time of the plot

    Returns: a string that can be added to the file's name'''

    date_str = start_time.strftime('%y%m%d')
    start_str = start_time.strftime('%H:%M:%S')
    end_str = end_time.strftime('%H:%M:%S')
    date_and_time = 'mtg_' + date_str + '_' + start_str + '-' + end_str + '_'

    return date_and_time

def add_xy_coords_2_mtg_df(mtg_df:pandas.DataFrame)-> pandas.DataFrame:
    '''Modifiy the datagrame given as argument to add 2 columns for x and y coordinates.

    Args:
        mtg_df: a dataframe of LGR or LFL data    '''

    x, y = lonlat_to_proj(mtg_df.loc[:,'longitude'], mtg_df.loc[:,'latitude'], wgs84_PROJ4) #WGS84

    mtg_df.insert(loc = len(mtg_df.columns), column = 'x', value = x)
    mtg_df.insert(loc = len(mtg_df.columns), column = 'y', value = y)

#######################################################################################################
#Used for the parallax
#######################################################################################################

def time_of_interest_from_prompt():
    if len(sys.argv) > 1:
        timeslot = sys.argv[1]    # inumpyut string --> '2021-11-10 08:00' the time being the start time of the 10 minutes interval
    else:
        print ("Missing timestamp as argument")
        sys.exit()

    start_datetime = datetime.strptime(timeslot,'%Y-%m-%d %H:%M')
    end_datetime = start_datetime + timedelta(minutes = 15)

    print(f'\n----- Open lightning flashes and cloud top height products for time: {start_datetime} to {end_datetime} \n ')

    return (start_datetime, end_datetime)


def initialize_mtg_df_in_target_proj(lon: numpy.array, lat:numpy.array,
                                    flash_time: numpy.array,
                                    flash_id: numpy.array,
                                    target_proj:str, area_extent:Tuple,
                                    start_datetime:datetime, end_datetime:datetime)-> pandas.DataFrame:
    '''
    Creates a Pandas Dataframe based on the arguments given. It stores LFL information: lat,lon and flash time.
    Then it calculates for each lat, lon the coordinates in the target_proj. Finally, filter out data outside of the domain of the area_extent

    Args:
        lon: Longitudes of flashes
        lat: Latitudes of flashes
        flash_time_index: the time of LFL, used later as the index of the dataframe.
        target_proj : a proj4 string like "+proj=eqc +lat_0=46.5 +lon_0=8.25 +lat_ts=46.0 +ellps=WGS84 +units=m +no_defs"

    '''

    x, y = lonlat_to_proj(lon, lat, target_proj) #WGS84

    mtg_df = pandas.DataFrame({ 'longitude': lon,
                                'latitude': lat,
                                'x': x,
                                'y': y,
                                'flash_time': flash_time
                                 }, index = flash_id.data.astype(int))

    #Slice dataframe based on time
    mtg_df_local = mtg_df.loc[(mtg_df['flash_time'] >= start_datetime ) & (mtg_df['flash_time'] < end_datetime )].copy()

    xmin, ymin, xmax, ymax  = area_extent

    mtg_df_local = mtg_df_local.loc[(xmin< mtg_df_local['x']) & (xmax > mtg_df_local['x'] ) & (ymin< mtg_df_local['y']) & (ymax > mtg_df_local['y'] )]

    return mtg_df_local

'''@timer
def handle_missing_heights(cth_dataset, height, x_idx, y_idx, mtg_df_local):

    cth_nan_idx = numpy.where(numpy.isnan(height))
    nan_x_idx_list = x_idx.iloc[cth_nan_idx]
    nan_y_idx_list = y_idx.iloc[cth_nan_idx]

    mtg_df_nan = mtg_df_local[mtg_df_local[['x_idcs', 'y_idcs']].apply(tuple, axis=1).isin(zip(nan_x_idx_list, nan_y_idx_list))]
    print(f'This corresponds to the following flashes: \n{mtg_df_nan} \n ')

    nearest_cth = numpy.zeros(mtg_df_nan.shape[0])
    x_grid_meter, y_grid_meter = numpy.meshgrid(cth_dataset.coords['x'], cth_dataset.coords['y'] )

    lfl_x_coords = mtg_df_nan.loc[:,'x'].to_numpy()[:, None, None]
    lfl_y_coords = mtg_df_nan.loc[:,'y'].to_numpy()[:, None, None]
    euclidian_dist = (x_grid_meter[None, :,:] - lfl_x_coords)**2 + (y_grid_meter[None, :,:] - lfl_y_coords)**2

    valid_euclidian_dist = numpy.where(~numpy.isnan(cth_dataset.data[None, :,:]), euclidian_dist, numpy.nan)
    valid_euclidian_dist = numpy.where(numpy.sqrt(valid_euclidian_dist) <= 50000, valid_euclidian_dist, numpy.nan )

    min_euclidian_dist = numpy.nanmin(valid_euclidian_dist, axis =(1,2)).reshape(-1,1,1)
    cth_idx =  numpy.argwhere(valid_euclidian_dist == min_euclidian_dist)

    nearest_cth = cth_dataset.data[cth_idx[:,1], cth_idx[:,2]]

    height_adjusted = height.copy()
    height_adjusted[numpy.argwhere(numpy.isnan(height))] = nearest_cth.reshape((-1,1))

    return (height_adjusted, mtg_df_nan)
'''
@timer
def handle_missing_heights(cth_dataset, height, x_idx, y_idx, mtg_df_local):

    cth_nan_idx = numpy.where(numpy.isnan(height))
    nan_x_idx_list = x_idx.iloc[cth_nan_idx]
    nan_y_idx_list = y_idx.iloc[cth_nan_idx]

    mtg_df_nan = mtg_df_local[mtg_df_local[['x_idcs', 'y_idcs']].apply(tuple, axis=1).isin(zip(nan_x_idx_list, nan_y_idx_list))]
    print(f'This corresponds to the following flashes: \n{mtg_df_nan} \n ')

    nearest_cth = numpy.zeros(mtg_df_nan.shape[0])
    for i, lfl_id in enumerate(mtg_df_nan.index):

        #compute euclidian distance of lfl with each centroid
        x_grid_meter, y_grid_meter = numpy.meshgrid(cth_dataset.coords['x'], cth_dataset.coords['y'] )
        euclidian_dist = (x_grid_meter - mtg_df_nan.loc[lfl_id, 'x'])**2 + (y_grid_meter - mtg_df_nan.loc[lfl_id, 'y'])**2
        nan_mask = numpy.full(euclidian_dist.shape, numpy.nan)

        valid_euclidian_dist = numpy.where(numpy.isnan(cth_dataset), nan_mask, euclidian_dist)
        valid_euclidian_dist = numpy.where(numpy.sqrt(valid_euclidian_dist) > 50000, nan_mask, valid_euclidian_dist)
        #TODO: Handle the case where no CTH is at max 50km from the flash
        cth_nearest_y_idx, cth_nearest_x_idx =  numpy.unravel_index(numpy.nanargmin(valid_euclidian_dist), valid_euclidian_dist.shape)
        nearest_cth[i] = cth_dataset[cth_nearest_y_idx, cth_nearest_x_idx]


    height_adjusted = height.copy()
    height_adjusted[numpy.argwhere(numpy.isnan(height))] = nearest_cth.reshape((-1,1))

    return (height_adjusted, mtg_df_nan)


def parallax_correct(sat_lon, sat_lat, sat_alt, lon, lat, height):
    '''
    Perform the parallax correction on the given lon and lat of flashes.

    Args:
        sat_lon, sat_lat, sat_alt: satellite's longitude, latitude and altitude
        lon, lat: longitude and latitude of the flashes
        height: cloud top height at the given lat and lon.
    '''
    #retrieve the 3D coordinates in Cartesian frame:
    X_sat = numpy.hstack(lonlat2xyz(sat_lon,sat_lat)) * sat_alt
    X = numpy.stack(lonlat2xyz(lon,lat), axis=-1) * EARTH_RADIUS

    (_, elevation) = get_observer_look(sat_lon, sat_lat, sat_alt, datetime(2000,1,1),
                                    numpy.array([lon]), numpy.array([lat]), EARTH_RADIUS)

    # TODO: handle cases where this could divide by 0
    parallax_distance = height / numpy.sin(numpy.deg2rad(elevation))

    #Xd is the vector leaving from the satellite to the point data of interest
    X_d = X - X_sat

    sat_distance = numpy.sqrt((X_d*X_d).sum(axis=-1))
    dist_shape = X_d.shape[:-1] + (1,) # force correct array broadcasting, parallax_distance's shape is (913,) otherwise
    X_top = X - X_d*(parallax_distance/sat_distance).reshape(dist_shape)

    (corrected_lon, corrected_lat) = xyz2lonlat(
        X_top[...,0],X_top[...,1],X_top[...,2])
    return (corrected_lon, corrected_lat)

def define_cth_colorbar(path_cth_pal):
    with open(path_cth_pal, "rb") as f :
        cth_palette  = pickle.load(f)


    cth_palette = cth_palette.values/255
    vmin, vmax = CTH_valid_range
    cth_cmap = matplotlib.colors.ListedColormap(cth_palette, name="cth_palette")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    return (cth_cmap, norm)
