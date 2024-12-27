"""
Author: Mark Lundine
Takes coastseg outputs
(specifically the transect_timeseries.csv or transect_timeseries_tidally_corrected_matrix.csv)
and makes a geojson that can be used to construct a web map
depicting linear shoreline change rates with linked csvs and figures of particular
transect data. Each transect's length is proportional to 100 years of shoreline growth/retreat
at the computed linear rate. The direction is shoreward if the computed rate is negative
and seaward if the computed rate is positive.

Some places to edit would lie within the get_trend() and plot_timeseries() functions.
More thought should be put into how each timeseries is sampled/processed.
In some cases, their are big gaps in time as well as obvious outliers due to faulty
shoreline delineation, due to noise in the source images (clouds, shadows, data gaps, etc).

A cool addtion would be a function that takes the geojson from get_trends() and
constructs the map in an open-source format.
"""
# load modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import datetime
import shapely
from math import degrees, atan2, radians
from scipy import stats

def add_north_arrow(ax, north_arrow_params):
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)
    
def gb(x1, y1, x2, y2):
    angle = degrees(atan2(y2 - y1, x2 - x1))
    return angle

def arr_to_LineString(coords):
    """
    Makes a line feature from a list of xy tuples
    inputs: coords
    outputs: line
    """
    points = [None]*len(coords)
    i=0
    for xy in coords:
        points[i] = shapely.geometry.Point(xy)
        i=i+1
    line = shapely.geometry.LineString(points)
    return line

def get_trend(filter_df,
              trend_plot_path):
    """
    LLS on single transect timeseries
    inputs:
    filter_df (pandas DataFrame): two columns, dates and cross-shore positions
    trend_plot_path (str): path to save plot to
    outputs:
    lls_result: all the lls results (slope, intercept, stderr, intercept_stderr, rvalue)
    """
    
    filtered_datetimes = np.array(filter_df['dates'])
    shore_pos = np.array(filter_df['pos'])
    datetimes_seconds = [None]*len(filtered_datetimes)
    initial_time = filtered_datetimes[0]
    for i in range(len(filter_df)):
        t = filter_df['dates'].iloc[i]
        dt = t-initial_time
        dt_sec = dt.total_seconds()
        datetimes_seconds[i] = dt_sec
    datetimes_seconds = np.array(datetimes_seconds)
    datetimes_years = datetimes_seconds/(60*60*24*365)
    x = datetimes_years
    y = shore_pos
    lls_result = stats.linregress(x,y)
    slope = lls_result.slope
    intercept = lls_result.intercept
    r_value = lls_result.rvalue**2
    intercept_err = lls_result.intercept_stderr
    slope_err = lls_result.stderr
    lab = ('OLS\nSlope: ' +
          str(np.round(slope,decimals=3)) + ' $+/-$ ' + str(np.round(slope_err, decimals=3)) +
          '\nIntercept: ' +
          str(np.round(intercept,decimals=3)) + ' $+/-$ ' + str(np.round(intercept_err, decimals=3)) +
          '\n$R^2$: ' + str(np.round(r_value,decimals=3)))
    fitx = np.linspace(min(x),max(x),len(x))
    fity = slope*fitx + intercept

    plt.rcParams["figure.figsize"] = (16,6)
    plt.plot(x, y, '--o', color = 'k', label='Raw')
    plt.plot(fitx, fity, '--', color = 'red', label = lab)
    plt.xlim(min(x), max(x))
    plt.ylim(np.nanmin(y), np.nanmax(np.concatenate((y,fity))))
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(trend_plot_path,dpi=300)
    plt.close()

    
    return lls_result

def plot_timeseries(filter_df,
                    timeseries_plot_path):
    """
    Makes and saves a plot of the timeseries
    inputs:
    filter_df (pandas DataFrame): two columns, dates and cross-shore positions
    timeseries_plot_path (str): path to save figure to
    outputs:
    nothing
    """
    plt.rcParams["figure.figsize"] = (16,6)
    x = filter_df['dates']
    y = filter_df['pos']
    plt.plot(x, y,  '--o', color='k', label='Raw')
    plt.xlabel('Time (UTC')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(x), max(x))
    plt.ylim(np.nanmin(y), np.nanmax(y))
    plt.minorticks_on()
    plt.legend()
    plt.tight_layout()
    plt.savefig(timeseries_plot_path,dpi=300)
    plt.close()
    
def get_trends(transect_timeseries_path,
               config_gdf_path,
               t_min,
               t_max):
    """
    Computes linear trends with LLS on each transect's timeseries data
    Saves geojson linking transect id's to trend values
    inputs:
    transect_timeseries (str): path to the transect_timeseries csv (or transect_timeseries_tidally_corrected_matrix.csv)
    config_gdf_path (str): path to the config_gdf (.geojson), it's assumed these are in WGS84
    outputs:
    save_path (str): path to geojson with adjusted transects (in WGS84), trends, csv path, timeseries plot path, trend plot path
    """

    ##Load in data
    timeseries_data = pd.read_csv(transect_timeseries_path)
    timeseries_data['dates'] = pd.to_datetime(timeseries_data['dates'], format='%Y-%m-%d %H:%M:%S+00:00')
    config_gdf = gpd.read_file(config_gdf_path)
    transects = config_gdf[config_gdf['type']=='transect']

    ##Make new directories
    home = os.path.dirname(transect_timeseries_path)
    save_path = os.path.join(home, os.path.splitext(os.path.basename(config_gdf_path))[0]+'_trends.geojson')
    timeseries_csv_dir = os.path.join(home, 'timeseries_csvs')
    timeseries_plot_dir = os.path.join(home, 'timeseries_plots')
    trend_plot_dir = os.path.join(home, 'timeseries_trends_plots')
    dirs = [timeseries_csv_dir, timeseries_plot_dir, trend_plot_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    
    ##For each transect, compute LLS, make plots, make csvs
    slopes = np.empty(len(transects))
    slopes[:] = np.nan
    intercepts = np.empty(len(transects))
    intercepts[:] = np.nan
    r_squares = np.empty(len(transects))
    r_squares[:] = np.nan
    slope_uncertainties = np.empty(len(transects))
    slope_uncertainties[:] = np.nan
    intercept_uncertainties = np.empty(len(transects))
    intercept_uncertainties[:] = np.nan
    timeseries_csvs = [None]*len(transects)
    timeseries_plot_paths = [None]*len(transects)
    trend_plot_paths = [None]*len(transects)
    for i in range(len(slopes)):
        transect_id = transects['id'].iloc[i]
        timeseries_csv_path = os.path.join(timeseries_csv_dir, transect_id+'.csv')
        timeseries_plot_path = os.path.join(timeseries_plot_dir, transect_id+'_timeseries.png')
        trend_plot_path = os.path.join(trend_plot_dir, transect_id+'_timeseries_trend.png')
        
        x = timeseries_data['dates']
        try:
            y = timeseries_data[transect_id]
        except:
            i=i+1
            continue
        df = pd.DataFrame({'dates':x,
                           'pos':y
                           })
        filter_df = df[df['dates']>datetime.datetime.strptime(t_min, '%Y-%m-%d %H:%M:%S+00:00')]
        filter_df = df[df['dates']<datetime.datetime.strptime(t_max, '%Y-%m-%d %H:%M:%S+00:00')]
        filter_df.reset_index()
        filter_df = filter_df.dropna(how='any')
        filter_df.to_csv(timeseries_csv_path)
        
        lls_result = get_trend(filter_df, trend_plot_path)
        plot_timeseries(filter_df, timeseries_plot_path)
        
        slopes[i] = lls_result.slope
        intercepts[i] = lls_result.intercept
        r_squares[i] = lls_result.rvalue**2
        slope_uncertainties[i] = lls_result.stderr
        intercept_uncertainties[i] = lls_result.intercept_stderr
        timeseries_csvs[i] = timeseries_csv_path
        timeseries_plot_paths[i] = timeseries_plot_path
        trend_plot_paths[i] = trend_plot_path

    ###Making the vector file with trends
    skip_idx = np.isnan(slopes)
    slopes = slopes[~np.isnan(slopes)]
    transect_ids = [None]*len(slopes)
    intercepts = intercepts[~np.isnan(intercepts)]
    r_squares = r_squares[~np.isnan(r_squares)]
    slope_uncertainties = slope_uncertainties[~np.isnan(slope_uncertainties)]
    intercept_uncertainties = intercept_uncertainties[~np.isnan(intercept_uncertainties)]
    timeseries_csvs = [ ele for ele in timeseries_csvs if ele is not None ]
    timeseries_plot_paths = [ ele for ele in timeseries_plot_paths if ele is not None ]
    trend_plot_paths = [ ele for ele in trend_plot_paths if ele is not None ]
    max_slope = np.max(np.abs(slopes))
    scaled_slopes = (np.array(slopes)/max_slope)*100
    new_lines = [None]*len(slopes)
    org_crs = transects.crs
    utm_crs = transects.estimate_utm_crs()
    transects_utm = transects.to_crs(utm_crs)
    j=0
    for i in range(len(transects)):
        transect = transects_utm.iloc[i]
        if skip_idx[i]:
            i=i+1
            continue
        else:
            transect_id = transect['id']
            transect_ids[j] = transect_id
            first = transect.geometry.coords[0]
            last = transect.geometry.coords[1]
            midpoint = transect.geometry.centroid
            distance = scaled_slopes[j]
            if distance<0:
                angle = radians(gb(first[0], first[1], last[0], last[1])+180)
            else:
                angle = radians(gb(first[0], first[1], last[0], last[1]))
            northing = midpoint.y + abs(distance)*np.sin(angle)
            easting = midpoint.x + abs(distance)*np.cos(angle)
            line_arr = [(midpoint.x,midpoint.y),(easting,northing)]
            line = arr_to_LineString(line_arr)
            new_lines[j] = line
            j=j+1

    ##This file can be used to link figures and csvs in a web-based GIS map
    new_df = pd.DataFrame({'id':transect_ids,
                           'linear_trend':slopes,
                           'linear_trend_unc':slope_uncertainties,
                           'intercept':intercepts,
                           'intercept_unc':intercept_uncertainties,
                           'timeseries_csv':timeseries_csvs,
                           'timeseries_plot':timeseries_plot_paths,
                           'trend_plot':trend_plot_paths
                           })
    new_geo_df = gpd.GeoDataFrame(new_df, crs=utm_crs, geometry=new_lines)
    new_geo_df_org_crs = new_geo_df.to_crs(org_crs)
    new_geo_df_org_crs.to_file(save_path)
    return save_path

def plot_trend_maps(transect_trends_geojson,
                    site,
                    north_arrow_parms=(0.15, 0.93, 0.2),
                    scale_bar_loc='upper left'):
    """
    Uses contextily and geopandas plotting to plot the trends on a map
    inputs:
    transect_trends_geojson (str): path to the transect with trends
                                   output from get_trends()
    site (str): site name
    north_arrow_params (tuple): (x, y, arrow_length), need to play with this for different locations
    scale_bar_loc (str): position for scale bar, need to play with this for different locations
    returns:
    None
    """
    transect_trends_gdf = gpd.read_file(transect_trends_geojson)
    transect_trends_gdf = transect_trends_gdf.to_crs('3857')
    ax = transect_trends_gdf.plot(column='linear_trend',
                                  legend=True,
                                  legend_kwds={'label':'Trend (m/year'}
                                  cmap='RdBu',
                                  )
    ax.set_title(site)
    cx.add_basemap(ax,
                   source=cx.providers.CartoDB.DarkMatter
                   attribution=False
                   )
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1,
                           location=scale_bar_loc
                           )
                  )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(transect_trends_geojson),
                             site+'_trend_map.png'),
                dpi=500)
    plt.close('all')
    





