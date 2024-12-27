import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import contextily as cx
import numpy as np
import os
import glob
import datetime
import warnings
import shapely
from math import degrees, atan2, radians
from scipy import stats

warnings.filterwarnings("ignore")

"""TODO FIX COMPARISONS DF AND PLOT TIMESERIES SECTION"""
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 2
##set basemap source
BASEMAP_DARK = cx.providers.CartoDB.DarkMatter ##dark mode for trend maps
BASEMAP_IMAGERY = cx.providers.Esri.WorldImagery ##world imagery for error maps

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
              ):
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
   
    return lls_result
    
def get_trends(transect_timeseries_path,
               config_gdf_path,
               t_min,
               t_max,
               scale):
    """
    Computes linear trends with LLS on each transect's timeseries data
    Saves geojson linking transect id's to trend values
    inputs:
    transect_timeseries (str): path to the transect_timeseries csv (or transect_timeseries_tidally_corrected_matrix.csv)
    config_gdf_path (str): path to the config_gdf (.geojson), it's assumed these are in WGS84
    scale (int): constant to multiply trends by to rescale lines
    outputs:
    save_path (str): path to geojson with adjusted transects (in WGS84), trends, csv path, timeseries plot path, trend plot path
    """

    ##Load in data
    timeseries_data = transect_timeseries_path
    config_gdf = config_gdf_path
    transects = config_gdf[config_gdf['type']=='transect']
    
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
    for i in range(len(slopes)):
        transect_id = transects['id'].iloc[i]
        
        x = timeseries_data['dates']
        try:
            y = timeseries_data[transect_id]
        except:
            i=i+1
            continue
        df = pd.DataFrame({'dates':x,
                           'pos':y
                           })
        filter_df = df[df['dates']>t_min]
        filter_df = df[df['dates']<t_max]
        filter_df.reset_index()
        filter_df = filter_df.dropna(how='any')
        
        lls_result = get_trend(filter_df)
        
        slopes[i] = lls_result.slope
        intercepts[i] = lls_result.intercept
        r_squares[i] = lls_result.rvalue**2
        slope_uncertainties[i] = lls_result.stderr
        intercept_uncertainties[i] = lls_result.intercept_stderr


    ###Making the vector file with trends
    skip_idx = np.isnan(slopes)
    slopes = slopes[~np.isnan(slopes)]
    transect_ids = [None]*len(slopes)
    intercepts = intercepts[~np.isnan(intercepts)]
    r_squares = r_squares[~np.isnan(r_squares)]
    slope_uncertainties = slope_uncertainties[~np.isnan(slope_uncertainties)]
    intercept_uncertainties = intercept_uncertainties[~np.isnan(intercept_uncertainties)]
    max_slope = np.max(np.abs(slopes))
    scaled_slopes = (np.array(slopes))*scale
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
                           })
    new_geo_df = gpd.GeoDataFrame(new_df, crs=utm_crs, geometry=new_lines)
    new_geo_df_org_crs = new_geo_df.to_crs(org_crs)
    #new_geo_df_org_crs.to_file(save_path)
    return new_geo_df_org_crs

def add_north_arrow(ax, north_arrow_params):
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)
    
def remove_nans(arr):
    return arr[~np.isnan(arr)]

def remove_nones(my_list):
    new_list = [x for x in my_list if x is not None]
    return new_list

def in_situ_comparison(home,
                       site,
                       window,
                       plot_timeseries=True,
                       legend_loc=(0.4,0.6),
                       north_arrow_params=(0.05,0.2,0.1),
                       scale_bar_loc='lower left',
                       trend_scale=100):
    """
    compares in situ shoreline measurements with sds measurements from CoastSeg
    will output a number of figures, a csv with the matched up comparisons, and a new transects geojson with linear trends as a column
    outputs are saved to home/site/analysis_outputs
    inputs:
    home (str): path to the home directory, containing a folder with the site
                structure:
                home
                ----site
                --------analysis_ready_data
                ----------------------in_situ
                ---------------------------in_situ_transect_time_series_merged.csv
                ----------------------raw
                ---------------------------raw_transect_time_series_merged.csv
                ----------------------tidally_corrected
                ---------------------------tidally_corrected_transect_time_series_merged.csv
                ----------------------transects
                ---------------------------transects.geojson
    site (str): the name of the site
    window (int): number of days to window the in-situ data for sds comparison
    plot_timeseries (bool): True will plot each timeseries, False will not plot any and just generate error figures
    legend_loc (tuple): coordinates for legend loc in abs errror distribution
    north_arrow_params (tuple): (x,y,arrow_length) for north arrow on maps
    scale_bar_loc (str): location for scale bar on maps
    trend_scale (float): multiplier for plotting trend maps
    returns:
    None
    """
    ##Constants
    ##fig parameter
    DPI = 500
    ##fig extension
    EXT = '.png'
    ##whether or not to make a plot for each timeseries
    PLOT_TIMESERIES = plot_timeseries
    ##window for analysis (comparing SDS vs in-situ within +/- WINDOW days)
    WINDOW = window
    ##SITE NAME
    SITE = site
    ##HOME
    home = home

    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams['lines.markersize'] = 2

    ##Setting up directories and data for analysis
    data_dir = os.path.join(home, site, 'analysis_ready_data')
    analysis_outputs = os.path.join(home, site, 'analysis_outputs')
    try:
        os.mkdir(analysis_outputs)
    except:
        pass


    ##Analysis starts here
    ##loading data
    ##transects_gdf
    transects_path = os.path.join(data_dir, 'transects','transects.geojson')
    transects_gdf = gpd.read_file(transects_path)
    transects_gdf['transect_id'] = transects_gdf['transect_id'].astype(int)

    ##for the datetimes, i'm chopping of the hours, minutes, and seconds so that there not duplicate sds measurements from the same day
    ##in situ
    in_situ_csv = os.path.join(data_dir, 'in_situ', 'in_situ_transect_time_series_merged.csv')
    in_situ_df = pd.read_csv(in_situ_csv)
    in_situ_df['dates'] = pd.to_datetime(in_situ_df['dates'], utc=True).dt.floor('d')

    ##raw
    raw_csv = os.path.join(data_dir, 'raw', 'raw_transect_time_series_merged.csv')
    df_raw = pd.read_csv(raw_csv)
    df_raw['dates'] = pd.to_datetime(df_raw['dates'], utc=True).dt.floor('d')

    ##tidally corrected
    tide_csv = os.path.join(data_dir, 'tidally_corrected', 'tidally_corrected_transect_time_series_merged.csv')
    df_tide = pd.read_csv(tide_csv)
    df_tide['dates'] = pd.to_datetime(df_tide['dates'], utc=True).dt.floor('d')

    ##loop over unique transects in sds data
    transects = sorted(df_raw['transect_id'].unique())

    ##Skipping transects
    skip_transects = [None]

    ##just removing transects at Barter that pick up old ridges
    remove_idxes=[True]*len(transects)
    i=0
    for transect in transects:
        if transect in skip_transects:
            remove_idxes[i]=False
        i=i+1
        
    transects = np.array(transects)

    transects = transects[remove_idxes]
            
    """
    Initializing arrays,
    rmse_ arrays will contain the per-transect RMSE
    mae_ arrays will contain the per-trasnect MAE
    err_ arrays will contain the in-situ vs SDS error for each measurement pair
    square_err_ arrays will contain the squared error for each measurement pair
    """

    ##no corrections
    rmse_raws = [None]*len(transects)
    mae_raws = [None]*len(transects)
    abs_raws = [None]*len(transects)
    err_raws = [None]*len(transects)
    square_err_raws = [None]*len(transects)

    ##low slope
    rmse_tides = [None]*len(transects)
    mae_tides = [None]*len(transects)
    abs_tides = [None]*len(transects)
    err_tides = [None]*len(transects)
    square_err_tides = [None]*len(transects)

    ##Looping over each transect in study area
    comparisons = [None]*len(transects)
    print('Analyzing ' + SITE)
    for j in range(len(transects)):
        transect = transects[j]
        print(transect)
        ##no corrections
        filter_df_raw = df_raw[df_raw['transect_id']==transect]
        filter_df_raw = filter_df_raw.drop_duplicates(subset=['dates'], keep='first').reset_index(drop=True)
        filter_df_raw = filter_df_raw.sort_values(by='dates').reset_index(drop=True)
        
        ##corrections
        filter_df_tide = df_tide[df_tide['transect_id']==transect]
        filter_df_tide = filter_df_tide.drop_duplicates(subset=['dates'], keep='first').reset_index(drop=True)
        filter_df_tide = filter_df_tide.sort_values(by='dates').reset_index(drop=True)

        #in situ
        filter_in_situ_df = in_situ_df[in_situ_df['transect_id']==transect]
        filter_in_situ_df = filter_in_situ_df.drop_duplicates(subset=['dates'], keep='first').reset_index(drop=True)
        filter_in_situ_df = filter_in_situ_df.sort_values(by='dates').reset_index(drop=True)


        ##matching up in-situ with sds,
        ##look at points where sds observation is within 11 days of in-situ observation
        filter_df_raw_merge = [None]*len(filter_in_situ_df)
        filter_df_tide_merge = [None]*len(filter_in_situ_df)

        ##no corrections matching
        if (len(filter_in_situ_df) == 0) or len(filter_df_raw) == 0:
            print('no matches')
            abs_raw = None
            abs_raws[j] = abs_raw
            square_err_raw = None
            square_err_raws[j] = square_err_raw
            rmse_raw = np.nan
            mae_raw = np.nan
            rmse_raws[j] = rmse_raw
            mae_raws[j] = mae_raw
            
            abs_tide = None
            abs_tides[j] = abs_tide
            square_err_tide = None
            square_err_tides[j] = square_err_tide
            rmse_tide = np.nan
            mae_tide = np.nan
            rmse_tides[j] = rmse_tide
            mae_tides[j] = mae_tide             
        else:
            merged_in_time_raw = pd.merge_asof(filter_in_situ_df.rename(columns={'dates':'dates_in_situ'}),
                                               filter_df_raw.rename(columns={'dates':'dates_sds'}),
                                               left_on='dates_in_situ',
                                               right_on='dates_sds',
                                               direction='nearest',
                                               suffixes=['_in_situ', '_sds'],
                                               tolerance=pd.Timedelta(days=WINDOW)).dropna()
            merged_in_time_raw['timedelta'] = merged_in_time_raw['dates_in_situ']-merged_in_time_raw['dates_sds']
            merged_in_time_raw = merged_in_time_raw.sort_values(by='timedelta').reset_index(drop=True)
            merged_in_time_raw = merged_in_time_raw.drop_duplicates(subset=['dates_sds'], keep='first').reset_index(drop=True)
            merged_in_time_raw = merged_in_time_raw.sort_values(by='dates_in_situ').reset_index(drop=True)
                                                                    
            ##low slope matching
            merged_in_time_tide = pd.merge_asof(filter_in_situ_df.rename(columns={'dates':'dates_in_situ'}),
                                               filter_df_tide.rename(columns={'dates':'dates_sds'}),
                                               left_on='dates_in_situ',
                                               right_on='dates_sds',
                                               direction='nearest',
                                               suffixes=['_in_situ', '_sds'],
                                               tolerance=pd.Timedelta(days=WINDOW)).dropna()
            merged_in_time_tide['timedelta'] = merged_in_time_tide['dates_in_situ']-merged_in_time_tide['dates_sds']
            merged_in_time_tide = merged_in_time_tide.sort_values(by='timedelta').reset_index(drop=True)
            merged_in_time_tide = merged_in_time_tide.drop_duplicates(subset=['dates_sds'], keep='first').reset_index(drop=True)
            merged_in_time_tide = merged_in_time_tide.sort_values(by='dates_in_situ').reset_index(drop=True)
            
            if len(merged_in_time_raw)==0:
                print('no matches')
                abs_raw = None
                abs_raws[j] = abs_raw
                square_err_raw = None
                square_err_raws[j] = square_err_raw
                rmse_raw = np.nan
                mae_raw = np.nan
                rmse_raws[j] = rmse_raw
                mae_raws[j] = mae_raw
                
                abs_tide = None
                abs_tides[j] = abs_tide
                square_err_tide = None
                square_err_tides[j] = square_err_tide
                rmse_tide = np.nan
                mae_tide = np.nan
                rmse_tides[j] = rmse_tide
                mae_tides[j] = mae_tide               
            else:
                ##computing rmse for a transect

                ##no corrections
                abs_raw = np.array(np.abs(merged_in_time_raw['cross_distance_in_situ'] - merged_in_time_raw['cross_distance_sds']))
                abs_raws[j] = abs_raw
                square_err_raw = np.array(((merged_in_time_raw['cross_distance_in_situ'] - merged_in_time_raw['cross_distance_sds'])**2))
                square_err_raws[j] = square_err_raw
                rmse_raw = np.sqrt(((merged_in_time_raw['cross_distance_in_situ'] - merged_in_time_raw['cross_distance_sds'])**2).mean())
                try:
                    mae_raw = sum(abs_raw)/len(abs_raw)
                except:
                    mae_raw = np.nan
                rmse_raws[j] = rmse_raw
                mae_raws[j] = mae_raw

                ##low slope
                abs_tide = np.array(np.abs(merged_in_time_tide['cross_distance_in_situ'] - merged_in_time_tide['cross_distance_sds']))
                abs_tides[j] = abs_tide
                square_err_tide = np.array(((merged_in_time_tide['cross_distance_in_situ'] - merged_in_time_tide['cross_distance_sds'])**2))
                square_err_tides[j] = square_err_tide
                rmse_tide = np.sqrt(((merged_in_time_tide['cross_distance_in_situ'] - merged_in_time_tide['cross_distance_sds'])**2).mean())
                try:
                    mae_tide = sum(abs_tide)/len(abs_tide)
                except:
                    mae_tide = np.nan
                rmse_tides[j] = rmse_tide
                mae_tides[j] = mae_tide
                
                ##plotting each timeseries
                lab_raw = ('Raw\nRMSE = ' +str(np.round(rmse_raw,decimals=3)) + ' m' +
                          '\nMAE = ' +str(np.round(mae_raw,decimals=3))+ ' m')
                lab_tide = ('Tidally Corrected\nRMSE = ' +str(np.round(rmse_tide,decimals=3)) + ' m' +
                          '\nMAE = ' +str(np.round(mae_tide,decimals=3))+ ' m')

                ##merging filtered sds data
                merged_in_time_raw = merged_in_time_raw.dropna().reset_index(drop=True)
                merged_in_time_tide = merged_in_time_tide.dropna().reset_index(drop=True)
                merged_in_time = pd.merge(left=merged_in_time_raw,
                                     left_on='dates_sds',
                                     right=merged_in_time_tide,
                                     right_on='dates_sds',
                                     suffixes=['_raw', '_tide'])
                comparisons[j] = merged_in_time
                ##merging unfiltered sds data
                merged_df = pd.merge(left=filter_df_raw,
                                     left_on='dates',
                                     right=filter_df_tide,
                                     right_on='dates',
                                     suffixes=['_raw', '_tide'])
                ##plotting
                if PLOT_TIMESERIES==True:
                    transect_dir = os.path.join(analysis_outputs, 'transect_timeseries')
                    try:
                        os.mkdir(transect_dir)
                    except:
                        pass
                    with plt.rc_context({"figure.figsize":(16,5)}):
                        plt.title('Transect ' + str(transect))


                        timedelta = datetime.timedelta(days=WINDOW)
                        ##plot raw
                        ##plot tide
                        ##plot in situ
                        plt.plot(merged_df['dates'],
                                 merged_df['cross_distance_raw'], '--', color='k', label='SDS Raw')
                        plt.plot(merged_df['dates'],
                                 merged_df['cross_distance_tide'], '--', color='salmon', label='SDS Tidally Corrected')
                        plt.scatter(merged_in_time['dates_sds'],
                                    merged_in_time['cross_distance_sds_tide'],
                                    s=10,
                                    color='red',
                                    label='SDS Observations in Comparison')

                        ##in situ
                        plt.scatter(filter_in_situ_df['dates'],
                                    filter_in_situ_df['cross_distance'],
                                    s=10,
                                    color='lightsteelblue',
                                    label='In Situ')
                        plt.scatter(merged_in_time['dates_in_situ_raw'],
                                    merged_in_time['cross_distance_in_situ_raw'],
                                    s=10,
                                    color='blue',
                                    label='In Situ Observations in Comparison')
                        for i in range(len(merged_in_time)):
                            plt.plot([merged_in_time['dates_in_situ_raw'].iloc[i],
                                      merged_in_time['dates_sds'].iloc[i]],
                                     [merged_in_time['cross_distance_in_situ_raw'].iloc[i],
                                      merged_in_time['cross_distance_sds_tide'].iloc[i]],
                                     color='gray')
                                      

                        plt.ylabel('Cross-Shore Position (m)')
                        plt.xlabel('Time (UTC)')
                        plt.xlim(min(merged_in_time['dates_in_situ_raw'])-timedelta,
                                 max(merged_in_time['dates_in_situ_raw'])+timedelta)
                
                        plt.legend()
                        plt.minorticks_on()
                        plt.tight_layout()
                        plt.savefig(os.path.join(transect_dir, str(transect)+'_timeseries'+EXT), dpi=DPI)
                        plt.close('all')

    comparisons_df = pd.concat(comparisons)
    rem_cols = ['shore_x_raw',
                'shore_y_raw',
                'x_raw',
                'y_raw',
                'transect_id_sds_raw',
                'timedelta_raw',
                'dates_in_situ_tide',
                'transect_id_in_situ_tide',
                'cross_distance_in_situ_tide',
                'shore_x_tide',
                'shore_y_tide',
                'x_tide',
                'y_tide',
                'tide_tide'
                'transect_id_sds_tide'
                ]
    for col in rem_cols:
        try:
            comparisons_df = comparisons_df.drop(columns=[col])
        except:
            pass



    comparisons_df = comparisons_df.rename(columns={'dates_in_situ_raw':'dates_in_situ',
                                                    'dates_sds':'dates_sds',
                                                    'timedelta_tide':'timedelta',
                                                    'transect_id_in_situ_raw':'transect_id',
                                                    'tide_raw':'tide',
                                                    'cross_distance_in_situ_raw':'cross_distance_in_situ',
                                                    'cross_distance_sds_raw':'cross_distance_sds_raw',
                                                    'cross_distance_sds_tide':'cross_distance_sds_tide',
                                                    }
                                           )
    comparisons_df.to_csv(os.path.join(analysis_outputs, SITE+'_compared_obs.csv'))

    rmse_df = pd.DataFrame({'transect_id':transects.astype(int),
                            'rmse_raw':rmse_raws,
                            'rmse_tidally_corrected':rmse_tides}
                           )
    mae_df = pd.DataFrame({'transect_id':transects.astype(int),
                           'mae_raw':mae_raws,
                           'mae_tidally_corrected':mae_tides}
                          )

    ##removing nans, these were in-situ points without an sds observation within 10 days
    rmse_raws = remove_nans(np.array(rmse_raws))
    rmse_tides = remove_nans(np.array(rmse_tides))
    mae_raws = remove_nans(np.array(mae_raws))
    mae_tides = remove_nans(np.array(mae_tides))
                           

    ##this list is used in the box plot later on
    data_rmse = [rmse_raws, rmse_tides]
    data_mae = [mae_raws, mae_tides]

    ##removing nones
    abs_raws_concat = np.concatenate(remove_nones(abs_raws))
    abs_tides_concat = np.concatenate(remove_nones(abs_tides))

    ##removing nones
    square_err_raws_concat = np.concatenate(remove_nones(square_err_raws))
    square_err_tides_concat = np.concatenate(remove_nones(square_err_tides))

    ##iqr of sds values
    iqr_sds_raw = np.nanquantile(df_raw['cross_distance'], 0.75)-np.nanquantile(df_raw['cross_distance'], 0.25)
    iqr_sds_tide = np.nanquantile(df_tide['cross_distance'], 0.75)-np.nanquantile(df_tide['cross_distance'], 0.25)

    ##labels for absolute error distribution plots
    raw_lab = ('Raw\nMAE = ' +
              str(np.round(np.mean(abs_raws_concat), decimals=3)) +
              '\nRMSE = ' + str(np.round(np.sqrt(square_err_raws_concat.mean()), decimals=3)) +
              '\nSDS Position IQR = ' + str(np.round(iqr_sds_raw, decimals=3)) + 
              '\n# of Observations = ' + str(len(abs_raws_concat))
              )
    tide_lab = ('Tidally Corrected\nMAE = ' +
               str(np.round(np.mean(abs_tides_concat), decimals=3)) +
               '\nRMSE = ' + str(np.round(np.sqrt(square_err_tides_concat.mean()), decimals=3)) +
               '\nSDS Position IQR = ' + str(np.round(iqr_sds_tide, decimals=3)) + 
               '\n# of Observations = ' + str(len(abs_tides_concat))
               )

    """
    Plotting absolute error distributions
    """
    with plt.rc_context({"figure.figsize":(12,12)}):
        plt.subplot(2,1,1)
        plt.suptitle(SITE)
        plt.hist(abs_raws_concat, label=raw_lab, color='gray', density=False, bins=np.arange(0,max(abs_raws_concat),1))
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.hist(abs_raws_concat,
                 label='Cumulative',
                 histtype='step',
                 color='blue',
                 cumulative=True,
                 density=True,
                 bins=np.arange(0,max(abs_raws_concat),1))
        ax2.set_yticks([0,0.25,0.5,0.75,1])
        plt.hlines([0.5],0, max(abs_raws_concat), colors = ['k'])
        plt.minorticks_on()
        ax.set_xlabel('Absolute Error (In-Situ vs. SDS, m)')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Cumulative Density (1/m)')
        ax.set_xlim(0, max(abs_raws_concat))
        ax.legend(loc=legend_loc)
        
        plt.subplot(2,1,2)
        plt.hist(abs_tides_concat,
                 label=tide_lab,
                 color='rosybrown',
                 density=False,
                 bins=np.arange(0,max(abs_tides_concat),1))
        ax = plt.gca()
        ax2 = ax.twinx()
        plt.hist(abs_tides_concat,
                 label='Cumulative',
                 histtype='step',
                 color='blue',
                 cumulative=True,
                 density=True,
                 bins=np.arange(0,max(abs_tides_concat),1))
        ax2.set_yticks([0,0.25,0.5,0.75,1])
        plt.hlines([0.5],0, max(abs_tides_concat), colors = ['k'])
        plt.minorticks_on()
        ax.set_xlabel('Absolute Error (In-Situ vs. SDS, m)')
        ax.set_ylabel('Count')
        ax2.set_ylabel('Cumulative Density (1/m)')
        ax.set_xlim(0, max(abs_tides_concat))
        ax.legend(loc=legend_loc)

        plt.tight_layout()
        plt.savefig(os.path.join(analysis_outputs, SITE+'_abs_err_dists'+EXT), dpi=DPI)
        plt.close('all')

    """
    Next set of code plots the maps of RMSE across the transects
    """

    transects_gdf = transects_gdf.merge(rmse_df, on='transect_id')
    transects_gdf = transects_gdf.merge(mae_df, on='transect_id')
    centroids = transects_gdf['geometry'].centroid
    all_data_rmse = np.concat([np.array(transects_gdf['rmse_raw']),
                               np.array(transects_gdf['rmse_tidally_corrected'])
                               ]
                              )
    all_data_mae = np.concat([np.array(transects_gdf['mae_raw']),
                              np.array(transects_gdf['mae_tidally_corrected'])
                              ]
                             )
    transects_gdf["markersize_raw_rmse"] = np.linspace(min(all_data_rmse),
                                                 max(all_data_rmse),
                                                 len(transects_gdf['rmse_raw']))/max(all_data_rmse)
    transects_gdf["markersize_tide_rmse"] = np.linspace(min(all_data_rmse),
                                                       max(all_data_rmse),
                                                       len(transects_gdf['rmse_tidally_corrected'])
                                                       )/max(all_data_rmse)

    transects_gdf["markersize_raw_mae"] = np.linspace(min(all_data_mae),
                                                 max(all_data_mae),
                                                 len(transects_gdf['mae_raw']))/max(all_data_mae)
    transects_gdf["markersize_tide_mae"] = np.linspace(min(all_data_mae),
                                                       max(all_data_mae),
                                                       len(transects_gdf['mae_tidally_corrected'])
                                                       )/max(all_data_mae)
    transects_gdf_centroid = transects_gdf.copy()
    transects_gdf_centroid['geometry'] = centroids
    transects_gdf_centroid = transects_gdf_centroid.to_crs(epsg=3857)

    ###Raw Experiment RMSE Map
    ax = transects_gdf_centroid.plot(column='rmse_raw',
                            legend=True,
                            vmin=min(all_data_rmse),
                            vmax=max(all_data_rmse),
                            legend_kwds={"label": "RMSE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='rmse_raw',
                            )
    ax.set_title('Raw')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_raw_map_rmse'+EXT), dpi=DPI)
    plt.close('all')

    ###Tidally Corrected Experiment RMSE Map
    ax = transects_gdf_centroid.plot(column='rmse_tidally_corrected',
                            legend=True,
                            vmin=min(all_data_rmse),
                            vmax=max(all_data_rmse),
                            legend_kwds={"label": "RMSE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='rmse_tidally_corrected',
                            )
    ax.set_title('Tidally Corrected')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_tidally_corrected_map_rmse'+EXT), dpi=DPI)
    plt.close('all')

    ###Raw Experiment MAE Map
    ax = transects_gdf_centroid.plot(column='mae_raw',
                            legend=True,
                            vmin=min(all_data_mae),
                            vmax=max(all_data_mae),
                            legend_kwds={"label": "MAE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='mae_raw',
                            )
    ax.set_title('Raw')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_raw_map_mae'+EXT), dpi=DPI)
    plt.close('all')

    ###Tidally Corrected Experiment MAE Map
    ax = transects_gdf_centroid.plot(column='mae_tidally_corrected',
                            legend=True,
                            vmin=min(all_data_mae),
                            vmax=max(all_data_mae),
                            legend_kwds={"label": "MAE (m)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='Reds',
                            markersize='mae_tidally_corrected',
                            )
    ax.set_title('Tidally Corrected')
    cx.add_basemap(ax,
                   source=BASEMAP_IMAGERY,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_tidally_corrected_map_mae'+EXT), dpi=DPI)
    plt.close('all')


    """
    This plots the violin plots of RMSE (per transect) for each experiment.
    """
    with plt.rc_context({"figure.figsize":(12,5)}):
        plt.title(SITE)
        plt.violinplot(data_rmse)
        plt.xticks([1, 2], ['Raw', 'Tidally Corrected'])
        plt.ylabel('RMSE (m)')
        plt.xlabel('Experiment')
        plt.minorticks_on()
        plt.savefig(os.path.join(analysis_outputs, SITE+ '_rmse_violinplots'+EXT), dpi=DPI)
        plt.close()

    """
    This plots the violin plots of MAE (per transect) for each experiment.
    """
    with plt.rc_context({"figure.figsize":(12,5)}):
        plt.title(SITE)
        plt.violinplot(data_mae)
        plt.xticks([1, 2], ['Raw', 'Tidally Corrected'])
        plt.ylabel('MAE (m)')
        plt.xlabel('Experiment')
        plt.minorticks_on()
        plt.savefig(os.path.join(analysis_outputs, SITE+ '_mae_violinplots'+EXT), dpi=DPI)
        plt.close()

    """
    Plotting trend maps
    """
    ##raw
    df_raw_new = df_raw[['dates','transect_id','cross_distance']]
    transect_timeseries_raw = pd.pivot_table(df_raw_new,index='dates',columns=['transect_id'],values=['cross_distance'])
    transect_timeseries_raw.columns = transect_timeseries_raw.columns.droplevel().rename(None)
    transect_timeseries_raw['dates'] = transect_timeseries_raw.index
    transect_timeseries_raw = transect_timeseries_raw.reset_index(drop=True)

    ##tidally corrected
    df_tide_new = df_tide[['dates','transect_id','cross_distance']]
    transect_timeseries_tide = pd.pivot_table(df_tide_new, index='dates',columns=['transect_id'],values=['cross_distance'])
    transect_timeseries_tide.columns = transect_timeseries_tide.columns.droplevel().rename(None)
    transect_timeseries_tide['dates'] = transect_timeseries_tide.index
    transect_timeseries_tide = transect_timeseries_tide.reset_index(drop=True)

    ##Trends
    transects_gdf = transects_gdf.rename(columns={'transect_id':'id'})
    transects_trends_gdf_raw = get_trends(transect_timeseries_raw,
                                                    transects_gdf,
                                                    min(transect_timeseries_raw['dates']),
                                                    max(transect_timeseries_raw['dates']),
                                                    trend_scale)
    transects_trends_gdf_tide = get_trends(transect_timeseries_tide,
                                                     transects_gdf,
                                                     min(transect_timeseries_tide['dates']),
                                                     max(transect_timeseries_tide['dates']),
                                                     trend_scale)
    ##save trends
    transects_trends_gdf_raw.to_file(os.path.join(analysis_outputs, SITE+'_raw_transect_trends.geojson'))
    transects_trends_gdf_tide.to_file(os.path.join(analysis_outputs, SITE+'_tidally_corrected_transect_trends.geojson'))

    ##project to web mercator for plotting
    transects_trends_gdf_raw = transects_trends_gdf_raw.to_crs(epsg=3857)
    transects_trends_gdf_tide = transects_trends_gdf_tide.to_crs(epsg=3857)

    ###No Corrections Trend Map
    ax = transects_trends_gdf_raw.plot(column='linear_trend',
                            legend=True,
                            legend_kwds={"label": "Trend (m/year)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='RdBu',
                            )
    ax.set_title('Raw')
    cx.add_basemap(ax,
                   source=BASEMAP_DARK,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_raw_map_trends'+EXT), dpi=DPI)
    plt.close('all')

    ###Low Slope Trend Map
    ax = transects_trends_gdf_tide.plot(column='linear_trend',
                            legend=True,
                            legend_kwds={"label": "Trend (m/year)", "orientation": "vertical", 'shrink': 0.3},
                            cmap='RdBu',
                            )
    ax.set_title('Tidally Corrected')
    cx.add_basemap(ax,
                   source=BASEMAP_DARK,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_outputs, SITE+'_tidally_corrected_map_trends'+EXT), dpi=DPI)
    plt.close('all')

    data_trends = [np.array(transects_trends_gdf_raw['linear_trend']),
                   np.array(transects_trends_gdf_tide['linear_trend'])
                   ]
    ##make boxplots of trends
    with plt.rc_context({"figure.figsize":(12,5)}):
        plt.title(SITE)
        plt.violinplot(data_trends)
        plt.xticks([1, 2], ['Raw', 'Tidally Corrected'])
        plt.ylabel('Trend (m/year)')
        plt.xlabel('Experiment')
        plt.minorticks_on()
        plt.savefig(os.path.join(analysis_outputs, SITE+'_trends_dist'+EXT), dpi=DPI)
        plt.close()

home = r'E:\test'
site = r'CapeCod'
window=10
in_situ_comparison(home,
                   site,
                   window,
                   plot_timeseries=False,
                   legend_loc=(0.4,0.6),
                   north_arrow_params=(0.05,0.2,0.1),
                   scale_bar_loc='lower left',
                   trend_scale=100)
