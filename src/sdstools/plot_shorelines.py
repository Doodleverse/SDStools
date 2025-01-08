"""
Applying Filters to shorelines and plot
"""
import geopandas as gpd
import contextily as cx
import numpy as np
import os
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.pyplot as plt
import shapely

def wgs84_to_utm_df(geo_df):
    """
    Converts wgs84 to UTM
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in wgs84
    outputs:
    geo_df_utm (geopandas  dataframe): a geopandas dataframe in utm
    """
    utm_crs = geo_df.estimate_utm_crs()
    gdf_utm = geo_df.to_crs(utm_crs)
    return gdf_utm

def utm_to_wgs84_df(geo_df):
    """
    Converts utm to wgs84
    inputs:
    geo_df (geopandas dataframe): a geopandas dataframe in utm
    outputs:
    geo_df_wgs84 (geopandas  dataframe): a geopandas dataframe in wgs84
    """
    wgs84_crs = 'epsg:4326'
    gdf_wgs84 = geo_df.to_crs(wgs84_crs)
    return gdf_wgs84

def LineString_to_arr(line):
    """
    Makes an array from linestring
    inputs: line
    outputs: array of xy tuples
    """
    listarray = []
    for pp in line.coords:
        listarray.append(pp)
    nparray = np.array(listarray)
    return nparray

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

def chaikins_corner_cutting(coords, refinements=3):
    """
    Smooths out lines or polygons with Chaikin's method
    inputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)]
    outputs:
    coords (list of tuples): [(x1,y1), (x..,y..), (xn,yn)],
                              this is the smooth line
    """
    i=0
    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25
        i=i+1
    return coords

def smooth_lines(lines,simplify_param=20,refinements=1):
    """
    Smooths out shorelines with Chaikin's method
    Shorelines need to be in UTM (or another planar coordinate system)

    inputs:
    shorelines (gdf): gdf of extracted shorelines in UTM
    refinements (int): number of refinemnets for Chaikin's smoothing algorithm
    outputs:
    new_lines (gdf): gdf of smooth lines in UTM
    """
    lines = wgs84_to_utm_df(lines)
    lines['geometry'] = lines['geometry']
    new_lines = lines.copy()
    for i in range(len(new_lines)):
        line = new_lines.iloc[i]['geometry'].simplify(simplify_param)
        coords = LineString_to_arr(line)
        refined = chaikins_corner_cutting(coords, refinements=refinements)
        refined_geom = arr_to_LineString(refined)
        new_lines.at[i,'geometry'] = refined_geom
    new_lines['geometry'] = new_lines['geometry'].simplify(simplify_param)
    new_lines = utm_to_wgs84_df(new_lines)
    return new_lines

def vertex_filter(shorelines,n_sigma):
    """
    Recursive 3-sigma filter on vertices in shorelines
    Will filter out shorelines that have too many or too few
    vertices until all of the shorelines left in the file are within
    Mean+/-3*std
    
    Saves output to the same directory with same name but with (_vtx) appended.

    inputs:
    shorelines (str): path to the extracted shorelines geojson
    outputs:
    new_path (str): path to the filtered file 
    """
    gdf = wgs84_to_utm_df(shorelines)
    
    count = len(gdf)
    new_count = None
    for index, row in gdf.iterrows():
        vtx = len(row['geometry'].coords)
        length = row['geometry'].length
        gdf.at[index,'vtx'] = vtx
        gdf.at[index,'length'] = length
        gdf.at[index,'length:vtx'] = length/vtx
    filter_gdf = gdf.copy()

    while count != new_count:
        count = len(filter_gdf)
        sigma = np.std(filter_gdf['length:vtx'])
        mean = np.mean(filter_gdf['length:vtx'])
        high_limit = mean+n_sigma*sigma
        low_limit = mean-n_sigma*sigma
        filter_gdf = gdf[gdf['length:vtx']< high_limit]
        filter_gdf = filter_gdf[filter_gdf['length:vtx']> low_limit]
        if mean < 5:
            break
        new_count = len(filter_gdf)
        
    filter_gdf = filter_gdf.reset_index(drop=True)
    filter_gdf = utm_to_wgs84_df(filter_gdf)
    return filter_gdf

def add_north_arrow(ax, north_arrow_params):
    x,y,arrow_length = north_arrow_params
    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
                arrowprops=dict(facecolor='white', width=2, headwidth=4),
                ha='center', va='center', fontsize=8, color='white',
                xycoords=ax.transAxes)
    
def min_max_normalize(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    return (arr - min_val) / (max_val - min_val)

def apply_filters(shorelines,

                  weighted_overall_score_thresh=0,
                  n_sigma_vertex=3):
    
    weighted_overall_scores = (0.2*shorelines['kde_value']+
                          0.5*shorelines['model_scores']+
                          0.3*shorelines['model_scores_seg'])/3
    weighted_overall_scores = min_max_normalize(weighted_overall_scores)
    shorelines['weighted_overall_score'] = weighted_overall_scores
    shorelines = shorelines[shorelines['weighted_overall_score']>=weighted_overall_score_thresh]
    #shorelines = shorelines[shorelines['model_scores']>=model_score_thresh]
    #shorelines = shorelines[shorelines['model_scores_seg']>=model_score_thresh]
    #shorelines = shorelines[shorelines['kde_value']>=kde_score_thresh].reset_index(drop=True)
    shorelines = vertex_filter(shorelines,n_sigma=n_sigma_vertex)
    return shorelines

def plot_shorelines_by_year(shorelines, site, north_arrow_params, scale_bar_loc, column='year'):
    shorelines['year'] = shorelines['dates'].dt.year
    shorelines = shorelines.sort_values(by=column)
    lines = shorelines.to_crs(epsg=3857)
    ax = lines.plot(column=column,
                            legend=True,
                            legend_kwds={"label": column, "orientation": "vertical", 'shrink': 0.3},
                            cmap='viridis',
                            linewidth=1,
                            )
    ax.set_title(site)
    cx.add_basemap(ax,
                   source=cx.providers.Esri.WorldImagery,
                   attribution=False)
    add_north_arrow(ax, north_arrow_params)
    ax.add_artist(ScaleBar(1, location=scale_bar_loc))
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
                  
north_arrow_params_dict = {'BarterIsland':(0.25, 0.93, 0.2),
                           'CapeCod':(0.05, 0.2, 0.1),
                           'Elwha':(0.23, 0.92, 0.2),
                           'Madeira':(0.05, 0.2, 0.1),
                           'PuertoRico':(0.05, 0.2, 0.1)
                           }
scale_bar_loc_dict = {'BarterIsland':'upper left',
                      'CapeCod':'lower left',
                      'Elwha':'upper left',
                      'Madeira':'lower left',
                      'PuertoRico':'lower left'
                      }
site = 'BarterIsland'
shorelines = gpd.read_file(r'E:\TCA_Shoreline_Sessions\analysis_ready_data\RGB\AK\BarterIsland\extracted_shorelines_lines_sep_joined.geojson')
model_score_thresh=0
seg_score_thresh=0
kde_score_thresh=0
weighted_overall_score_thresh=0.9
filtered_shorelines = apply_filters(shorelines,
                           model_score_thresh=model_score_thresh,
                           seg_score_thresh=seg_score_thresh,
                           kde_score_thresh=kde_score_thresh,
                           weighted_overall_score_thresh=weighted_overall_score_thresh,
                           )
column = 'year'
north_arrow_params = north_arrow_params_dict[site]
scale_bar_loc = scale_bar_loc_dict[site]
plot_shorelines_by_year(shorelines, site, north_arrow_params, scale_bar_loc,column='weighted_overall_score')
plot_shorelines_by_year(filtered_shorelines, site, north_arrow_params,scale_bar_loc,column='weighted_overall_score')            





                           
