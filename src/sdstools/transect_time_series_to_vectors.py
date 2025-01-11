import geopandas as gpd
import pandas as pd
import os
import numpy as np
import shapely

def split_list_at_none(lst):
    # Initialize variables
    result = []
    temp = []

    # Iterate through the list
    for item in lst:
        if item is None:
            # Append the current sublist to the result and reset temp
            result.append(temp)
            temp = []
        else:
            # Add item to the current sublist
            temp.append(item)

    # Append the last sublist if not empty
    if temp:
        result.append(temp)

    return result

def remove_nones(my_list):
    new_list = [x for x in my_list if x is not None]
    return new_list

def tidally_corrected_time_series_merged_to_vectors(time_series_merged_path,
                                                    config_gdf_path,
                                                    output_vectors_path,
                                                    boundary_transect_ids = [(1,153),
                                                                             (154, 179),
                                                                             (180, 358)]
                                                    ):
    """
    Goes from time_series_merged_path.csv to line vectors
    inputs:
    time_series_merged_path (str): path to raw_transect_time_series_merged.csv or path to tidally_corrected_transect_time_series_merged.csv
    config_gdf_path (str): path to the config_gdf
    output_vectors_path (str): path to save line vectors to
    boundary_transect_ids (list): list of tuples with ids that define where to start and end lines
    """
    config_gdf = gpd.read_file(config_gdf_path)
    time_series = pd.read_csv(time_series_merged_path)
    time_series['transect_id'] = time_series['transect_id'].astype(int)
    time_series = time_series.sort_values(by='transect_id')
    all_lines = [None]*len(boundary_transect_ids)
    for i in range(len(boundary_transect_ids)):
        boundary = boundary_transect_ids[i]
        boundary_start = boundary[0]
        boundary_end = boundary[1]
        transects = config_gdf[config_gdf['type']=='transect']
        transects['id'] = transects['id'].astype(int)
        transects = transects[(transects['id']>=boundary_start) & (transects['id']<=boundary_end)]
        transects = transects.sort_values(by='id').reset_index(drop=True)
        dates = np.unique(time_series['dates'])
        new_lines_dates = []
        new_lines = []
        for j in range(len(dates)):
            date = dates[j]
            time_series_date_filter = time_series[time_series['dates']==date]
            points = [None]*len(transects)
            points_ids = [None]*len(transects)
            for k in range(len(transects)):
                transect = transects.iloc[k]
                transect_id = transect['id']
                time_series_transect_filter = time_series_date_filter[time_series_date_filter['transect_id']==transect_id]
                x = time_series_transect_filter['shore_x']
                y = time_series_transect_filter['shore_y']
                points_ids[k] = transect_id
                try:
                    coords = shapely.Point((x,y))
                    points[k] = coords
                except:
                    continue
            temp_df = pd.DataFrame({'ids':points_ids,
                                    'points':points})
            
            split_lists = split_list_at_none(points)
            for spl in split_lists:
                if len(spl)>1:
                    new_lines.append(shapely.LineString(spl))
                    new_lines_dates.append(date)
                elif len(spl) == 1:
                    # create a new point by adding a small amount to the x value and y value of the point
                    point = spl[0]
                    x = point.x + 0.00001
                    y = point.y + 0.00001
                    new_point = (x,y)
                    new_lines.append(shapely.LineString([point, new_point]))
                    new_lines_dates.append(date)
        new_gdf_dict = {'date':new_lines_dates,
                        'geometry':new_lines}
        new_gdf = gpd.GeoDataFrame(pd.DataFrame(new_gdf_dict), crs=config_gdf.crs)
        all_lines[i] = new_gdf
    final_gdf = pd.concat(all_lines)
    final_gdf['date'] = pd.to_datetime(final_gdf['date'])
    final_gdf['year'] = final_gdf['date'].dt.year
    final_gdf.to_file(output_vectors_path)
            
