### a test of some functions
from src.sdstools import detrend
from src.sdstools import io 
from src.sdstools import interpolation 
from src.sdstools import filter
from src.sdstools import viz
import os


### input files
cs_file = os.path.normpath(os.getcwd()+'/src/sdstools/example_data/transect_time_series_coastsat.csv')
zoo_file = os.path.normpath(os.getcwd()+'/src/sdstools/example_data/transect_time_series_zoo.csv')


######################## get data
cs_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(cs_file)
zoo_data_matrix, zoo_dates_vector, zoo_transects_vector = io.read_merged_transect_time_series_file(zoo_file)

######################## impute
cs_data_matrix_nonans = interpolation.inpaint_spacetime_matrix(cs_data_matrix)
zoo_data_matrix_nonans = interpolation.inpaint_spacetime_matrix(zoo_data_matrix)

######################## denoise
cs_data_matrix_nonans_denoised = filter.filter_wavelet_auto(cs_data_matrix_nonans)
zoo_data_matrix_nonans_denoised = filter.filter_wavelet_auto(zoo_data_matrix_nonans)

######################## detrend
cs_detrend = detrend.detrend_shoreline_rel_start(cs_data_matrix_nonans_denoised,N=10)
zoo_detrend = detrend.detrend_shoreline_rel_start(zoo_data_matrix_nonans_denoised,N=10)

######################## plots
viz.p_cs_zoo_sidebyside(cs_dates_vector, zoo_dates_vector, cs_detrend, zoo_detrend)



##################################################
### input files
## shoreline data by time and transect
dat_time_by_transect = os.path.normpath(os.getcwd()+'/src/sdstools/example_data/elwha_mainROI_df_distances_by_time_and_transect_CoastSat.csv')
## shoreline data by tide and transect 
dat_tide_by_transect = os.path.normpath(os.getcwd()+'/src/sdstools/example_data/elwha_mainROI_df_tides_by_time_and_transect_CoastSat.csv')

cs_shoreline_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(dat_time_by_transect)
cs_tide_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(dat_tide_by_transect)


