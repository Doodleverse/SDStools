

##### WORK IN PROGRESS - NOTHING TO SEE HERE YET

### a test of some functions
from sdstools import io 
from sdstools import interpolation 
from sdstools import filter
# from src.SDStools import viz
from sdstools import detrend
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

csv_file = '/media/marda/TWOTB/USGS/Doodleverse/github/SDStools/example_data/elwha_mainROI_df_distances_by_time_and_transect_CoastSat.csv'

### input files
cs_file = os.path.normpath(csv_file)
cs_data_matrix, cs_dates_vector, cs_transects_vector = io.read_merged_transect_time_series_file(cs_file)


iterations = 5
windowPerc   = .05

orig = cs_data_matrix.copy()

cs_data_matrix_outliers_removed = cs_data_matrix.copy()
for k in range(orig.shape[0]):
    SDS_timeseries = orig[k,:]
    try:
        outliers = filter.hampel_filter(SDS_timeseries, window_size=int(windowPerc * len(SDS_timeseries)), n_sigma=2) 
    except:
        outliers = filter.hampel_filter(SDS_timeseries, window_size=1+int(windowPerc * len(SDS_timeseries)), n_sigma=2) 

    # outliers = filter.hampel_filter_matlab(SDS_timeseries, NoSTDsRemoved = 3, iterations   = 5, windowPerc   = .05)
    # print(len(outliers))
    cs_data_matrix_outliers_removed[k,outliers] = np.nan 




# cs_data_matrix_outliers_removed[np.isnan(cs_data_matrix)] = np.nan

num_outliers_removed = np.sum(np.isnan(cs_data_matrix_outliers_removed)) -  np.sum(np.isnan(cs_data_matrix))
print(f"Outliers removed: {num_outliers_removed}")
print(f"Outliers removed percent: {100*(num_outliers_removed/np.prod(np.shape(cs_data_matrix_outliers_removed)))}")


df = pd.DataFrame(cs_data_matrix_outliers_removed.T,columns=cs_transects_vector)
df.set_index(cs_dates_vector)
df.to_csv(csv_file.replace(".csv","_nooutliers.csv"))

plt.subplot(121)
plt.imshow(cs_data_matrix)
plt.subplot(122)
plt.imshow(cs_data_matrix_outliers_removed)
plt.show()


######################## impute
cs_data_matrix_nooutliers_nonans = interpolation.inpaint_spacetime_matrix(cs_data_matrix_outliers_removed)


plt.subplot(121)
plt.imshow(cs_data_matrix_outliers_removed)
plt.subplot(122)
plt.imshow(cs_data_matrix_nooutliers_nonans)
plt.show()



######################## denoise
cs_data_matrix_nonans_denoised = filter.filter_wavelet_auto(cs_data_matrix_nonans)
# zoo_data_matrix_nonans_denoised = filter.filter_wavelet_auto(zoo_data_matrix_nonans)

######################## detrend
cs_detrend = detrend.detrend_shoreline_rel_start(cs_data_matrix_nonans_denoised,N=10)
# zoo_detrend = detrend.detrend_shoreline_rel_start(zoo_data_matrix_nonans_denoised,N=10)

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


