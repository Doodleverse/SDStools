
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL

from scipy import signal
import calendar
import pywt
import matplotlib.pyplot as plt



def toTimestamp(d):
  return calendar.timegm(d.timetuple())


def sds_wavelet_decompose(SDS_timeseries, SDS_timeseries_timestamp):
    #interpolate to one observation per day
    ts = np.array([toTimestamp(d) for d in SDS_timeseries_timestamp]) 
    tsi = np.arange(SDS_timeseries_timestamp[0],SDS_timeseries_timestamp[-1],60*60*24)

    result = np.interp(tsi,ts,SDS_timeseries)
    len(result)

    ## wavelet analysis

    [cfs, frequencies] = pywt.cwt(result, np.arange(1, 365, 1),  'morl' , .5)
    period = 1. / frequencies
    power =(abs(cfs)) ** 2
    power = np.mean(np.abs(power), axis=1)/(period**2)


    tsi_dt = [datetime.fromtimestamp(t) for t in tsi]
    ts_dt = [datetime.fromtimestamp(t) for t in ts]


# plt.figure(figsize=(10,10))
# plt.subplot(211)
# plt.plot(tsi_dt,result)
# plt.plot(ts_dt,secondmaxvar,'r.')
# plt.ylabel('Shoreline trend, Agate Beach')
# plt.xlabel('Time')

# plt.subplot(212)
# plt.plot(period, power,'m', lw=2)
# plt.xlabel('Frequency (day)')
# plt.ylabel(r'Power ($m^2$/day)')
# plt.axvline(365/2)
# plt.axvline(365/4)
# plt.text(365/2,5,'6-mo.\n periodicity')
# plt.text(365/4,5,'3-mo.\n periodicity')
# # plt.show()
# plt.savefig('Agate_beach_periodicity.png', dpi=200, bbox_inches='tight')
# plt.close()
    


def sds_periodogram_decompose(SDS_timeseries, SDS_timeseries_timestamp):

    fs = 1/(SDS_timeseries_timestamp[1]-SDS_timeseries_timestamp[0])

    f, Pxx_den = signal.periodogram(SDS_timeseries, fs)

    plt.semilogy(f, Pxx_den)

    plt.xlabel('frequency [Hz]')

    plt.ylabel('PSD [V**2/Hz]')

    plt.show()


    # signal.welch
    f, Pxx_spec = signal.welch(SDS_timeseries, fs, 'flattop', 1024, scaling='spectrum')
    plt.figure()
    plt.semilogy(f, np.sqrt(Pxx_spec))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.title('Power spectrum (scipy.signal.welch)')
    plt.show()



def shoreline_seasonality(ds):
    ## weighted average shoreline per season

    month_length = ds.time.dt.days_in_month
    # Calculate the weights by grouping by 'time.season'.
    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby("time.season").sum(dim="time")
    return ds_weighted




# x = (100*np.arange(len(kmt_t)))/1000

# plt.figure(figsize=(12,6))
# plt.subplot(211)
# plt.plot(x,ds_weighted[0], label='JFM')
# plt.plot(x,ds_weighted[1], label='AMJ')
# plt.plot(x,ds_weighted[2], label='JAS')
# plt.plot(x,ds_weighted[3], label='OND')
# plt.legend(fontsize=9)
# plt.axhline(0, color='k')

# for k in coastal_features.keys():
#     ind = np.where(np.array(kmt_t)==k)[0]
#     plt.axvline(x[ind], color='k')
#     plt.text(x[ind]-.5,5, coastal_features[k], rotation=90, color='k', fontsize=7)


# plt.xlabel('Kilometers alongshore')
# plt.ylabel('Weighted seasonal \n average shoreline trend')
# # plt.show()
# plt.savefig('Seasonal_trends_alongshore.png', dpi=200, bbox_inches='tight')
# plt.close()



def compute_alongshore_autocorrelation(data_ref, data_matrix):
    data_ref = (data_ref-data_ref.mean())/data_ref.std()

    alongcorr=[]
    for k in np.arange(data_matrix.shape[0]):
        alongcorr.append(np.correlate(data_ref,(data_matrix[k,:]-data_matrix[k,:].mean())/data_matrix[k,:].std(),'same'))

    alongcorr = np.vstack(alongcorr)

    return alongcorr



# name = 'kmt_transects_demean_vs_time_correl'

# plt.figure(figsize=(18,18))
# x = np.arange(len(kmt_t))

# plt.pcolormesh(kmt_dt,x,np.flipud(alongcorr.T), cmap=plt.cm.RdYlGn)
# ind = np.round(np.linspace(0,len(kmt_t)-1,10)).astype('int')
# plt.gca().set_yticks(ind)
# plt.gca().set_yticklabels([kmt_t[i] for i in ind.astype('int')], rotation=45)

# cb=plt.colorbar()
# cb.set_label('Temporal correlation')
# plt.xlabel('Time')
# plt.ylabel('Transect')

# # plt.show()
# plt.savefig(name+'.png', dpi=200, bbox_inches='tight')
# plt.close()



def seasonal_decompose_data_matrix(data_matrix):
    output = seasonal_decompose(data_matrix,
                       model='additive', 
                       filt=None, 
                       period=None, 
                       two_sided=True, 
                       extrapolate_trend=0).fit()
    
    return output


def seasonal_decompose_data_matrix(data_matrix):
    output = STL(data_matrix, period=None, 
                                seasonal=7, 
                                trend=None, 
                                low_pass=None, 
                                seasonal_deg=1, 
                                trend_deg=1, 
                                low_pass_deg=1, 
                                robust=False, 
                                seasonal_jump=1, 
                                trend_jump=1, 
                                low_pass_jump=1).fit()
    return output
