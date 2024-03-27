"""
Mark Lundine
Trains a parallel LSTM model given a matrix of shoreline positions (x-dimension is longshore distance, y-dimension is time)
To make a prediction, the model uses the previous n positions across every transect.
Its output is the next position at every transect.
If we have a matrix with all of the cross-shore positions, with y as our time dimension and x as our spatial dimension,
this model essentially is making the next row of cross-shore positions based on the previous n rows.
"""
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
import gc
plt.rcParams["figure.figsize"] = (16,6)
        
def split_sequences(sequences, n_steps_in, n_steps_out):
    """
    split a multivariate sequence into samples
    """
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix > len(sequences):
            break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
    X.append(seq_x)
    y.append(seq_y)
    return array(X), array(y)

def setup_data(timeseries_mat,
               transect_ids,
               split_percent,
               look_back,
               batch_size):
    """
    Organizes data to be keras-ready
    """
    ###load matrix and reshape to make a stacked timeseries
    dataset_list = []
    for transect_id in transect_ids:
        in_seq = np.array(timeseries_mat[transect_id])
        in_seq = in_seq.reshape((len(in_seq),1))
        dataset_list.append(in_seq)

    ###stack timeseries
    dataset = np.hstack(dataset_list)
    n_features = np.shape(dataset)[1]

    ### split into training and testing
    split = int(split_percent*len(dataset))
    shore_train = dataset[:split]
    shore_test = dataset[split:]

    ### make generators for train and test dataset
    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    prediction_generator = TimeseriesGenerator(dataset, dataset, length=look_back, batch_size=1)
 
    return dataset, train_generator, test_generator, n_features, prediction_generator

def project(timeseries_mat,
            dataset,
            look_back,
            num_prediction,
            n_features,
            model,
            freq):
    """
    Projects data beyond observed temporal range
    """
    ##original dimesion is time x transect
    prediction_mat = np.zeros((num_prediction+look_back, n_features))
    prediction_mat[0:look_back,:] = dataset[-look_back:, :]

    ##generating predctions recursively
    for i in range(num_prediction):
        update_idx = look_back+i
        x = prediction_mat[i:update_idx, :]
        x = x.reshape((1, look_back, n_features))
        out = model.predict(x, verbose=0)
        prediction_mat[update_idx, :] = out
    prediction = prediction_mat[look_back-1:, :]

    ##potentially use scaler here to normalize data
    #prediction = scaler.inverse_transform(prediction)

    last_date = timeseries_mat['date'].iloc[-1]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq=freq).tolist()
        

    forecast = prediction
    forecast_dates = prediction_dates

    return forecast, forecast_dates

def reset_keras():
    """
    Used to reset keras session and clear memory
    """
    ## this function is for clearing memory
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()
    sess = keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it does something you should see a number as output
    
def train_model(train_generator,
                test_generator,
                num_epochs,
                look_back,
                units,
                n_features):
    """
    Training the parallel LSTM model

    define model, two bidirection lstm layers and one dense layer, each with n units
    relu activation function
    use recurrent dropout
    look_back is number of previous data points to use in the prediction of the next data point
    n_features is the number of transects
    early stopping callback where best weights are restored
    loss function is mean absolute error
    optimizer is adam
    
    returns: model and history
    """
    model = Sequential()
    
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   return_sequences=True,
                   input_shape=(look_back, n_features),
                   recurrent_dropout=0.5
                   ))
              )
    
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   recurrent_dropout=0.5
                   ))
              )
    model.add(Dense(n_features))
    
    ### use adam optimizer and mae as loss function to keep the results of the loss function interpretable
    ### potential to use other loss functions to improve results but this makes the outputs less interpretable
    model.compile(optimizer='adam', loss='mae')
    
    ### fit model, use early stopping for number of epochs
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
    history = model.fit_generator(train_generator,
                                  epochs=num_epochs,
                                  callbacks=[early_stopping_callback],
                                  validation_data=test_generator,
                                  verbose=1)

    return model, history

def predict_data(model, prediction_generator):
    """
    Generating predictions from a trained model and a prediction generator
    """
    prediction = model.predict_generator(prediction_generator)
    #prediction = scaler.inverse_transform(prediction)
    return prediction

def process_results(sitename,
                    folder,
                    transect_ids,
                    mega_arr_pred,
                    dataset,
                    observed_dates,
                    date_predict,
                    forecast_dates,
                    mega_arr_forecast,
                    lookback,
                    split_percent):
    """
    This function makes two plots, one with the model results overlaying the satellite timeseries
    The other plot with the forecasted results
    So two plots for each transect
    It also saves csvs for the forecasted timeseries and the predicted timeseries
    inputs:
    sitename (str): name of site
    folder (str): path to save outputs
    transect_ids (array): transect ids in an array (in order along the shore)
    mega_arr_pred (array): array with all of the predictions (model outputs during observed temporal range)
    dataset (hstack): hstack of observed data
    observed_dates (array): array of the observed dates
    date_predict (array): array of the predicted dates
    forecast_dates (array): array of the forecasted dates
    mega_arr_forecast (array): array of all the model outputs for the forecast period
    lookback (int): lookback value
    split_percent (float): train/val split fraction (0.80 for 80/20 split)
    """
    n_features = np.shape(mega_arr_pred)[1]
    bootstrap = np.shape(mega_arr_pred)[2]
    forecast_dfs = [None]*len(transect_ids)
    predict_dfs = [None]*len(transect_ids)
    for i in range(len(transect_ids)):
        site = sitename+'_'+str(transect_ids[i])
        transect_data = mega_arr_pred[:,i,:]
        prediction_mean = np.mean(transect_data, axis=1)
        prediction_std_error = np.std(transect_data, axis=1)/np.sqrt(bootstrap)
        upper_conf_interval = prediction_mean + (prediction_std_error*1.96)
        lower_conf_interval = prediction_mean - (prediction_std_error*1.96)

        gt_date = observed_dates
        gt_vals = dataset[:,i]
        gt_split_idx = int(split_percent*len(gt_date)-1)
        gt_split_date = gt_date[gt_split_idx]
        plt.plot(gt_date, gt_vals, color='blue',label='Observed')
        plt.plot(date_predict,prediction_mean, '--', color='red', label='LSTM Projection Mean')
        ax = plt.gca()
        ylim = ax.get_ylim()
        plt.vlines(gt_split_date, ylim[0], ylim[1], color='k', label='Train/Val Split')
        plt.fill_between(date_predict, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlim(min(gt_date), max(date_predict))
        plt.minorticks_on()
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, site+'_predict.png'), dpi=300)
        plt.close('all')

        predict_df_dict = {'date': date_predict,
                           'predicted_mean_position': prediction_mean,
                           'predicted_upper_conf': upper_conf_interval,
                           'predicted_lower_conf': lower_conf_interval,
                           'observed_position': gt_vals[lookback:]}
        predict_df = pd.DataFrame(predict_df_dict)
        predict_df.to_csv(os.path.join(folder, site+'_predict.csv'),index=False)

        forecast_array = mega_arr_forecast[:,i,:]
        forecast_mean = np.mean(forecast_array, axis=1)
        forecast_std_error = np.std(forecast_array, axis=1)/np.sqrt(bootstrap)
        upper_conf_interval = forecast_mean + (forecast_std_error*1.96)
        lower_conf_interval = forecast_mean - (forecast_std_error*1.96)
        plt.plot(gt_date, gt_vals, color='blue',label='Observed')
        plt.plot(forecast_dates,forecast_mean, '--', color='red', label='LSTM Projection Mean')
        plt.fill_between(forecast_dates, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
        plt.xlabel('Time (UTC)')
        plt.ylabel('Cross-Shore Position (m)')
        plt.xlim(min(gt_date), max(forecast_dates))
        plt.minorticks_on()
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(os.path.join(folder, site+'_forecast.png'), dpi=300)
        plt.close('all')

        forecast_df_dict = {'date': forecast_dates,
                            'forecast_mean_position': forecast_mean,
                            'forecast_upper_conf': upper_conf_interval,
                            'forecast_lower_conf': lower_conf_interval}
        
        forecast_df = pd.DataFrame(forecast_df_dict)
        forecast_df.to_csv(os.path.join(folder, site+'_forecast.csv'),index=False)

        predict_dfs[i] = predict_df
        forecast_dfs[i] = forecast_df
        
    return predict_dfs, forecast_dfs
      
def plot_history(history):
    """
    This makes a plot of the loss curve
    inputs:
    history: history object from model.fit_generator
    """
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('Mean Absolute Error (m)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
def main(sitename,
         coastseg_matrix_path,
         bootstrap=30,
         num_prediction=30,
         num_epochs=2000,
         units=64,
         batch_size=32,
         look_back=3,
         split_percent=0.80,
         freq='30D'):
    """
    Trains parallel LSTM model on shoreline data
    inputs:
    sitename (str): name of the sitename/study area
    coastseg_matrix_path (str): path to the matrix output from coastseg_time_and_space_analysis_matrix.py
    bootstrap (int): number of times to train the model
    num_prediction (int): number of timesteps to project model
    num_epochs (int): number of epochs to train the model
    units (int): number of units for LSTM layers
    batch_size (int): batch size for training
    look_back (int): number of previous timesteps to use to predict next value
    split_percent (int): fraction of timeseries to use as training data
    freq (str): timestep, this should match the timedelta of the coastseg_matrix
    """
    ## just naming these variables again for my sanity
    look_back=look_back
    num_prediction=num_prediction
    bootstrap=bootstrap
    batch_size = batch_size
    num_epochs=num_epochs
    units=units

    output_folder = os.path.join(os.path.dirname(coastseg_matrix_path), 'lstm')
    try:
        os.mkdir(output_folder)
    except:
        pass
    
    ## need to get a slightly altered timeseries matrix and the transect ids with timeseries data
    timeseries_mat = pd.read_csv(coastseg_matrix_path)
    timeseries_mat['date'] = pd.to_datetime(timeseries_mat['date'], format = '%Y-%m-%d')
    
    ## get dates of observed data and model ready datasets
    observed_dates = timeseries_mat['date']
    transect_ids = list(timeseries_mat.columns[1:])
    dataset, train_generator, test_generator, n_features, prediction_generator = setup_data(timeseries_mat,
                                                                                            transect_ids,
                                                                                            split_percent,
                                                                                            look_back,
                                                                                            batch_size)
    ## setting up stuff for validation section and forecasting/projecting section
    date_predict = observed_dates[look_back:]
    mega_arr_pred = np.zeros((len(date_predict), n_features, bootstrap))
    mega_arr_forecast = np.zeros((num_prediction+1, n_features, bootstrap))

    ## training the model a bunch of times to get confidence intervals
    histories = [None]*bootstrap
    for i in range(bootstrap):
        ## clearing memory
        reset_keras()
        ## training the model
        model, history = train_model(train_generator,
                            test_generator,
                            num_epochs,
                            look_back,
                            units,
                            n_features)
        ## saving the loss history
        histories[i] = history

        ## getting validation predictions
        prediction = predict_data(model, prediction_generator)
        mega_arr_pred[:,:,i] = prediction

        ## getting projections/forecasts
        forecast, forecast_dates = project(timeseries_mat,
                                           dataset,
                                           look_back,
                                           num_prediction,
                                           n_features,
                                           model,
                                           freq)
        mega_arr_forecast[:,:,i] = forecast
        
    ## plotting data and exporting csvs for each timeseries
    predict_dfs, forecast_dfs = process_results(sitename,
                                                output_folder,
                                                transect_ids,
                                                mega_arr_pred,
                                                dataset,
                                                observed_dates,
                                                date_predict,
                                                forecast_dates,
                                                mega_arr_forecast,
                                                look_back,
                                                split_percent)

    predict_stacked_df = pd.concat(predict_dfs, keys=transect_ids)
    forecast_stacked_df = pd.concat(forecast_dfs, keys=transect_ids)
    
    predict_stacked_df = predict_stacked_df.reset_index()
    forecast_stacked_df = forecast_stacked_df.reset_index()

    predict_stacked_df.drop(columns=['level_1'],inplace=True)
    forecast_stacked_df.drop(columns=['level_1'],inplace=True)

    predict_stacked_df.rename(columns={'level_0':'transect_id'},inplace=True)
    forecast_stacked_df.rename(columns={'level_0':'transect_id'},inplace=True)

    predict_stacked_df.to_csv(os.path.join(output_folder, 'predict_stacked.csv'))
    forecast_stacked_df.to_csv(os.path.join(output_folder, 'forecast_stacked.csv'))

    ##Plot the loss curves
    i = 0
    for history in histories:
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        # save to csv: 
        hist_csv_file = os.path.join(output_folder,'history'+str(i)+'.csv')
        hist_df.to_csv(hist_csv_file, index=False)
        plot_history(history)
        i=i+1
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'), dpi=300)
    plt.close('all')

    ##reset keras again
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()
    
##if __name__ == "__main__":
##    from argparse import ArgumentParser
##    parser = ArgumentParser()
##    parser.add_argument("--site", type=str,required=True, help="site name")
##    parser.add_argument("--coastseg_matrix", type=str,required=True,help="path to resampled matrix")
##    parser.add_argument("--bootstrap", type=int, required=True, help="number of repeat trials")
##    parser.add_argument("--num_prediction",type=int, required=True, help="number of predictions")
##    parser.add_argument("--epochs",type=int, required=True, help="number of epochs to train")
##    parser.add_argument("--units",type=int, required=True, help="number of LSTM layers")
##    parser.add_argument("--batch_size",type=int, required=True, help="training batch size")
##    parser.add_argument("--lookback",type=int, required=True, help="look back value")
##    parser.add_argument("--split_percent",type=float, required=True, help="training data fraction")
##    parser.add_argument("--freq", type=str, required=True, help="prediction frequency (monthly, seasonally, biannually, yearly")
##    args = parser.parse_args()
##    main(args.site,
##         args.coastseg_matrix_path,
##         bootstrap=args.bootstrap,
##         num_prediction=args.num_prediction,
##         num_epochs=args.epochs,
##         units=args.units,
##         batch_size=args.batch_size,
##         look_back=args.lookback,
##         split_percent=args.split_percent,
##         freq=args.freq)   


    
