import pandas as pd
import warnings
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import shoreline_timeseries_analysis_single as stas
import gc


def get_shoreline_data(transect_timeseries_path,
                       transect_id,
                       folder,
                       which_timedelta,
                       timedelta=None):
    """
    Getting data from transect_time_series.csv
    Running it through the timeseries analysis cookbook
    """
    ##importing data
    df = pd.read_csv(transect_timeseries_path)
    new_df = pd.DataFrame({'date':pd.to_datetime(df['dates'], format='%Y-%m-%d %H:%M:%S+00:00'),
                           'position':df[transect_id]})

    analysis_result, new_df, timedelta = stas.main_df(new_df,
                                                      folder,
                                                      transect_id,
                                                      which_timedelta,
                                                      timedelta=timedelta)
    return new_df, timedelta

def setup_data(df, look_back, batch_size, split_percent=0.80):
    """
    Formatting data to feed into Keras model
    """
    shore_data = df['position'].values
    shore_data = shore_data.reshape((-1,1))

    split = int(split_percent*len(shore_data))

    shore_train = shore_data[:split]
    shore_test = shore_data[split:]

    date_train = df['date'][:split]
    date_test = df['date'][split:]

    train_generator = TimeseriesGenerator(shore_train, shore_train, length=look_back, batch_size=batch_size)     
    test_generator = TimeseriesGenerator(shore_test, shore_test, length=look_back, batch_size=1)
    prediction_generator = TimeseriesGenerator(shore_data, shore_data, length=look_back, batch_size=1)
    return shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator, prediction_generator

# Reset Keras Session
def reset_keras():
    """
    Used to clean up memory
    """
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()
    sess = keras.backend.get_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it does something you should see a number as output

def train_model(train_generator, test_generator, look_back, units=30, num_epochs=60):
    """
    Training the LSTM
    Structure is
    Bidirectional LSTM layer
    Bidirectional LSTM layer
    Dense Layer

    Loss function: mean absolute error (m)
    Optimizer: Adam
    Early stopping callback with restore best weights
    """
    model = Sequential()
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   return_sequences=True,
                   input_shape=(look_back, 1),
                   recurrent_dropout=0.5
                   ))
              )
    model.add(Bidirectional(LSTM(units,
                   activation='relu',
                   recurrent_dropout=0.5
                   ))
              )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mae')
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
    history = model.fit_generator(train_generator,
                                  epochs=num_epochs,
                                  callbacks=[early_stopping_callback],
                                  validation_data=test_generator,
                                  verbose=1)
    return model, history

def predict_data(model, prediction_generator):
    """
    Getting predictions from trained model
    """
    prediction = model.predict_generator(prediction_generator)
    return prediction

def project(df,
            shore_data,
            look_back,
            num_prediction,
            model,
            timedelta):
    """
    Projecting the model beyond the observed time period
    """

    shore_data = shore_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = shore_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1, freq=timedelta).tolist()
        return prediction_dates

    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    
    return forecast, forecast_dates

def plot_history(history):
    """
    This makes a plot of the loss curve
    """
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('Mean Absolute Error (m)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    
def run(csv_path,
        transect_id,
        site,
        folder,
        bootstrap=30,
        num_prediction=30,
        epochs=300,
        units=6,
        batch_size=32,
        lookback=4,
        split_percent=0.80,
        which_timedelta='maximum',
        timedelta=None):
    """
    Trains an LSTM model on a single transect of shoreline positions obtained from CoastSeg
    Runs model on observed timesteps and then projects model recursively num_prediction timesteps into the future
    
    This will save the following files:
    All outputs from shoreline_timeseries_analysis_single.py (result.csv, resampled.csv, timeseries.png, autocorrelation.png)
    history#.csv: for each training trial which contains the epoch, training loss, and validation loss
    loss_plot.png: which plots each loss curve
    sitepredict.png: plots the training and validation data with the model prediction
    sitepredict.csv: csv with training and validation data as well as the model prediction
    siteproject.png: plots the ground-truth data and then the model projection
    siteproject.csv: csv with the projected data (timestamps and positions)
    
    inputs:
    csv_path (str): path to the transect_time_series.csv or to the transect_time_series_tidally_corrected.csv
    transect_id (str): which transect_id to train the model on
    site (str): a name for the site
    folder (str): where to save the results to
    boostrap (int): default is 30, this is the number of models to train to get a confidence interval of outputs
    num_prediction (int): default is 30, this is the number of future timesteps to project model
    epochs (int): default is 300, set this to a larger number if you find training is ending too early, there is an early stopping callback
                  to ensure model trains until validation loss is minimized
    units (int): default is 64, number of LSTM units, this can be thought of as the complexity of the LSTM layers
    batch_size (int): default is 32, can make larger to speed things up but beware of too big of a batch size and its effects on val_loss
    lookback (int): default is 3, number of previous timesteps to use in the prediction of the next timestep
    split_percent (float): default is 0.80, this is the train/val split, 0.80 would be a split of 80/20
    which_timedelta (str): 'maximum' 'minimum' 'average' or 'custom', method for resampling the timeseries, beware of using 'minimum'
    timedelta (str): default is None, only specify a timedelta if using 'custom', if you choose custom and want to resample at 12 days,
                     timedelta would be '12D'
    """

    ##Just repeating parameters here
    look_back=lookback
    num_prediction=num_prediction
    bootstrap=bootstrap
    batch_size = batch_size
    epochs=epochs
    units=units
    split_percent=split_percent

    ##Get the resampled timeseries and the new timedelta
    df, timedelta = get_shoreline_data(csv_path, transect_id, folder, which_timedelta, timedelta=timedelta)

    ##Getting the observed dates
    observed_dates = df['date']
    date_predict = observed_dates[look_back:]

    ##Making arrays to hold predictions and projections/forecasts
    predict_array = np.zeros((bootstrap, len(date_predict)))
    forecast_array = np.zeros((bootstrap, num_prediction+1))

    ##Formatting data to train model
    shore_data, shore_train, shore_test, date_train, date_test, train_generator, test_generator, prediction_generator = setup_data(df,
                                                                                                                                   look_back,
                                                                                                                                   batch_size,
                                                                                                                                   split_percent=split_percent)

    ##Looping over bootstrap for multiple training trials
    histories = [None]*bootstrap
    for i in range(bootstrap):
        print('trial: '+str(i+1))
        reset_keras()
        model, history = train_model(train_generator, test_generator, look_back, units=units, num_epochs=epochs)
        prediction = predict_data(model, prediction_generator)
        forecast, forecast_dates = project(df,
                                           shore_data,
                                           look_back,
                                           num_prediction,
                                           model,
                                           timedelta)
        predict_array[i,:] = prediction[:,0]
        forecast_array[i,:] = forecast
        histories[i] = history


    ##Getting the mean and confidence intervals for the forecasts
    forecast_mean = np.mean(forecast_array, axis=0)
    forecast_std_error = np.std(forecast_array, axis=0)/np.sqrt(bootstrap)
    upper_conf_interval = forecast_mean + (forecast_std_error*1.96)
    lower_conf_interval = forecast_mean - (forecast_std_error*1.96)

    ##Getting the mean and confidence intervals for the predictions during the observed time period
    prediction_mean = np.mean(predict_array, axis=0)
    prediction_std_error = np.std(predict_array, axis=0)/np.sqrt(bootstrap)
    pred_upper_conf_interval = prediction_mean + (prediction_std_error*1.96)
    pred_lower_conf_interval = prediction_mean - (prediction_std_error*1.96)

    ##Just some reshaping
    shore_train = shore_train.reshape((-1))
    shore_test = shore_test.reshape((-1))
    prediction = prediction.reshape((-1))

    ##Get last date of the training data
    gt_split_date = date_train[-1]

    ##Plot for observed time period
    plt.rcParams["figure.figsize"] = (16,4)
    plt.plot(observed_dates[:gt_split_date], df['position'][:gt_split_date], color='blue', label='Training Data')
    plt.plot(observed_dates[gt_split_date:], df['position'][gt_split_date:], color='green', label='Validation Data')
    plt.plot(date_predict, prediction, label='Prediction', color='red')
    plt.fill_between(date_predict, pred_lower_conf_interval, pred_upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.vlines(gt_split_date, ylim[0], ylim[1], color='k', label='Train/Val Split')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.legend()
    plt.minorticks_on()
    plt.xlim(min(date_train), max(date_test))
    plt.tight_layout()
    plt.savefig(os.path.join(folder, site+'predict.png'), dpi=300)
    plt.close('all')

    ##Save observed time period data to csv
    new_df_dict = {'time': date_predict,
                   'forecast_mean_position': prediction_mean,
                   'forecast_upper_conf': pred_upper_conf_interval,
                   'forecast_lower_conf': pred_lower_conf_interval,
                   'observed_position': df['position'][lookback:]}
    new_df = pd.DataFrame(new_df_dict)
    new_df.to_csv(os.path.join(folder, site+'predict.csv'),index=False)

    ##Plot for projections
    plt.plot(df['date'], df['position'], color='blue',label='Observed Three Month Moving Average')
    plt.plot(forecast_dates,forecast_mean, '--', color='red', label='LSTM Projection Mean')
    plt.fill_between(forecast_dates, lower_conf_interval, upper_conf_interval, color='red', alpha=0.4, label='LSTM 95% Confidence Interval')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.xlim(min(df['date']), max(forecast_dates))
    plt.minorticks_on()
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(folder, site+'project.png'), dpi=300)
    plt.close('all')

    ##Save projections to csv
    new_df_dict = {'time': forecast_dates,
                   'forecast_mean_position': forecast_mean,
                   'forecast_upper_conf': upper_conf_interval,
                   'forecast_lower_conf': lower_conf_interval}
    new_df = pd.DataFrame(new_df_dict)
    new_df.to_csv(os.path.join(folder, site+'project.csv'),index=False)

    ##Making the loss curve plot
    i = 0
    for history in histories:
        # convert the history.history dict to a pandas DataFrame:     
        hist_df = pd.DataFrame(history.history) 
        # save to csv: 
        hist_csv_file = os.path.join(folder,'history'+str(i)+'.csv')
        hist_df.to_csv(hist_csv_file, index=False)
        plot_history(history)
        i=i+1
    plt.savefig(os.path.join(folder, 'loss_plot.png'), dpi=300)
    plt.close('all')
    sess = keras.backend.get_session()
    keras.backend.clear_session()
    sess.close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str,required=True,help="path to transect csv")
    parser.add_argument("--transect_id", type=str,required=True,help="transect id")
    parser.add_argument("--site", type=str,required=True, help="site name")
    parser.add_argument("--folder", type=str, required=True, help="path to projected folder")
    parser.add_argument("--bootstrap", type=int, required=True, help="number of repeat trials")
    parser.add_argument("--num_prediction",type=int, required=True, help="number of predictions")
    parser.add_argument("--epochs",type=int, required=True, help="number of epochs to train")
    parser.add_argument("--units",type=int, required=True, help="number of LSTM layers")
    parser.add_argument("--batch_size",type=int, required=True, help="training batch size")
    parser.add_argument("--lookback",type=int, required=True, help="look back value")
    parser.add_argument("--split_percent",type=float, required=True, help="train/test split fraction, ex: 0.80 for 80/20 split")
    parser.add_argument("--which_timedelta",type=str, required=True, help="resample method")
    parser.add_argument("--timedelta",type=str, required=True, help="if custom for which_timedelta, the new timedelta")
    args = parser.parse_args()
    run(args.csv_path,
        args.transect_id,
        args.site,
        args.folder,
        bootstrap=args.bootstrap,
        num_prediction=args.num_prediction,
        epochs=args.epochs,
        units=args.units,
        batch_size=args.batch_size,
        lookback=args.lookback,
        split_percent=args.split_percent,
        which_timedelta=args.which_timedelta,
        timdelta=args.timedelta)   

   
