"""
Mark Lundine
Experimental Script for
training and implementing a multilayer perceptron for timeseries prediction.

Spoiler alert they don't work that well like most NNs for timeseries prediction!

I just feel this is a more honest way of showing what they actually can do in an operational sense.
"""

import warnings
import os
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import gc
from CustomEarlyStopping import CustomEarlyStopping



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
# This function keeps the initial learning rate for the first ten epochs
# and decreases it exponentially after that.

def plot_history(history):
    """
    Plots loss curve
    inputs:
    history (history object from training a keras model)
    """
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('RMSE Loss (m)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')

def make_prediction(inputs, model):
    """
    Generates a single prediction from a trained model
    inputs:
    inputs (np.ndarray)
    model (keras model)
    """
    y_hat = model.predict(inputs)
    return y_hat

def define_model(num_forcing, neurons):
    """
    Defines the MLP model
    inputs:
    num_forcing (int): number of forcing variables
    neurons (int): number of neurons for the hidden (non-linear) layer
    outputs: model (Keras model) and callbacks (list)
    """
    
    def scheduler(epoch, lr):
        """
        Using a scheduler for smoother loss curves,
        this gets fed as a model callback
        """
        return lr * np.exp(-0.01)
    
    def rmse(y_true, y_pred):
        """
        Root mean squared error loss fed to keras model
        """
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    """
    Defining the model
    Number of neorons should be equal to number of forcing variables + 1 (extra neuron as a bias term)

    Model structure:
    Input Layer (num_forcing x 1)
    One non-linear (hidden) layer (num_focing +1 neurons)
    One linear layer (1 neuron for 1 output)
    """

    # input
    visible = Input(shape=(num_forcing,))
    
    # hidden
    hidden1 = Dense(neurons, kernel_initializer='he_normal')(visible)
    hidden1 = BatchNormalization()(hidden1)
    hidden1 = keras.activations.relu(hidden1)
    
    # regression output
    out_reg = Dense(1,kernel_initializer='he_normal',activation='linear')(hidden1)

    # define model
    model = Model(inputs=visible, outputs=out_reg)
    print(model.summary())
    
    ##Callbacks
    lr_scheduler = LearningRateScheduler(scheduler)
    ckpt = keras.callbacks.ModelCheckpoint('model_weights.keras')
    callbacks = [lr_scheduler, EarlyStopping(monitor='val_loss',patience=50)]
    model.compile(loss=rmse,
                  optimizer='Adam')
    
    return model, callbacks

def get_prediction(model, data_dict, corrector):
    """
    Getting predictions over timeseries by iterating the trained model in order over the trained timeseries
    and then using the predictions to build the shoreline data.
    This is how an actual model would be used in practice, unlike the usual presentation of other NN-based timeseries models.

    inputs:
    model: trained keras model
    data_dict: this is created in setup_data()
    corrector: int, frequency in days to correct the model
    outputs:
    predictions (np.ndarray): these are the predicted shoreline positions (m)
    y_hats1 (np.ndarray): these are the predicted shoreline daily changes (m)
    """
    X = data_dict['X']
    y_hats1 = [None]*len(X)
    i=0
    for x in X:
        y_hat1 = model.predict(np.array([x]))*5
        y_hat1 = y_hat1[0][0]
        y_hats1[i] = y_hat1
        i=i+1
    predictions = [None]*len(data_dict['df'])
    start_pos = data_dict['df']['position'].iloc[0]
    for i in range(len(y_hats1)):
        if i==0:
            prediction = start_pos
        elif i % corrector == 0:
            prediction = data_dict['df']['position'].iloc[i]
        else:
            prediction = y_hats1[i] + predictions[i-1]
        predictions[i] = prediction
    return predictions, y_hats1

def get_forecast(model, data_dict):
    """
    This forecasts the model beyond the observed time period.
    Beware of results, deep learning models are usually overfit!
    They also are not good at short timeseries data. They need a lot of features to be useful.
    A single 20 year timeseries is not that much data for this model to work with.

    inputs:
    model: trained Keras model
    data_dict: this is constructed with setup_data

    outputs:
    forecasts (np.ndarray): array of forecasts, here it goes to 20% beyond the original timeseries length
    y_hats1 (np.ndarray): the forecasted daily shoreline changes (m)
    """

    ##Get record length
    record_length = data_dict['record_length']

    ##assign new record length
    new_record_length = int(record_length*1.2)

    ##get number of forecasted timesteps
    num_forecast = new_record_length-record_length+1

    ##making array for forecast timesteps
    x = np.arange(0, new_record_length, 1)/new_record_length
    x_forecast = np.hstack([x[record_length-1:].reshape((num_forecast,-1))])

    ##get the forecasted shoreline changes
    y_hats1 = [None]*len(x_forecast)
    i=0
    for t in x_forecast:
        y_hat1 = model.predict(np.array([t]))*5
        y_hat1 = y_hat1[0][0]
        y_hats1[i] = y_hat1
        i=i+1

    ##get the forecasted shoreline positions
    forecasts = [None]*len(y_hats1)
    start_pos = data_dict['df']['position'].iloc[-1]
    for i in range(len(y_hats1)):
        if i==0:
            forecast = start_pos
        else:
            forecast = y_hats1[i] + forecasts[i-1]
        forecasts[i] = forecast
        
    return forecasts, y_hats1
    
def setup_data(csv_path,
               forcing_vars,
               split_percent,
               idxes):
    """
    Makes a dictionary with all the relevant info for trianing/testing/validation data

    inputs:
    csv_path (str): path to the csv with the data (columns are date, position, Hs, Tp, Wvx, Wvy)
    forcing_vars (list): list of which forcing variables to use ['time', 'position', 'Hs', 'Tp', 'Wvx', 'Wvy']
    split_percent (float): fraction of timeseries for training data, the rest get split half and half validation and test
    idxes (np.ndarray): the shuffled indices to mix up the timeseries before train/val/test split

    outputs:
    data_dict (dictionary): take a look at what's inside
    """
    ##read in csv, handle the datetime strings
    df = pd.read_csv(csv_path)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S+00:00')
    except:
        try:
            df['date']  = pd.to_datetime(df['date'], format='%m/%d/%Y')
        except:
            df['date']  = pd.to_datetime(df['date'], format='%Y-%m-%d')

    ##Get the shoreline daily difference and scale it to a max value of 5, append 0 to the front for the start of the timeseries
    dy = np.diff(df['position'].values)/5
    dy = np.concatenate((np.array([0]), dy))
    
    
    ## Get x variables, I am scaling everything to a max value I have decided upon.
    record_length = len(df)
    Xs = []
    for forcing_var in forcing_vars:
        try:
            x = df[forcing_var].values.reshape((record_length, 1))
            if forcing_var == 'Hs':
                x = x/10
            if forcing_var == 'Tp':
                x = x/30
            if forcing_var == 'Wvx':
                x = x/40
            if forcing_var == 'Wvy':
                x = x/40
            Xs.append(x)
        except:
            pass

    ##Need to reshape into tensors
    if ['time'] == forcing_vars:
        time = np.arange(0, record_length, 1).reshape((record_length,1))
        time = time/(record_length*1.2)
        Xs.append(time)
    elif 'time' in forcing_vars and len(forcing_vars)>1:
        time = np.arange(0, record_length, 1).reshape((record_length,1))
        time = time/(record_length)
        Xs.append(time)

    ##Stack all the forcing variables
    X = np.hstack(Xs)

    ##Stack the desired outputs
    Y = np.hstack([dy.reshape((len(dy),1))
                   ]
                  )
    
    ##shuffle it all up
    X_shuffled = X[idxes]
    Y_shuffled = Y[idxes]
    Y_reg = Y_shuffled[:,0]

    ##get train/test/val lengths
    num_forcing = len(forcing_vars)
    num_train = int(split_percent*record_length)
    num_val = int((record_length - num_train)/2)
    num_test = record_length - num_train - num_val

    ##Split train/test/val
    x_train = X_shuffled[:num_train]
    y_train_reg = Y_reg[:num_train]
    x_val = X_shuffled[num_train:num_train+num_val]
    y_val_reg = Y_reg[num_train:num_train+num_val]
    x_test = X_shuffled[num_train+num_val:]
    y_test_reg = Y_reg[num_train+num_val:]

    ##make the data dictionary
    data_dict = {'df':df,
                 'X':X,
                 'Y':Y,
                 'x_train':x_train,
                 'y_train_reg':y_train_reg,
                 'x_val':x_val,
                 'y_val_reg':y_val_reg,
                 'x_test':x_test,
                 'y_test_reg':y_test_reg,
                 'record_length':record_length,
                 'num_forcing':num_forcing,
                 'num_train':num_train,
                 'num_val':num_val,
                 'num_test':num_test
                 }
    
    return data_dict

# load dataset
def multilayer_perceptron(csv_path,
                          output_folder,
                          name,
                          forcing_vars,
                          batch_size,
                          num_epochs,
                          split_percent,
                          trials,
                          corrector,
                          idxes):
    """
    dy = f(time, wave_height, wave_period, wind_vel_x, wind_vel_y)
    Closest to solving a differential equation I have been in a while.

    inputs:
    csv_path (str): path to the timeseries data (columns date, position, Hs, Tp, Wvx, Wvy)
    output_folder (str): path to save results
    name (str): give the trial a name
    forcing_vars (list): list of which forcing variables to use ['time', 'Hs', 'Tp', 'Wvx', 'Wvy']
    batch_size (int): I used 256 to smooth the loss curve
    num_epochs (int): I used 1000 and have an early stopping callback on validation loss
    split_percent (float): amount of data to use for training, try 0.50
    trials (int): number of times to repeat training of the model, yay monte carlo-esque trials
    corrector (int): frequency in days to provide the correct position so that the model doesn't get too crazy
    idxes (np.ndarray): the shuffled indices send to setup_data()
    """

    ##making the output directory
    save_dir = os.path.join(output_folder, name)
    try:
        os.mkdir(save_dir)
    except:
        pass
    print(forcing_vars)

    ##getting the split data
    data_dict = setup_data(csv_path,
                           forcing_vars,
                           split_percent,
                           idxes)

    ##getting record lengths for prediction and forecast
    neurons = len(forcing_vars)+1
    record_length = data_dict['record_length']
    new_record_length = int(record_length*1.2)
    num_forecast = new_record_length-record_length+1
    
    ##Making arrays to hold predictions and forecasts
    predict_array = np.zeros((trials, record_length))

    y_hats1_array = np.zeros((trials, record_length))

    if forcing_vars == ['time']:
        forecast_array = np.zeros((trials, num_forecast))
        y_hats1_forecast_array = np.zeros((trials, num_forecast))

    ##lists to hold test losses and training/val losses
    test_results = [None]*trials
    histories = [None]*trials
    
    ##Train Model
    for trial in range(trials):
        model, callbacks = define_model(data_dict['num_forcing'], neurons)
        history = model.fit(x=data_dict['x_train'],
                            y=data_dict['y_train_reg'],
                            batch_size=batch_size,
                            epochs=num_epochs,
                            callbacks=callbacks,
                            validation_data=(data_dict['x_val'], data_dict['y_val_reg']),
                            verbose=2,
                            shuffle=False)
        print('Test Results')
        test_results[trial] = model.evaluate(x=data_dict['x_test'], y=data_dict['y_test_reg'])

        ##Put trial results in respective arrays
        histories[trial] = history
        prediction, y_hats1 = get_prediction(model, data_dict, corrector)
        predict_array[trial,:] = prediction
        y_hats1_array[trial, :] = y_hats1

        ##Only forecast if time is the only variable, we don't have forecasted winds or waves, yet
        if forcing_vars == ['time']:
            forecasts, y_hats1_forecast = get_forecast(model, data_dict)
            forecast_array[trial, :] = forecasts
            y_hats1_forecast_array[trial, :] = y_hats1_forecast
        reset_keras()

    ##Getting the confidence intervals for the predicted shoreline positions during the observed time period
    pred_mean = np.mean(predict_array, axis=0)
    pred_upper_conf_interval = np.quantile(predict_array, 1, axis=0)
    pred_lower_conf_interval = np.quantile(predict_array, 0, axis=0)

    ##means and bounds for predicted shoreline daily changes
    y_hats1_mean = np.mean(y_hats1_array, axis=0)
    y_hats1_upper_conf_interval = np.quantile(y_hats1_array, 1, axis=0)
    y_hats1_lower_conf_interval = np.quantile(y_hats1_array, 0, axis=0)

    ##need to get this to get datetimes
    df = data_dict['df']
    
    ##Plot for observed time period
    plt.rcParams["figure.figsize"] = (16,4)
    plt.plot(df['date'], df['position'], color='k', label='Observed Shoreline Position')
    plt.plot(df['date'], pred_mean, label='MLP Mean', color='violet')
    plt.fill_between(df['date'], pred_lower_conf_interval, pred_upper_conf_interval, color='violet', alpha=0.4, label='MLP Min-Max Interval')
    ax = plt.gca()
    ylim = ax.get_ylim()
    plt.xlabel('Time (UTC)')
    plt.ylabel('Cross-Shore Position (m)')
    plt.legend()
    plt.minorticks_on()
    plt.xlim(min(df['date']), max(df['date']))
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, name, 'prediction.png'), dpi=300)
    plt.close()

    ##Plot each trial monte carlo style
    plt.plot(df['date'], df['position'], color='k', label='Observed Shoreline Position')
    for prediction in predict_array:
        plt.plot(df['date'], prediction)
    plt.ylabel('Cross-Shore Position (m)')
    plt.legend()
    plt.minorticks_on()
    plt.xlim(min(df['date']), max(df['date']))
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, name, 'trials.png'), dpi=300)
    plt.close()
    
    ##Plot Change
    plt.plot(df['date'], np.concatenate((np.array([0]), np.diff(df['position'].values))), color = 'navy', label='Observed')
    plt.plot(df['date'], y_hats1_mean, color='violet', label='NN Mean')
    plt.fill_between(df['date'], y_hats1_lower_conf_interval, y_hats1_upper_conf_interval, color='violet', alpha=0.4, label='NN Min-Max Interval')
    plt.xlabel('Time (UTC')
    plt.ylabel('Shoreline Daily Change (m)')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_folder, name, 'change.png'), dpi=300)
    plt.close()

    ##plot forecasts monte carlo style
    if forcing_vars == ['time']:
        for forecast in forecast_array:
            plt.plot(forecast)
        plt.ylabel('Shoreline Daily Change (m)')
        plt.savefig(os.path.join(output_folder, name, 'forecast.png'), dpi=300)
        plt.close()

    ##plot forecasts mean and bounds
    for history in histories:
        plot_history(history)
    plt.savefig(os.path.join(output_folder, name, 'loss_curve.png'), dpi=300)
    plt.close()
    return test_results

