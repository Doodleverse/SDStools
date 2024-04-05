"""
Mark Lundine
Good/bad coastal image classification
adapted from https://keras.io/examples/vision/image_classification_from_scratch/
"""
import os
import numpy as np
import glob
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import shutil

def load_dataset(training_data_directory,
                 image_size,
                 batch_size):
    """
    loads in training data
    inputs:
    training_data_directory (str): path to the training data
    returns:
    train_ds, val_ds
    """
    train_ds, val_ds = keras.utils.image_dataset_from_directory(training_data_directory,
                                                                validation_split=0.2,
                                                                subset="both",
                                                                seed=1337,
                                                                image_size=image_size,
                                                                batch_size=batch_size
                                                                )
    return train_ds, val_ds

def data_augmentation(images):
    """
    applies data augmentation to images
    """
    data_augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.1),
    ]
    return images

def define_model(input_shape, num_classes=2):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

def train_model(model,
                image_size,
                train_ds,
                val_ds,
                model_folder,
                epochs=100):
    model = make_model(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)
    ckpt_file = os.path.join(model_folder, "model_{epoch}.keras")
    epochs = 25
    early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=100, mode='auto', restore_best_weights=True)
    callbacks = [keras.callbacks.ModelCheckpoint(ckpt_file),
                 early_stopping_callback
                 ]
    model.compile(optimizer=keras.optimizers.Adam(3e-4),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")]
                  )
    history = model.fit(train_ds,
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=val_ds
                        )
    return model, history, ckpt_file

def sort_images(inference_df_path,
                inference_images_path,
                ):
    bad_dir = os.path.join(inference_images_path, 'bad')
    good_dir = os.path.join(inference_images_path, 'good')
    dirs = [bad_dir, good_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path) 
        if file['im_classes'].iloc[i] = 'good':
            output_image_path = os.path.join(good_dir, im_name)
        else:
            output_image_path = os.path.join(bad_dir, im_name)
        shutil.move(input_image_path, output_image_path)
        
def run_inference(path_to_model_ckpt,
                  path_to_inference_imgs,
                  output_folder,
                  image_size,
                  result_path):
    
    model = keras.saving.load_model(path_to_model_ckpt)
    im_paths = glob.glob(path_to_inference_imgs+'\*.jpg')
    im_classes = [None]*len(im_paths)
    im_scores = [None]*len(im_paths)
    i=0
    for im_path in im_paths:
        img = keras.utils.load_img(img, target_size=image_size)
        img_array = keras.utils.img_to_array(img)
        img_array = keras.ops.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = float(keras.ops.sigmoid(predictions[0][0]))
        good_score = 1 - score
        bad_score = score
        if good_score>bad_score:
            im_classes[i] = 'good'
            im_scores[i] = good_score
        else:
            im_classes[i] = 'bad'
            im_scores[i] = bad_score

    ##save results to a csv
    df = pd.DataFrame({'im_paths':im_paths,
                       'im_classes':im_classes,
                       'im_scores':im_scores})
    df.to_csv(result_path)
    sort_images(result_path,
                path_to_inference_images)
    return result_path

def sort_images(inference_df_path,
                inference_images_path):
    bad_dir = os.path.join(inference_images_path, 'bad')
    good_dir = os.path.join(inference_images_path, 'good')
    dirs = [bad_dir, good_dir]
    for d in dirs:
        try:
            os.mkdir(d)
        except:
            pass
    inference_df = pd.read_csv(inference_df_path)
    for i in range(len(inference_df)):
        input_image_path = inference_df['im_paths'].iloc[i]
        im_name = os.path.basename(input_image_path) 
        if file['im_classes'].iloc[i] = 'good':
            output_image_path = os.path.join(good_dir, im_name)
        else:
            output_image_path = os.path.join(bad_dir, im_name)
        shutil.move(input_image_path, output_image_path)
    
def plot_history(history,
                 history_save_path):
    """
    This makes a plot of the loss curve
    inputs:
    history: history object from model.fit_generator
    """
    plt.plot(history.history['loss'], color='b')
    plt.plot(history.history['val_loss'], color='r')
    plt.minorticks_on()
    plt.ylabel('Loss (BCE)')
    plt.xlabel('Epoch')
    plt.legend(['Training Data', 'Validation Data'],loc='upper right')
    plt.savefig(history_save_path, dpi=300)
    
def training(path_to_training_data,
             epochs,
             output_folder):
    model_folder = os.path.join(output_folder, 'models')
    history_save_path = os.path.join(model_folder, 'history.png')
    try:
        os.mkdir(model_folder)
    except:
        pass
    image_size = (256, 256)
    train_ds, val_ds = load_dataset(path_to_training_data,
                                    image_shape,
                                    32)
    model = define_model(input_shape=image_size + (3,), num_classes=2)
    model, history, ckpt_file = train_model(model,
                                            image_size,
                                            train_ds,
                                            val_ds,
                                            model_folder,
                                            epochs=epochs)
    plot_history(history, history_save_path)
    return ckpt_file
    


    
    
    
    







    
