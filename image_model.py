import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import dataload


BATCH_SIZE = 32
IMG_SIZE = (320, 200)
# Define the preprocess input function from MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE
# Load datasets using the dataload module's stratified split
train_ds, test_ds = dataload.split_dataset_stratified(
                            csv_path="data/clean_dataset.csv",
                            train_frac=0.8,
                            batch_size=32,
                            random_state=42)


def images_and_labels_only(dataset: tf.data.Dataset) -> tf.data.Dataset:
    """
    Map dataset (features_dict, label) -> (image_tensor, label)
    Assumes features_dict contains "image_input".
    """
    return dataset.map(lambda features, label: (features["image_input"], label),
                       num_parallel_calls=AUTOTUNE)
# Quick: map to images + labels only (useful if you don't need to split)
image_train_ds = images_and_labels_only(train_ds) 
image_test_ds = images_and_labels_only(test_ds)

def bili_model(image_shape=IMG_SIZE):
    ''' Define a tf.keras model for softmax classification out of the MobileNetV2 model
    Arguments:
        image_shape -- Image width and height
    Returns:
        tf.keras.model
    '''

    input_shape = image_shape + (3,)
    
    base_model_path="imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    
    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                   include_top=False,  
                                                   weights=base_model_path)
    
    # freeze the base model by making it non trainable
    base_model.trainable = False

    # create the input layer (Same as the imageNetv2 input size)
    inputs = tf.keras.Input(shape=input_shape) 
    
    # data preprocessing using the same weights the model was trained on
    x = preprocess_input(inputs) 

    # set training to False to avoid keeping track of statistics in the batch norm layer
    x = base_model(x, training=False) 
    
    # add the new classification layers
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
     # include dropout with probability of 0.4 to avoid overfitting
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # feature vector output
    feature_output = tf.keras.layers.Dense(128, activation=None, name="image_embedding")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=feature_output)
    
    return model

model = bili_model(IMG_SIZE)
base_learning_rate = 0.001
metrics=['accuracy']
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)

# Train the model
initial_epochs = 20

history = model.fit(image_train_ds, validation_data=image_test_ds, epochs=initial_epochs)
model.save("image_embedding_model.h5")

