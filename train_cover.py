import matplotlib.pyplot as plt
import json
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
from plot import plot_metrics


BATCH_SIZE = 32
IMG_SIZE = (320, 200)
# Load datasets and split into training and validation sets
# directory = "test/"
directory = "dataset/"
train_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='training',
                                             seed=42)
validation_dataset = image_dataset_from_directory(directory,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE,
                                             validation_split=0.1,
                                             subset='validation',
                                             seed=42)

# Get class names
class_names = train_dataset.class_names
print("Class names:", class_names)
# # Compute class weights to handle class imbalance
# counts = [1947, 50]
# labels = np.array([0] * counts[0] + [1] * counts[1])
# class_weights_array = compute_class_weight(
#     class_weight='balanced',
#     classes=np.array([0, 1]),
#     y=labels
# )
# class_weights = {
#     0: class_weights_array[0], 
#     1: class_weights_array[1]   
# }
# print(class_weights)

def expand_labels(image, label):
    return image, tf.expand_dims(label, axis=-1)
# Expand label dimensions to use f1_score metric
train_dataset = train_dataset.map(expand_labels)
validation_dataset = validation_dataset.map(expand_labels)

# Optimize dataset performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
# Define the preprocess input function from MobileNetV2
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def bili_model(image_shape=IMG_SIZE):
    ''' Define a tf.keras model for binary classification out of the MobileNetV2 model
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
    
    # add the new Binary classification layers
    # use global avg pooling to summarize the info in each channel
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # include dropout with probability of 0.2 to avoid overfitting
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    # # fully connected layer with ReLU activation
    x = tf.keras.layers.Dense(256, activation="relu")(x)

    # use a prediction layer with one neuron (as a binary classifier only needs one)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
 
    model = tf.keras.Model(inputs, outputs)
    
    return model

# Create and compile the model
model = bili_model(IMG_SIZE)
base_learning_rate = 0.001
f1_metric = tf.keras.metrics.F1Score(name='f1_score', threshold=0.5)
#metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), f1_metric]
metrics=['accuracy', f1_metric]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=metrics)

# Train the model
initial_epochs = 20

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=initial_epochs)

# # Fine-tune the model
# base_model = model.get_layer('mobilenetv2_1.00_200')
# fine_tune_at = 120
# base_model.trainable = False
# # unfreeze all the layers after the `fine_tune_at` layer
# for layer in base_model.layers[fine_tune_at:]:
#     layer.trainable = True

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate*0.1),
#               loss=tf.keras.losses.BinaryCrossentropy(),
#               metrics=['accuracy', tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall"), f1_metric])

# fine_tune_epochs = 5
# total_epochs =  initial_epochs + fine_tune_epochs

# history_fine = model.fit(train_dataset,
#                          epochs=total_epochs,
#                          initial_epoch=history.epoch[-1],
#                          validation_data=validation_dataset)

plot_metrics(history, None) 


