import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model
import dataload

BATCH_SIZE = 32
IMG_SIZE = (320, 200)
AUTOTUNE = tf.data.AUTOTUNE

# Load datasets using the dataload module's stratified split
train_ds, test_ds = dataload.split_dataset_stratified(
    csv_path="data/clean_dataset.csv",
    train_frac=0.8,
    batch_size=BATCH_SIZE,
    random_state=42,
)

# Map to (image, label) only
def images_and_labels_only(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(lambda features, label: (features["image_input"], label),
                       num_parallel_calls=AUTOTUNE)

image_train_ds = images_and_labels_only(train_ds)
image_test_ds = images_and_labels_only(test_ds)

# Ensure dataset is shuffled and prefetched
image_train_ds = image_train_ds.shuffle(1024).prefetch(AUTOTUNE)
image_test_ds = image_test_ds.prefetch(AUTOTUNE)

# ------------------------------
# Build image model with 128-d embedding + classifier head
# ------------------------------
def bili_model_with_classifier(image_shape=IMG_SIZE, embedding_dim=128, num_classes=5):
    input_shape = image_shape + (3,)

    # Load a MobileNetV2 base; replace weights path or set weights='imagenet' to use TF Hub weights
    base_model_path = "imagenet_base_model/without_top_mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_160_no_top.h5"
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=base_model_path
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image_input")
    # preprocess_input for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)

    # Embedding output (this is what we will extract later)
    feature_output = layers.Dense(embedding_dim, activation=None, name="image_embedding")(x)

    # Classification head for pretraining
    classifier_output = layers.Dense(num_classes, activation="softmax", name="classifier_output")(feature_output)

    # Model returns both embedding and classifier outputs
    model = Model(inputs=inputs, outputs=[feature_output, classifier_output], name="image_model_with_classifier")
    return model

# Instantiate model
num_classes = 5
model = bili_model_with_classifier(IMG_SIZE, embedding_dim=128, num_classes=num_classes)

# Compile: use sparse_categorical_crossentropy for 0..4 labels and train the classifier output
# We only provide loss for the classifier output; embedding output has no direct loss here.
losses = {
    "classifier_output": tf.keras.losses.SparseCategoricalCrossentropy()
}
metrics = {
    "classifier_output": ["accuracy"]
}
# Keras needs a loss for every output; set loss weight 0 for embedding (or omit by naming losses)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=losses,
              metrics=metrics)

# For training, Keras expects y values corresponding to each model output.
# We can provide y as just the classifier labels when fitting by using a dict or tuple.
def prepare_for_keras(ds: tf.data.Dataset):
    # ds yields (image, label). Convert to ((embedding_input...), {classifier_output: label})
    # Keras will feed image to the model inputs and compare classifier_output to label.
    def map_fn(image, label):
        # For a model with outputs [image_embedding, classifier_output],
        # Keras fit accepts y as a dict mapping output names to labels:
        return image, {"classifier_output": label}
    return ds.map(map_fn, num_parallel_calls=AUTOTUNE)

train_ready = prepare_for_keras(image_train_ds)
test_ready = prepare_for_keras(image_test_ds)

# Train classifier head (pretraining)
initial_epochs = 10
history = model.fit(train_ready, validation_data=test_ready, epochs=initial_epochs)

# After training, extract the embedding-only model
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("image_embedding").output, name="image_embedding_model")
# Optionally, save both the full model and embedding model
model.save("image_model_with_classifier.h5")
embedding_model.save("image_embedding_model.h5")
print("Saved image_model_with_classifier.h5 and image_embedding_model.h5")