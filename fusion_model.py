import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Dropout, Input
from tensorflow.keras.models import Model, load_model

# ---------------------------------------------------
# 1. Load pretrained sub-models (image + text)
# ---------------------------------------------------
image_model = load_model("image_embedding_model.keras")
text_model = load_model("text_embedding_model.keras")

# Freeze both embedding models
image_model.trainable = False
text_model.trainable = False

# ---------------------------------------------------
# 2. Define Inputs for both models
# ---------------------------------------------------
# Image input (200x320x3 typical for MobileNet)
image_input = Input(shape=(200,320, 3), name="image_input")

# Text input (token ids + mask)
text_input_ids = Input(shape=(64,), dtype=tf.int32, name="input_ids")
text_attention = Input(shape=(64,), dtype=tf.int32, name="attention_mask")

# ---------------------------------------------------
# 3. Get embeddings
# ---------------------------------------------------
img_emb = image_model(image_input)             # (128,)
txt_emb = text_model([text_input_ids, text_attention])  # (128,)

# ---------------------------------------------------
# 4. Fuse the embeddings
# ---------------------------------------------------
fusion = Concatenate()([img_emb, txt_emb])  # (256,)

# classifier MLP
x = Dense(256, activation="relu")(fusion)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

# final 5 buckets output
output = Dense(5, activation="softmax", name="virality_output")(x)

# ---------------------------------------------------
# 5. Build the multimodal model
# ---------------------------------------------------
fuse_model = Model(
    inputs=[image_input, text_input_ids, text_attention],
    outputs=output
)

fuse_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

fuse_model.summary()

# ---------------------------------------------------
# 6. Training example
# ---------------------------------------------------
fuse_model.fit(
    x = {
        "image_input": image_array,
        "input_ids": token_ids,
        "attention_mask": attention_masks
    },
    y = labels,
    batch_size = 32,
    epochs = 10,
    validation_split = 0.2
)

fuse_model.save("multimodal_virality_model.keras")
