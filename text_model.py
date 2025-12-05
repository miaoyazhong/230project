import tensorflow as tf
from data_load import split_text_dataset_stratified, MAX_LEN, AUTOTUNE
from transformers import  TFBertModel
from plot import plot_metrics
import os

CSV_PATH = "data/train_split.csv"
BATCH_SIZE = 32
EPOCHS = 3
NUM_LABELS = 5  # buckets 0..4

def build_text_model(num_labels=NUM_LABELS, max_len=MAX_LEN, bert_name="bert-base-chinese"):
    # Load a TensorFlow BERT encoder (no classification head)
    bert = TFBertModel.from_pretrained(bert_name)
    bert.trainable = False  # Freeze BERT layers

    input_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert(input_ids, attention_mask=attention_mask)
    pooled = bert_outputs.pooler_output  # (batch, hidden_size)

    x = tf.keras.layers.Dense(256, activation="relu")(pooled)
    text_embedding = tf.keras.layers.Dense(128, activation=None, name="text_embedding")(x)
    logits = tf.keras.layers.Dense(num_labels, activation="softmax", name="classifier")(text_embedding)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=logits, name="text_classifier")
    return model


def main():
    print("Building text-only datasets from CSV...")
    train_ds, test_ds = split_text_dataset_stratified(CSV_PATH, train_frac=0.9, batch_size=BATCH_SIZE, random_state=42)

    # For training shuffle the training dataset
    train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    # Build model
    model = build_text_model()
    model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
    )
 
    model.summary()

    # Train
    history = model.fit(train_ds, validation_data=test_ds, epochs=EPOCHS)
    plot_metrics(history, None) 
    # Save full classifier model
    classifier_path = "text_classifier.keras"
    model.save(classifier_path)
    print(f"Saved classifier model to {classifier_path}")

    # Export embedding-only model
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("text_embedding").output, name="text_embedding_model")
    embedding_path = "text_embedding_model.keras"
    embedding_model.save(embedding_path)
    print(f"Saved embedding model to {embedding_path}")

if __name__ == "__main__":
    main()