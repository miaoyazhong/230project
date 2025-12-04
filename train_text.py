import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
import os
import dataload

# # 1. Load and label datasets
# # Read entire lines (since some titles contain commas)
# def load_titles(filename):
#     # Read each line as one record, even if it has commas or quotes
#     with open(filename, "r", encoding="utf-8") as f:
#         lines = f.readlines()

#     # Strip newlines and surrounding quotes if present
#     titles = [line.strip() for line in lines if line.strip()]

#     # Return a DataFrame
#     return pd.DataFrame({"title": titles})

# # Add labels: 1 = viral, 0 = not viral
# viral_df = load_titles("data/viral_title.csv")
# viral_df["label"] = 1

# no_viral_df = load_titles("data/no_viral_title.csv")
# no_viral_df["label"] = 0

# # Combine
# df = pd.concat([viral_df, no_viral_df], ignore_index=True)
# print(f"Dataset loaded: {len(df)} samples")


# 2.Split train/test
# df = pd.read_csv("data/train_title.csv", header=0)
df = dataload.load_dataset("data/train_title.csv")
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
test_dataset = Dataset.from_pandas(test_df)
train_dataset = Dataset.from_pandas(train_df)


# 3.Tokenize Chinese titles

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

def tokenize_function(examples):
    return tokenizer(
        examples["title"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


#4.Load pretrained BERT model
bert = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

#5.Training setup
input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")
bert_outputs = bert(input_ids, attention_mask=attention_mask)
pooled = bert_outputs.pooler_output    # shape = (batch, 768)

x = tf.keras.layers.Dense(256, activation="relu")(pooled)
text_embedding = tf.keras.layers.Dense(128, activation=None, name="text_embedding")(x)
model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=text_embedding)

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)


# 6. Train and save
trainer.train()

# model.save_pretrained("bert_click_predictor")
# tokenizer.save_pretrained("bert_click_predictor")
model.save("text_embedding_model.h5")

print("Training complete. Model saved to 'bert_click_predictor'.")