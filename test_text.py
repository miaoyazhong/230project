import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# # Load dataset
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
# print(viral_df.head(5))
# print(no_viral_df.head(5))

# # Combine
# df = pd.concat([viral_df, no_viral_df], ignore_index=True)
# print(f"Dataset loaded: {len(df)} samples")


# #Split train/test
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
test_df = pd.read_csv("data/test_title.csv", header=0)
test_dataset = Dataset.from_pandas(test_df)

#load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert_click_predictor")
model = BertForSequenceClassification.from_pretrained("bert_click_predictor")

def tokenize_function(examples):
    return tokenizer(
        examples["title"],
        padding="max_length",
        truncation=True,
        max_length=64
    )

test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

#predict the click probability 
def predict_click(title):
    inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        click_prob = probs[0][1].item()
    return click_prob

# Run model on test dataset
model.eval()

# Collect predictions and true labels
all_preds = []
all_labels = []

for batch in torch.utils.data.DataLoader(test_dataset, batch_size=16):
    with torch.no_grad():
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print("=== Test Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Not Viral", "Viral"]))

# Plot metrics
metrics = {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}
plt.figure(figsize=(6,4))
plt.bar(metrics.keys(), metrics.values(), color=["skyblue", "lightgreen", "salmon", "plum"])
plt.ylim(0, 1)
plt.title("Model Performance on Test Dataset")
plt.ylabel("Score")
plt.show()

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=["Not Viral", "Viral"], cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
