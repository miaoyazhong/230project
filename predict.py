import tensorflow as tf
from transformers import BertTokenizer
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
MAX_LEN = 32

# -------------------------------------------------------
# Load models
# -------------------------------------------------------
fuse_model = load_model("multimodal_virality_model.h5")
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
bucket_ranges = {
    0: "0–1,000 views",
    1: "1,000–5,000 views",
    2: "5,000–20,000 views",
    3: "20,000–100,000 views",
    4: "100,000+ views"
}

# -------------------------------------------------------
# Preprocess image
# -------------------------------------------------------
def preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = img / 255.0
    return img


# -------------------------------------------------------
# Preprocess text
# -------------------------------------------------------
def preprocess_text(title):
    encoded = tokenizer(
        title,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="tf"
    )
    return encoded["input_ids"], encoded["attention_mask"]


# -------------------------------------------------------
# Prediction function
# -------------------------------------------------------
def predict_virality(image_path, title):
    img = preprocess_image(image_path)
    ids, mask = preprocess_text(title)

    pred = fuse_model.predict({
        "image_input": tf.expand_dims(img, axis=0),
        "input_ids": ids,
        "attention_mask": mask
    })
    probs = pred[0]
    bucket_id = np.argmax(probs)
    confidence = probs[bucket_id]

    print("Predicted bucket:", bucket_id)
    print("Expected views:", bucket_ranges[bucket_id])
    print("Confidence:", confidence)

    # prob = float(pred[0][0])
    return bucket_id 


# -------------------------------------------------------
# CLI Example
# -------------------------------------------------------
if __name__ == "__main__":
    img = "test/alpaca/2c6af9b53f4cb15a.jpg"
    title = "你绝对想不到的羊驼冷知识！"

    prob = predict_virality(img, title)
    print(f"Predicted viral probability: {prob:.4f}")
