import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import Counter
import pandas as pd
# Import data loader
from dataload import split_dataset_stratified, IMG_SIZE, AUTOTUNE

CSV_PATH = "data/clean_dataset.csv"
BATCH_SIZE = 32

def print_element_spec(ds, name="dataset"):
    print(f"\n{name} element_spec:")
    print(ds.element_spec)

def count_samples(ds):
    # ds yields (images, labels) batches; this sums batch sizes to get total samples
    total = ds.reduce(tf.constant(0), lambda acc, batch: acc + tf.shape(batch[0])[0])
    return int(total.numpy())

def label_distribution_from_df(csv_path):
    df = pd.read_csv(csv_path,
                     header=0,
                     names=["views", "image_path", "title"],
                     quotechar='"',
                     escapechar='\\',
                     engine="python")
    # Reuse your bucketize function by importing from dataload if you want consistency.
    # Here we recompute same buckets quickly:
    def bucketize_view_count(v):
        v = int(float(v))
        if v < 1000:
            return 0
        elif v < 5000:
            return 1
        elif v < 20000:
            return 2
        elif v < 100000:
            return 3
        else:
            return 4
    buckets = df["views"].map(bucketize_view_count)
    return dict(buckets.value_counts().sort_index())

def label_distribution_from_dataset(ds):
    # Unbatch for per-sample iteration and count labels
    c = Counter()
    for _, label in ds.unbatch().as_numpy_iterator():
        c[int(label)] += 1
    # Ensure keys 0..4 exist even if zero
    return {i: c.get(i, 0) for i in range(5)}

def show_one_batch_images(ds, n=9):
    for images, labels in ds.take(1):
        images_np = images.numpy()
        labels_np = labels.numpy()
        print("Batch images shape:", images_np.shape)
        print("Batch labels shape:", labels_np.shape, "labels sample:", labels_np[:min(10, len(labels_np))])
        # Plot the first n images
        n = min(n, images_np.shape[0])
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axs = axs.flatten()
        for i in range(n):
            axs[i].imshow(images_np[i])
            axs[i].set_title(f"label={labels_np[i]}")
            axs[i].axis("off")
        for j in range(n, len(axs)):
            axs[j].axis("off")
        plt.tight_layout()
        plt.show()
        break

print("CSV label distribution (raw):")
print(label_distribution_from_df(CSV_PATH))

train_ds, test_ds = split_dataset_stratified(csv_path=CSV_PATH,
                                                train_frac=0.9,
                                                batch_size=BATCH_SIZE,
                                                random_state=42)

print_element_spec(train_ds, "train_ds")
print_element_spec(test_ds, "test_ds")

train_count = count_samples(train_ds)
test_count = count_samples(test_ds)
print(f"\nTrain samples: {train_count}")
print(f"Test samples: {test_count}")
print(f"Total from split: {train_count + test_count}")

print("\nLabel distribution in train dataset:")
print(label_distribution_from_dataset(train_ds))
print("\nLabel distribution in test dataset:")
print(label_distribution_from_dataset(test_ds))

print("\nInspect one batch from train_ds (shapes, dtypes, images):")
show_one_batch_images(train_ds, n=9)

# Save train and test datasets back to CSV files
output_dir = "data"
output_file_csv = "train.csv"
output_path_csv = os.path.join(output_dir, output_file_csv)
train_ds.to_csv(output_path_csv, index=False) 

output_file_csv = "test.csv"
output_path_csv = os.path.join(output_dir, output_file_csv)
test_ds.to_csv(output_path_csv, index=False) 

print("Dataset successfully saved")


