import os
import shutil
import pandas as pd

# Paths (CHANGE if needed)
image_dir = "DATASET"
train_labels_file = "train_labels.csv"
test_labels_file = "test_labels.csv"

output_dir = "data"

# Emotion mapping
emotion_map = {
    1: "surprise",
    2: "fear",
    3: "disgust",
    4: "happy",
    5: "sad",
    6: "angry",
    7: "neutral"
}

# Create folders
for split in ["train", "val"]:
    for emotion in emotion_map.values():
        os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

# -----------------------------
# Process training data
# -----------------------------
train_df = pd.read_csv(train_labels_file)

for _, row in train_df.iterrows():
    img_name = row[0]
    label = row[1]

    emotion = emotion_map[label]

    src = os.path.join(image_dir, img_name)
    dst = os.path.join(output_dir, "train", emotion, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

# -----------------------------
# Process validation data
# -----------------------------
test_df = pd.read_csv(test_labels_file)

for _, row in test_df.iterrows():
    img_name = row[0]
    label = row[1]

    emotion = emotion_map[label]

    src = os.path.join(image_dir, img_name)
    dst = os.path.join(output_dir, "val", emotion, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset prepared successfully!")