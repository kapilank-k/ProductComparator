import random
from base_training_data import TRAIN_DATA

random.shuffle(TRAIN_DATA)
split_index = int(0.8 * len(TRAIN_DATA))

train_samples = TRAIN_DATA[:split_index]
test_samples = TRAIN_DATA[split_index:]

# Save train data
with open("train_data.py", "w", encoding="utf-8") as f:
    f.write("TRAIN_DATA = [\n")
    for text, ann in train_samples:
        f.write(f"    ({text!r}, {ann}),\n")
    f.write("]\n")

# Save test data
with open("test_data.py", "w", encoding="utf-8") as f:
    f.write("TEST_DATA = [\n")
    for text, ann in test_samples:
        f.write(f"    ({text!r}, {ann}),\n")
    f.write("]\n")

print("✅ Train/Test split done — saved as train_data.py and test_data.py")
