import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure outputs folder exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)
LABELS_CSV = os.path.join(OUTPUTS_DIR, "labels.csv")

def create_labels_csv():
    # If labels.csv exists, delete it first to avoid PermissionError
    if os.path.exists(LABELS_CSV):
        try:
            os.remove(LABELS_CSV)
            print("🗑️ Old labels.csv deleted.")
        except Exception as e:
            print(f"⚠️ Could not delete old labels.csv: {e}")
            return

    rows = []
    classes = sorted(os.listdir(PROCESSED_DATA_DIR))  # A-, A+, AB-, ...

    for label_idx, label in enumerate(classes):
        class_dir = os.path.join(PROCESSED_DATA_DIR, label)

        for fname in os.listdir(class_dir):
            if fname.endswith(".npy"):
                file_path = os.path.join("data", "processed", label, fname)
                rows.append([file_path, label, label_idx])

    with open(LABELS_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "label", "label_idx"])
        writer.writerows(rows)

    print(f"✅ labels.csv created at {LABELS_CSV}")

if __name__ == "__main__":
    create_labels_csv()
