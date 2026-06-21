import cv2
import numpy as np
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "model" / "best.onnx"
print("model_path:", model_path)
print("exists:", model_path.exists())

classes = {
    0: "fractured",
    1: "not fractured"
}

net = cv2.dnn.readNetFromONNX(str(model_path))

DATASET_DIR = BASE_DIR / "dataset" / "Bone_Fracture_Binary_Classification"

test_dirs = {
    "fractured": DATASET_DIR / "test" / "fractured",
    "not fractured": DATASET_DIR / "test" / "not fractured"
}

def preprocess_none(img):
    return img

def preprocess_clahe(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

def preprocess_unsharp_masking(img, sigma=3, strength=1.5):
    blurred = cv2.GaussianBlur(img, (0,0), sigmaX=sigma)
    sharpened = cv2.addWeighted(img, strength, blurred, -(strength-1), 0)
    return sharpened

def preprocess_normalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)

def preprocess_norm_sharp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(norm, (0,0), 1)
    sharp = cv2.addWeighted(norm, 1.2, blur, -0.2, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

def preprocess_median(img):
    return cv2.medianBlur(img, 3)

def preprocess_histogram(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

preprocessing = {
    "none":                 preprocess_none,
    "clahe":                preprocess_clahe,
    "unsharp_masking":      preprocess_unsharp_masking,
    "normalization":        preprocess_normalization,
    "median_blur":          preprocess_median,
    "norm_sharp":           preprocess_norm_sharp,
    "equalize_histogram":   preprocess_histogram
}

examples = {}

for prep_name, prep_function in preprocessing.items():
    print("\n=========================================")
    print(f"Testing preprocessing: {prep_name}")
    print("=========================================")

    total = 0
    correct = 0
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for true_label, folder_path in test_dirs.items():
        for file_path in folder_path.iterdir():
            if not file_path.is_file():
                continue

            img = cv2.imread(str(file_path))
            if img is None:
                continue

            processed_img = prep_function(img)

            if prep_name not in examples:
                examples[prep_name] = processed_img.copy()

            blob = cv2.dnn.blobFromImage(
                processed_img, scalefactor=1/255.0, size=(256,256), swapRB=True
            )

            net.setInput(blob)
            output = net.forward()

            probs = output[0]
            class_id = probs.argmax()
            prediction = classes[class_id]

            total += 1
            if prediction == true_label:
                correct += 1

            if true_label == "fractured" and prediction == "fractured":
                true_positive += 1
            elif true_label == "not fractured" and prediction == "fractured":
                false_positive += 1
            elif true_label == "not fractured" and prediction == "not fractured":
                true_negative += 1
            elif true_label == "fractured" and prediction == "not fractured":
                false_negative += 1

    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {100*correct/total:.2f}%")
    print(f"TP: {true_positive}")
    print(f"FP: {false_positive}")
    print(f"TN: {true_negative}")
    print(f"FN: {false_negative}")

images = []
for name, img in examples.items():
    img = cv2.resize(img, (256,256))
    cv2.putText(img, name, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    images.append(img)

rows = []
for i in range(0, len(images), 4):
    row = images[i:i+4]
    while len(row) < 4:
        row.append(np.zeros_like(images[0]))
    rows.append(cv2.hconcat(row))

collage = cv2.vconcat(rows)
cv2.imwrite(str(BASE_DIR / "code" / "preprocessing_collage.png"), collage)

print("\nImage collage saved as preprocessing_collage.png")
