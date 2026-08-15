# AI-POWERED SKIN LESION CLASSIFIER

## Project Engineering, Learning & Implementation Documentation

**Project Type:** BTech CSE / AI-ML Project
**Application:** Skin lesion image classification
**Initial Goal:** Classify a skin lesion image as benign or malignant
**Primary Dataset:** HAM10000
**Development Environment:** Windows + VS Code + Python + Jupyter
**Current Phase:** Dataset Exploration and Understanding

---

# 1. PROJECT OBJECTIVE

Build a deployable AI-powered application that accepts a skin lesion image and predicts whether the lesion belongs to the project's benign or malignant category.

The project is intended as an **educational screening-support prototype**, not as a replacement for a dermatologist or a clinical diagnostic system.

The complete system will eventually contain:

```text
Mobile/Web Camera
       ↓
Image Upload
       ↓
Image Preprocessing
       ↓
Lightweight CNN
       ↓
Prediction
       ↓
Probability / Confidence
       ↓
User Interface
```

The machine-learning pipeline will eventually be:

```text
HAM10000
    ↓
Dataset Exploration
    ↓
Label Construction
    ↓
Leakage-Safe Split
    ↓
Image Preprocessing
    ↓
Baseline CNN
    ↓
Transfer Learning
    ↓
Class-Imbalance Handling
    ↓
Fine-Tuning
    ↓
Medical-AI Evaluation
    ↓
Model Optimization
    ↓
Deployment
```

---

# 2. PROJECT LEARNING PHILOSOPHY

This project is being built using a **reverse-engineering approach**.

For every major component, the objective is not merely to use a library.

We want to understand:

1. What problem does this component solve?
2. Why is it needed?
3. What happens internally?
4. What are the alternatives?
5. What can go wrong?
6. How do we test it?
7. How does it affect the final application?

For example, instead of simply using a CNN:

```text
model = MobileNet(...)
```

we will first understand:

```text
Image
 ↓
Convolution
 ↓
Feature Map
 ↓
Activation
 ↓
Pooling
 ↓
Feature Extraction
 ↓
Classifier
 ↓
Prediction
```

---

# 3. CURRENT PROJECT STRUCTURE

Current project structure:

```text
skin-lesion-classifier/

├── .venv/
├── backend/
├── data/
│   ├── metadata/
│   ├── processed/
│   └── raw/
│       └── HAM10000/
│           ├── HAM10000_images_part_1/
│           ├── HAM10000_images_part_2/
│           └── HAM10000_metadata.csv
│
├── docs/
├── frontend/
├── models/
├── notebooks/
│   └── 01_dataset_exploration.ipynb
│
├── src/
├── tests/
├── .gitignore
├── README.md
└── requirements.txt
```

The directory structure separates:

* raw data
* processed data
* experiments
* models
* backend
* frontend
* documentation
* tests

This separation will become increasingly important as the project grows.

---

# 4. DEVELOPMENT ENVIRONMENT

## Python Environment

The project now uses a dedicated virtual environment:

```text
.venv
```

The notebook kernel is:

```text
Python (Skin Lesion)
```

The notebook is successfully connected to the project environment.

Important principle:

> The Python interpreter used by the terminal and the Python kernel used by Jupyter must point to the same project environment.

This prevents errors where a package is installed in one Python environment but the notebook executes in another.

---

# 5. DATASET

## HAM10000

HAM10000 contains approximately 10,000 dermatoscopic images of pigmented skin lesions.

The metadata contains fields including:

```text
lesion_id
image_id
dx
dx_type
age
sex
localization
```

### Important distinction

`image_id` identifies an individual image.

`lesion_id` identifies the lesion associated with the image.

Therefore:

```text
lesion
   ├── image 1
   ├── image 2
   └── image 3
```

can occur.

This distinction becomes critical when splitting the dataset.

---

# 6. DATASET EXPLORATION

Notebook:

```text
notebooks/01_dataset_exploration.ipynb
```

## 6.1 Loading the metadata

```python
import pandas as pd

df = pd.read_csv(
    "../data/raw/HAM10000/HAM10000_metadata.csv"
)

df.head()
```

The dataset contains 10,015 metadata records.

Important columns:

```text
lesion_id
image_id
dx
dx_type
age
sex
localization
```

---

# 7. ORIGINAL HAM10000 CLASSES

The `dx` column contains seven diagnostic categories:

| Code    | Category                                      |
| ------- | --------------------------------------------- |
| `akiec` | Actinic keratoses / intraepithelial carcinoma |
| `bcc`   | Basal cell carcinoma                          |
| `bkl`   | Benign keratosis                              |
| `df`    | Dermatofibroma                                |
| `mel`   | Melanoma                                      |
| `nv`    | Melanocytic nevi                              |
| `vasc`  | Vascular lesions                              |

Originally, HAM10000 is therefore a **7-class classification problem**.

Our application, however, initially targets:

```text
Benign
vs
Malignant
```

Therefore we need to construct a binary target.

---

# 8. BINARY TARGET CONSTRUCTION

For the current project prototype, the working mapping is:

### Malignant target = 1

```text
mel
bcc
akiec
```

### Benign target = 0

```text
nv
bkl
df
vasc
```

Implementation:

```python
malignant_classes = ["mel", "bcc", "akiec"]

df["target"] = df["dx"].apply(
    lambda x: 1 if x in malignant_classes else 0
)
```

The mapping should always be documented rather than hidden inside preprocessing code.

---

# 9. CLASS IMBALANCE DISCOVERY

The binary class distribution was visualized.

The resulting graph showed approximately:

```text
Benign      ≈ 8,000+
Malignant   ≈ 2,000
```

Therefore the dataset is approximately:

```text
80% Benign
20% Malignant
```

The approximate ratio is:

```text
4 : 1
```

## Why this matters

A model could achieve high accuracy simply by favoring the majority class.

For example, if approximately 80% of the dataset is benign, a useless model that predicts:

```text
Benign
```

for every image could achieve approximately 80% accuracy.

Therefore:

> Accuracy alone is not sufficient for evaluating this medical-image classifier.

Later evaluation must include:

```text
Confusion Matrix
Sensitivity / Recall
Specificity
Precision
F1-score
ROC-AUC
PR-AUC
```

False negatives are especially important because a malignant lesion predicted as benign is a potentially dangerous error.

---

# 10. VISUAL IMAGE EXPLORATION

Images from classes including:

```text
mel
nv
bkl
```

were successfully visualized.

The images demonstrated substantial variation in:

* lesion color
* lesion shape
* lesion size
* surrounding skin
* lighting
* image artifacts
* hair
* visual texture

This demonstrates why skin-lesion classification is more complicated than a clean educational image-classification dataset.

A CNN must learn relevant visual patterns rather than memorizing superficial properties.

---

# 11. WHY WE MUST NOT TRAIN YET

The current temptation would be:

```text
Load images
↓
Train CNN
↓
Check accuracy
```

This is the wrong workflow.

Before training, we must establish:

```text
Correct labels
        ↓
Correct image paths
        ↓
Class distribution
        ↓
Lesion grouping
        ↓
Leakage-safe dataset split
        ↓
Preprocessing
        ↓
Training
```

Otherwise, the final evaluation may be misleading.

---

# 12. DATA LEAKAGE

A particularly important issue is that multiple images can correspond to the same lesion.

For example:

```text
Lesion A
├── Image A1
├── Image A2
└── Image A3
```

A bad random image split could produce:

```text
TRAIN:
Image A1

TEST:
Image A2
```

The model has already encountered another image of the same lesion.

This can make the test performance artificially optimistic.

Therefore, we will investigate and use a **lesion-level split**.

The desired rule is:

```text
TRAIN
Lesion A
Lesion B
Lesion C

VALIDATION
Lesion D
Lesion E

TEST
Lesion F
Lesion G
```

No lesion should occur across multiple splits.

---

# 13. CURRENT IMPLEMENTATION STATUS

## Completed

* [x] Project repository created
* [x] Project folder structure created
* [x] HAM10000 downloaded
* [x] HAM10000 extracted
* [x] Metadata loaded
* [x] Pandas environment configured
* [x] Seaborn environment configured
* [x] OpenCV configured
* [x] Jupyter configured
* [x] Python project environment configured
* [x] Dataset dimensions inspected
* [x] Metadata columns inspected
* [x] Missing values inspected
* [x] Original 7-class distribution inspected
* [x] Images visualized
* [x] Binary target created
* [x] Binary class imbalance visualized

## Current task

Investigate:

```text
lesion_id
image_id
```

and verify the image-to-metadata mapping.

---

# 14. NEXT EXPERIMENT

Run:

```python
print("Images:", len(df))
print("Unique images:", df["image_id"].nunique())
print("Unique lesions:", df["lesion_id"].nunique())
```

Then:

```python
print(df["target"].value_counts())
```

Then:

```python
print(df["target"].value_counts(normalize=True) * 100)
```

Then:

```python
df["lesion_id"].value_counts().head(10)
```

These experiments answer:

1. How many images exist?
2. Are image IDs unique?
3. How many unique lesions exist?
4. What is the exact class imbalance?
5. How many images can belong to the same lesion?

---

# 15. UPCOMING MODULES

## Module 2: Leakage-Safe Dataset Splitting

Learn:

* train set
* validation set
* test set
* stratification
* grouping
* lesion-level splitting
* data leakage
* reproducibility

---

## Module 3: Image Preprocessing

Learn:

* image loading
* RGB channels
* image dimensions
* resizing
* normalization
* augmentation
* interpolation
* image quality
* artifact handling

---

## Module 4: CNN Fundamentals

Build a CNN from scratch and understand:

```text
Convolution
Filters
Kernels
Feature maps
Stride
Padding
ReLU
Pooling
Flattening
Dense layers
Sigmoid
Binary cross-entropy
```

---

## Module 5: Baseline CNN

Train the first real model.

Measure:

```text
Training loss
Validation loss
Accuracy
Precision
Recall
F1
Confusion matrix
ROC-AUC
```

This becomes our baseline.

---

## Module 6: Class Imbalance

Experiment with:

```text
Class weights
Weighted loss
Augmentation
Oversampling
Focal loss
```

We will compare experiments rather than blindly applying every technique.

---

## Module 7: Transfer Learning

Study:

```text
Pretrained CNN
       ↓
Feature extraction
       ↓
Frozen layers
       ↓
Classification head
       ↓
Fine-tuning
```

Candidate models:

```text
MobileNetV3
EfficientNet
```

The final choice will depend on performance and deployment constraints.

---

## Module 8: Medical-AI Evaluation

Study:

```text
Confusion Matrix
Sensitivity
Specificity
Precision
Recall
F1
ROC-AUC
PR-AUC
Threshold selection
Calibration
```

We will specifically analyze false negatives.

---

## Module 9: Model Optimization

Learn:

```text
Model size
Inference latency
Quantization
Float32
Float16
INT8
```

Goal:

> Reduce model size and inference cost without destroying clinically relevant performance.

---

## Module 10: Application Backend

Build:

```text
FastAPI
   ↓
Image upload
   ↓
Preprocessing
   ↓
Model inference
   ↓
JSON response
```

---

## Module 11: Frontend

Build:

```text
Upload / Camera
      ↓
Image preview
      ↓
Prediction
      ↓
Probability
      ↓
Safety / medical disclaimer
```

---

## Module 12: Deployment

Final architecture:

```text
User
 ↓
Web / Mobile UI
 ↓
Backend API
 ↓
Preprocessing
 ↓
Optimized model
 ↓
Prediction
```

Deployment will be designed around the actual final model and application architecture rather than chosen prematurely.

---

# 16. PROJECT RULES

Throughout this project:

### Rule 1

Never trust accuracy alone.

### Rule 2

Never split medical images blindly without investigating patient/lesion grouping.

### Rule 3

Never use the test set for tuning.

### Rule 4

Never claim that a college prototype is a medical diagnostic device.

### Rule 5

Every major experiment gets documented.

### Rule 6

Every model improvement must be compared against a baseline.

### Rule 7

Keep the raw dataset untouched.

### Rule 8

Use reproducible random seeds wherever appropriate.

### Rule 9

Document failed experiments too.

A failed experiment is useful if we understand why it failed.

---

# 17. CURRENT PROJECT STATUS

```text
Environment             ████████████████████ 100%
Dataset acquisition     ████████████████████ 100%
Dataset inspection      ████████████████████ 100%
Binary target           ████████████████████ 100%
Visual exploration      ████████████████████ 100%

Lesion-level analysis   ███████░░░░░░░░░░░░░  ~35%
Dataset splitting       ░░░░░░░░░░░░░░░░░░░░   0%
Preprocessing           ░░░░░░░░░░░░░░░░░░░░   0%
CNN                     ░░░░░░░░░░░░░░░░░░░░   0%
Transfer learning       ░░░░░░░░░░░░░░░░░░░░   0%
Evaluation              ░░░░░░░░░░░░░░░░░░░░   0%
Deployment              ░░░░░░░░░░░░░░░░░░░░   0%
```

**Current position:**

```text
HAM10000
   ↓
EDA
   ↓
Binary classification
   ↓
Class imbalance
   ↓
YOU ARE HERE
   ↓
Lesion-level analysis
   ↓
Leakage-safe splitting
```

---

# 18. VIVA / EXPLANATION VERSION

If someone asks:

### "What have you done so far?"

A concise explanation is:

> We selected HAM10000 as the initial dataset and first performed exploratory data analysis rather than immediately training a model. We inspected the metadata, identified the seven diagnostic categories, constructed a binary benign-versus-malignant target for our prototype, visualized the class distribution, and found significant class imbalance. We also inspected the actual dermatoscopic images and identified an important dataset issue: multiple images may correspond to the same lesion. Therefore, before training, we need a lesion-level train-validation-test split to reduce data leakage.

That is the correct engineering story.

---

# END OF CURRENT DOCUMENTATION

# NEXT: LESION-LEVEL DATA ANALYSIS AND LEAKAGE-SAFE SPLITTING
