# 🩺 AI-Powered Skin Lesion Classifier

> A BTech CSE (AI/ML) project for building and understanding a lightweight deep-learning system that classifies skin-lesion images into a project-defined **benign vs malignant** classification, with the long-term goal of creating a deployable web/mobile application.

---

## ⚠️ Medical Disclaimer

This project is an **educational/research prototype**.

It is **not a medical device**, does not provide a clinical diagnosis, and must not be used as a replacement for examination or advice from a qualified healthcare professional.

Model predictions and confidence scores must not be interpreted as medical certainty.

The project is intended to demonstrate:

- Computer vision
- CNNs
- Transfer learning
- Medical image classification
- Class imbalance handling
- Model evaluation
- Model optimization
- Web/mobile ML deployment

---

# 📑 Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Problem Statement](#2-problem-statement)
- [3. Objectives](#3-objectives)
- [4. Motivation](#4-motivation)
- [5. Project Scope](#5-project-scope)
- [6. System Architecture](#6-system-architecture)
- [7. Complete ML Pipeline](#7-complete-ml-pipeline)
- [8. Dataset](#8-dataset)
- [9. Understanding Skin Lesions](#9-understanding-skin-lesions)
- [10. HAM10000 Classes](#10-ham10000-classes)
- [11. Binary Classification Strategy](#11-binary-classification-strategy)
- [12. Important Dataset Problems](#12-important-dataset-problems)
- [13. Technology Stack](#13-technology-stack)
- [14. Project Structure](#14-project-structure)
- [15. Learning Roadmap](#15-learning-roadmap)
- [16. Implementation Roadmap](#16-implementation-roadmap)
- [17. Medical-AI Evaluation](#17-medical-ai-evaluation)
- [18. Model Optimization](#18-model-optimization)
- [19. Application Architecture](#19-application-architecture)
- [20. Common Pitfalls](#20-common-pitfalls)
- [21. Documentation Strategy](#21-documentation-strategy)
- [22. Experiment Tracking](#22-experiment-tracking)
- [23. Success Criteria](#23-success-criteria)
- [24. Current Status](#24-current-status)
- [25. Git Workflow](#25-git-workflow)
- [26. Future Improvements](#26-future-improvements)
- [27. References](#27-references)

---

# 1. Project Overview

The project aims to build a lightweight computer-vision system that receives a skin-lesion image and produces a binary classification:

```text
                Skin Lesion Image
                       │
                       ▼
              Image Preprocessing
                       │
                       ▼
                  CNN Model
                       │
                       ▼
                Classification
                       │
             ┌─────────┴─────────┐
             │                   │
          Benign             Malignant
             │                   │
             └─────────┬─────────┘
                       ▼
                Result + Probability
                       │
                       ▼
                 Safety Message
