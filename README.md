# Lip Reading with Computer Vision & Machine Learning

## Overview

This project explores a lightweight, educational approach to **visual speech recognition** (lip reading) using computer vision techniques and machine learning models. It is intended as a collaborative, two-month project aimed at learning core concepts in **feature extraction**, **temporal modeling**, and **model training**, while also building a project suitable for showcasing on resumes and portfolios.

## Motivation

As computing systems increasingly interface with humans through visual and multimodal inputs, lip reading has emerged as a valuable skill for accessibility (e.g., aiding the hearing impaired), human-computer interaction, and low-bandwidth speech processing. This project aims to provide a hands-on introduction to this space using modern tools, while maintaining a manageable scope for undergraduate research or engineering students.

## Approach

This project uses a hybrid technique combining:

- **Computer Vision** (via MediaPipe or OpenCV) to extract key features such as **lip contours, landmarks, and angles**.
- **Custom Feature Engineering** to encode lip positions and movement over time.
- **Machine Learning Models** such as:
  - Simple MLPs (Multilayer Perceptrons)
  - Temporal models (e.g., 1D CNNs or TCNs)
  - Possibly recurrent or transformer-based baselines for comparison.

The core pipeline includes:
1. **Video input** (pre-recorded or webcam)
2. **Landmark detection** and lip-specific feature extraction
3. **Temporal feature aggregation**
4. **Model inference** on predicted phoneme/word/class
5. (Optional) **Visualization** of landmarks or prediction output

## Goals

- Build a working pipeline from raw video to classification output.
- Experiment with different feature encodings and model architectures.
- Quantitatively and qualitatively evaluate model performance.
- Learn the strengths and limits of data-driven lip reading.
- Collaborate in an efficient and educational way while documenting our learnings.

## Tools & Libraries

- Python
- OpenCV or MediaPipe
- NumPy / Pandas
- PyTorch or TensorFlow
- Matplotlib / Seaborn for visualization

## Status

🟨 Planning and early implementation phase.

🟩 Core goals:
- [ ] Lip feature extraction prototype
- [ ] Baseline ML model
- [ ] Evaluation metrics setup
- [ ] Visual debug interface

## Credits

Developed by Ryan Vo and Daniel Nguyen as part of an exploratory project in computer vision and machine learning, with a focus on simplicity, modularity, and learning.

---

