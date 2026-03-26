# Drowsiness Detection System

A simple yet effective Machine Learning project that detects if a person is drowsy (sleeping) using a CNN model. The system can process **images**, **videos**, and **live webcam** feed, highlights drowsy persons with red boxes, predicts their approximate age, and shows a pop-up alert when drowsiness is detected.

---

## Project Overview

This project aims to detect drowsiness in real-time or from uploaded media, which is particularly useful in driver safety and monitoring systems.

**Key Features:**
- Trained a CNN model to classify **Open Eyes (Awake)** vs **Closed Eyes (Drowsy)**
- Detects multiple faces in a single frame
- Draws **red bounding box** around drowsy persons
- Predicts approximate **age** of the person
- Shows a clear **pop-up alert** with number of sleeping people and their ages
- Supports **Image**, **Video**, and **Webcam** input
- Clean and user-friendly GUI using Tkinter

---

## Project Structure

```plaintext
Drowsiness_Detection/
├── data/
│   ├── train/
│   │   ├── open_eyes/
│   │   └── closed_eyes/
│   └── test/
│       ├── open_eyes/
│       └── closed_eyes/
├── models/
│   └── drowsiness_model.h5          # Trained CNN model
├── train.py                         # Script used to train the model
├── drowsiness_detector.py           # Main detection + GUI application
├── requirements.txt                 # List of dependencies
└── README.md
