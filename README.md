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


'''

## How to Run the Project

### 1. Setup Environment

```bash
# Create and activate virtual environment (if not already done)
python -m venv sleepenv
sleepenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Train the Model (Optional)

If you want to retrain the model:

```bash
python train.py
```

The trained model will be saved as `models/drowsiness_model.h5`

---

### 3. Run the Drowsiness Detector

```bash
python drowsiness_detector.py
```

A GUI window will open with three options:

- Upload Image  
- Upload Video  
- Start Webcam  

---

## Model Details

- Architecture: Simple CNN (3 Conv + MaxPooling layers + Dense layers)  
- Input Size: 64x64 RGB images  
- Dataset: MRL Eyes Dataset (Open vs Closed eyes)  
- Training Accuracy: ~92-95% on test set (depends on training)  
- Framework: TensorFlow 2.15 + Keras  

Age Prediction: Uses DeepFace library for realistic age estimation from detected faces.

---



## How to Add Your Own Output Images (For Submission)

### Recommended Way:

- Create a new folder in your project: `sample_outputs/`  
- Run the detector and test with good images  
- Take screenshots of the output window (including red/green boxes and pop-up if possible)  
- Save the screenshots as:  
  - `drowsy_example.jpg`  
  - `awake_example.jpg`  
  - `multiple_people.jpg` (optional)  

- Update the README paths accordingly  

Tip: Use Snipping Tool or Windows + Shift + S to capture clean screenshots of the result window.

---

## Technologies Used

- Python 3.11  
- TensorFlow 2.15 + Keras  
- OpenCV  
- DeepFace (for age estimation)  
- Tkinter (GUI)  
- Haar Cascade (face detection)  
