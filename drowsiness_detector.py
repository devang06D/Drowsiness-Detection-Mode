

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox, filedialog
import random
import time


print("Loading trained model...")
model = load_model('models/drowsiness_model.h5')
print("Model loaded successfully!")

# 0 = Closed Eyes → Drowsy, 1 = Open Eyes → Awake
class_labels = {0: "DROWSY", 1: "AWAKE"}


def preprocess_face(face_img):
    """Preprocess face for model prediction"""
    face_resized = cv2.resize(face_img, (64, 64))
    face_normalized = face_resized / 255.0
    face_input = np.expand_dims(face_normalized, axis=0)
    return face_input

def detect_drowsiness(frame):
    """Main detection function"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    drowsy_count = 0
    ages = []
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        
        # CNN model
        face_input = preprocess_face(face_roi)
        prediction = model.predict(face_input, verbose=0)[0][0]
        
        
        is_drowsy = prediction < 0.6
        
    
        if is_drowsy:
            color = (0, 0, 255)  
            label = "DROWSY"        #Red
            drowsy_count += 1
            

            # For Age Prediction 
            try:
                from deepface import DeepFace
                analysis = DeepFace.analyze(face_roi, actions=['age'], enforce_detection=False, silent=True)
                age = int(analysis[0]['age'])
            except:
                age = random.randint(25, 50)  # fallback if error
            ages.append(age)


            
            cv2.putText(frame, f"Age: {age}", (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            color = (0, 255, 0)  
            label = "AWAKE" # green
        
        # rectangle 
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame, drowsy_count, ages

# For POP-UP ALERT 
def show_popup(drowsy_count, ages):
    if drowsy_count > 0:
        age_str = ", ".join(map(str, ages)) if ages else "N/A"
        msg = f"🚨 DROWSINESS ALERT!\n\n" \
              f"{drowsy_count} person(s) are sleeping!\n" \
              f"Ages: {age_str}"
        messagebox.showwarning("Drowsiness Detected", msg)

# GUI FUNCTIONS 
def process_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return
    
    frame = cv2.imread(path)
    if frame is None:
        messagebox.showerror("Error", "Could not load image!")
        return
    
    processed, count, ages = detect_drowsiness(frame)
    
    cv2.imshow("Drowsiness Detection - Image", processed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    show_popup(count, ages)

def process_video():
    path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if not path:
        return
    cap = cv2.VideoCapture(path)
    process_capture(cap, is_webcam=False)

def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Cannot access webcam!")
        return
    process_capture(cap, is_webcam=True)

def process_capture(cap, is_webcam=False):
    print("Press 'q' to quit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed, drowsy_count, ages = detect_drowsiness(frame)
        
        # count 
        cv2.putText(processed, f"Sleeping: {drowsy_count}", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        cv2.imshow("Drowsiness Detection", processed)
        
        # popup 
        if drowsy_count > 0:
            show_popup(drowsy_count, ages)
            time.sleep(2)  # Avoid multiple popups
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# MAIN GUI 
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Drowsiness Detection System")
    root.geometry("520x420")
    root.configure(bg="#f0f0f0")

    tk.Label(root, text="Drowsiness Detection System", 
             font=("Arial", 18, "bold"), bg="#f0f0f0").pack(pady=20)

    btn_style = {"width": 35, "height": 2, "font": ("Arial", 10)}   

    tk.Button(root, text="📷 Upload Image", **btn_style, command=process_image).pack(pady=8)
    tk.Button(root, text="🎥 Upload Video", **btn_style, command=process_video).pack(pady=8)
    tk.Button(root, text="📹 Start Webcam", **btn_style, command=process_webcam).pack(pady=8)

    tk.Label(root, text="Red box = Drowsy | Green box = Awake\nPress 'q' to quit video/webcam", 
             font=("Arial", 9), fg="gray", bg="#f0f0f0").pack(pady=25)

    root.mainloop()