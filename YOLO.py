import math
import os
import time
import cv2
from ultralytics import YOLO
from tkinter import *
from PIL import Image, ImageTk
import threading
import cvzone


# Specify the full path to the weights file
weights_path = os.path.abspath('yolov8l.pt')

# Check and create the directory if it doesn't exist
weights_dir = os.path.dirname(weights_path)
os.makedirs(weights_dir, exist_ok=True)

# Check if the weights file exists before creating the YOLO model
if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"The weights file '{weights_path}' does not exist.")

# Open the default camera (index 0)
cap = cv2.VideoCapture(1)  # 0 represents the default webcam

# Create YOLO model using the specified weights file
model = YOLO(weights_path)

# Predefined dictionary for object translation

urdu_translations={
    "person":"شخص",
    "backpack": "بیک پیک",
    "umbrella": "چھاؤنی",
    "handbag": "ہینڈ بیگ",
    "tie": "ٹائی",
    "suitcase": "سیسکیس",
    "frisbee": "فرسیبی",
    "skis": "اسکی",
    "snowboard": "برف بورڈ",
    "sports ball": "کھیل کا گیند",
    "kite": "پتنگ",
    "baseball bat": "بیسبال بیٹ",
    "baseball glove": "بیسبال گلو",
    "skateboard": "اسکیٹ بورڈ",
    "surfboard": "سرف بورڈ",
    "tennis racket": "ٹینس ریکٹ",
    "bottle": "بوتل",
    "wine glass": "شراب کا گلاس",
    "cup": "کپ",
    "fork": "فورک",
    "knife": "چاقو",
    "spoon": "چمچ",
    "bowl": "کھال",
    "banana": "کیلا",
    "apple": "سیب",
    "sandwich": "سینڈوچ",
    "orange": "نارنجی",
    "broccoli": "بروکلی",
    "carrot": "گاجر",
    "hot dog": "ہاٹ ڈاگ",
    "pizza": "پزّا",
    "donut": "ڈونٹ",
    "cake": "کیک",
    "chair": "کرسی",
    "couch": "کوچ",
    "potted plant": "گملا",
    "bed": "بستر",
    "dining table": "خوراک کا میز",
    "toilet": "ٹوائلٹ",
    "tv": "ٹیلی ویژن",
    "laptop": "لیپ ٹاپ",
    "mouse": "ماؤس",
    "remote": "ریموٹ",
    "keyboard": "کی بورڈ",
    "cell phone": "سیل فون",
    "microwave": "مائیکرو ویو",
    "oven": "اوون",
    "toaster": "ٹوسٹر",
    "sink": "سینک",
    "refrigerator": "ریفریجریٹر",
    "book": "کتاب",
    "clock": "گھڑی",
    "vase": "گلدان",
    "scissors": "قینچی",
    "teddy bear": "ٹیڈی بیئر",
    "hair drier": "بالوں کی ڈرائر",
    "toothbrush": "ٹوتھ برش",
}

# Initialize prev_frame_time
prev_frame_time = time.time()

# Create a lock to prevent conflicts between threads
lock = threading.Lock()

# Variable to keep track of recording state
recording = False

# Detected object name variable
detected_object = ""

# Function to update GUI
def update_gui():
    global prev_frame_time, detected_object
    lock.acquire()
    new_frame_time = time.time()
    success, img = cap.read()

    # Check if the frame is None
    if img is None:
        lock.release()
        return

    if recording:
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Check if bounding box coordinates are valid
                if x2 > x1 and y2 > y1:
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if model.names else str(cls)
                    detected_object = f'{urdu_translations.get(class_name, class_name)}{conf} '



    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    # Convert the image to RGB format for tkinter
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((800, 600))

    # Create a new PhotoImage object and update the label
    img_tk = ImageTk.PhotoImage(img_pil)
    video_label.img_tk = img_tk  # Keep a reference to prevent garbage collection
    video_label.configure(image=img_tk)

    # Update the detected object label
    detected_label.config(text=detected_object)

    lock.release()

    # Schedule the function to be called after 10 milliseconds
    root.after(10, update_gui)

# Function to start/stop recording
def toggle_recording():
    global recording
    recording = not recording

# Function to start the GUI and initial update
def start_gui():
    global root, video_label, detected_label

    root = Tk()
    root.title("OBJECT DETECTION USING YOLO IN URDU LANGUAGE")
    root.geometry("1900x1000")

    # Title and subtitle
    title_label = Label(root, text="OBJECT DETECTION USING YOLO IN URDU", font=("Arial", 18, "bold"))
    title_label.pack(pady=10)

    subtitle1_label = Label(root, text="Supervised by: Dr Prof. Javed Ahmed Mahar", font=("Arial", 12))
    subtitle1_label.pack()

    # Video feed label
    video_label = Label(root, width=1800, height=500)
    video_label.pack(pady=10)

    # Detected object label box
    detected_label = Label(root, text="", font=("Arial", 12), bg="white", width=50, height=2, anchor="w")
    detected_label.pack(pady=5)

    subtitle2_label = Label(root, text="Developed by: Nadir Ali, Tahir Hussain Shar, Zohaib Solangi", font=("Arial", 12))
    subtitle2_label.pack()

    # Start/Stop recording buttons
    record_button = Button(root, text="Start Detection", command=toggle_recording, width=20, height=2)
    record_button.pack(pady=5)

    # Footer label with larger font and red color
    footer_label = Label(root, text="SHAH ABDUL LATIF UNIVERSITY KHAIRPUR MIRS", font=("Arial", 16), fg="red")
    footer_label.pack(pady=10)

    # Schedule the initial update
    root.after(10, update_gui)

    # Start the GUI main loop
    root.mainloop()

# Create a separate thread to handle the video feed and GUI updates
gui_thread = threading.Thread(target=start_gui)
gui_thread.start()

# Join the threads to wait for them to finish
gui_thread.join()

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:




