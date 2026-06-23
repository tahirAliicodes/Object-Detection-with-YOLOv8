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
cap = cv2.VideoCapture(0)

# Create YOLO model using the specified weights file
model = YOLO(weights_path)

# Predefined dictionary for object translation
urdu_translations = {
    "person": "شخص",
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

# ── Colour tokens ────────────────────────────────────────────
BG_DARK     = "#0A0E1A"   # near-black background
BG_PANEL    = "#0F1629"   # slightly lighter panel
BLUE_BRIGHT = "#1E90FF"   # dodger blue — primary accent
BLUE_DIM    = "#0D47A1"   # deep blue — button base
BLUE_HOVER  = "#1565C0"   # button hover
BLUE_STOP   = "#B71C1C"   # red when detection is running
TEXT_WHITE  = "#E8F0FE"   # cool white
TEXT_DIM    = "#7B9CC4"   # muted blue-grey
BORDER_BLUE = "#1A3A6B"   # subtle border

# Initialize prev_frame_time
prev_frame_time = time.time()

# Create a lock to prevent conflicts between threads
lock = threading.Lock()

# Variable to keep track of recording state
recording = False

# Detected object name variable
detected_object = ""

# Global GUI references
root = None
video_label = None
detected_label = None
record_button = None


# Function to update GUI
def update_gui():
    global prev_frame_time, detected_object
    lock.acquire()
    new_frame_time = time.time()
    success, img = cap.read()

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
                if x2 > x1 and y2 > y1:
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if model.names else str(cls)
                    detected_object = f'{urdu_translations.get(class_name, class_name)}  {conf}'

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((940, 680))

    img_tk = ImageTk.PhotoImage(img_pil)
    video_label.img_tk = img_tk
    video_label.configure(image=img_tk)

    detected_label.config(text=detected_object if detected_object else "—")

    lock.release()
    root.after(10, update_gui)


# Function to toggle detection and update button appearance
def toggle_recording_ui():
    global recording
    recording = not recording
    if recording:
        record_button.config(
            text="■  STOP DETECTION",
            bg=BLUE_STOP,
        )
    else:
        record_button.config(
            text="▶  START DETECTION",
            bg=BLUE_DIM,
        )


# Function to start the GUI and initial update
def start_gui():
    global root, video_label, detected_label, record_button

    root = Tk()
    root.title("YOLO Object Detection — Urdu")
    root.geometry("1280x820")
    root.configure(bg=BG_DARK)
    root.resizable(True, True)

    # ── Top header bar ───────────────────────────────────────
    header = Frame(root, bg=BG_PANEL, height=70)
    header.pack(fill=X, side=TOP)
    header.pack_propagate(False)

    # Blue left accent stripe
    Frame(header, bg=BLUE_BRIGHT, width=5).pack(side=LEFT, fill=Y)

    header_inner = Frame(header, bg=BG_PANEL)
    header_inner.pack(side=LEFT, fill=BOTH, expand=True, padx=20)

    Label(
        header_inner,
        text="YOLO  OBJECT DETECTION  —  اردو",
        font=("Consolas", 17, "bold"),
        bg=BG_PANEL, fg=BLUE_BRIGHT,
    ).pack(anchor="w", pady=(12, 0))

    Label(
        header_inner,
        text="Supervised by: Dr Prof. Javed Ahmed Mahar",
        font=("Consolas", 9),
        bg=BG_PANEL, fg=TEXT_DIM,
    ).pack(anchor="w")

    # Live indicator top-right
    status_frame = Frame(header, bg=BG_PANEL)
    status_frame.pack(side=RIGHT, padx=24)
    Label(
        status_frame,
        text="● LIVE",
        font=("Consolas", 10, "bold"),
        bg=BG_PANEL, fg="#00E676",
    ).pack(pady=22)

    # ── Main content area ────────────────────────────────────
    content = Frame(root, bg=BG_DARK)
    content.pack(fill=BOTH, expand=True, padx=14, pady=10)

    # Left: video feed (inside a 1px blue border frame)
    video_outer = Frame(content, bg=BORDER_BLUE, bd=1)
    video_outer.pack(side=LEFT, fill=BOTH, expand=True)

    video_label = Label(video_outer, bg="#000000")
    video_label.pack(fill=BOTH, expand=True)

    # Right: info sidebar
    sidebar = Frame(content, bg=BG_PANEL, width=290)
    sidebar.pack(side=RIGHT, fill=Y, padx=(12, 0))
    sidebar.pack_propagate(False)

    # ── Detected object box ──────────────────────────────────
    Label(
        sidebar,
        text="DETECTED OBJECT",
        font=("Consolas", 9, "bold"),
        bg=BG_PANEL, fg=TEXT_DIM,
    ).pack(anchor="w", padx=16, pady=(20, 4))

    det_border = Frame(sidebar, bg=BORDER_BLUE, bd=1)
    det_border.pack(fill=X, padx=16)

    detected_label = Label(
        det_border,
        text="—",
        font=("Arial", 15),
        bg=BG_PANEL, fg=TEXT_WHITE,
        wraplength=250,
        justify="right",
        anchor="e",
        padx=10, pady=14,
    )
    detected_label.pack(fill=X)

    # ── Model info box ───────────────────────────────────────
    Label(
        sidebar,
        text="MODEL INFO",
        font=("Consolas", 9, "bold"),
        bg=BG_PANEL, fg=TEXT_DIM,
    ).pack(anchor="w", padx=16, pady=(24, 4))

    info_border = Frame(sidebar, bg=BORDER_BLUE, bd=1)
    info_border.pack(fill=X, padx=16)

    for lbl, val in [("Model", "YOLOv8-L"), ("Source", "Webcam"), ("Language", "اردو")]:
        row = Frame(info_border, bg=BG_PANEL)
        row.pack(fill=X, padx=12, pady=5)
        Label(row, text=lbl, font=("Consolas", 9),
              bg=BG_PANEL, fg=TEXT_DIM, anchor="w").pack(side=LEFT)
        Label(row, text=val, font=("Consolas", 9, "bold"),
              bg=BG_PANEL, fg=TEXT_WHITE, anchor="e").pack(side=RIGHT)

    # Thin divider
    Frame(sidebar, bg=BORDER_BLUE, height=1).pack(fill=X, padx=16, pady=(20, 0))

    # ── Start / Stop button ──────────────────────────────────
    record_button = Button(
        sidebar,
        text="▶  START DETECTION",
        font=("Consolas", 11, "bold"),
        bg=BLUE_DIM, fg=TEXT_WHITE,
        activebackground=BLUE_HOVER,
        activeforeground="white",
        relief=FLAT,
        cursor="hand2",
        padx=10, pady=14,
        command=toggle_recording_ui,
    )
    record_button.pack(fill=X, padx=16, pady=20)

    def on_enter(e):
        record_button.config(bg=BLUE_HOVER if not recording else "#C62828")
    def on_leave(e):
        record_button.config(bg=BLUE_DIM if not recording else BLUE_STOP)

    record_button.bind("<Enter>", on_enter)
    record_button.bind("<Leave>", on_leave)

    # ── Bottom footer bar ─────────────────────────────────────
    footer = Frame(root, bg=BG_PANEL, height=38)
    footer.pack(fill=X, side=BOTTOM)
    footer.pack_propagate(False)

    Frame(footer, bg=BLUE_BRIGHT, width=5).pack(side=LEFT, fill=Y)

    Label(
        footer,
        text="SHAH ABDUL LATIF UNIVERSITY  •  KHAIRPUR MIRS",
        font=("Consolas", 9, "bold"),
        bg=BG_PANEL, fg=BLUE_BRIGHT,
    ).pack(side=LEFT, padx=16, pady=10)

    Label(
        footer,
        text="Nadir Ali  •  Tahir Hussain Shar  •  Zohaib Solangi",
        font=("Consolas", 9),
        bg=BG_PANEL, fg=TEXT_DIM,
    ).pack(side=RIGHT, padx=16)

    # Schedule the initial update
    root.after(10, update_gui)
    root.mainloop()


# Create a separate thread to handle the video feed and GUI updates
gui_thread = threading.Thread(target=start_gui)
gui_thread.start()
gui_thread.join()

cap.release()
cv2.destroyAllWindows()