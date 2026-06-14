# 🔍 Object Detection Using YOLOv8 in Urdu Language

> A real-time object detection system that identifies objects through a webcam and displays their names in **Urdu language (اردو)** using a graphical interface.

---

## 🏫 Project Info

| | |
|---|---|
| **University** | Shah Abdul Latif University, Khairpur Mirs |
| **Supervisor** | Dr. Prof. Javed Ahmed Mahar |
| **Developers** |  Tahir Hussain Shar,Nadir Ali, Zohaib Solangi |

---

## 📸 Features

- ✅ Real-time object detection using **YOLOv8**
- ✅ Object names translated into **Urdu language**
- ✅ Live webcam feed with bounding boxes
- ✅ Clean **Tkinter GUI** interface
- ✅ FPS (Frames Per Second) display
- ✅ Start/Stop detection button

---

## 🛠️ Technologies Used

- [Python 3.x](https://www.python.org/)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)
- [Pillow (PIL)](https://pillow.readthedocs.io/)
- [CVZone](https://github.com/cvzone/cvzone)

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/tahirAliicodes/Object-Detection-with-YOLOv8.git
cd Object-Detection-with-YOLOv8
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download YOLOv8 weights
Download the `yolov8l.pt` model from [Ultralytics](https://github.com/ultralytics/assets/releases) and place it in the project root folder.

```bash
# Or download automatically using Python
from ultralytics import YOLO
model = YOLO('yolov8l.pt')  # Downloads automatically on first run
```

---

## ▶️ How to Run

```bash
python YOLO.py
```

- Allow webcam access when prompted
- Click **"Start Detection"** to begin detecting objects
- Detected object names will appear in **Urdu** below the video feed

---

## 📋 Requirements

See `requirements.txt` for full list. Main dependencies:

```
ultralytics
opencv-python
Pillow
cvzone
```

---

## 🌐 Supported Objects (Sample)

| English | اردو |
|---------|------|
| Person | شخص |
| Chair | کرسی |
| Laptop | لیپ ٹاپ |
| Phone | سیل فون |
| Book | کتاب |
| Banana | کیلا |
| Car | گاڑی |

> The system supports **80+ object categories** from the COCO dataset.

---

## ⚠️ Notes

- The `yolov8l.pt` weights file is **not included** in this repository due to its large size (~87MB). Please download it separately (see Installation step 3).
- Make sure your webcam is connected and accessible.
- If you have multiple cameras, change `cv2.VideoCapture(1)` to `cv2.VideoCapture(0)` in `YOLO.py`.

---

## 📄 License

This project was developed for academic purposes at Shah Abdul Latif University, Khairpur Mirs.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.
