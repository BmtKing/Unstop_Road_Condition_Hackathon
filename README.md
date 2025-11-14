# Unstop_Road_Condition_Hackathon
A YOLOv8 project for the Unstop Road Hackathon. Trains a model to detect road damage (potholes, cracks) and includes a script to run inference on images.
# 🚗 AI-Powered Road Damage Detection

This project is a submission for the Unstop Road Hackathon, designed to automatically detect and classify various types of road damage (like potholes, cracks, etc.) from images using a **YOLOv8** object detection model.

The solution is divided into two main parts:
1.  **`road.ipynb`**: A Jupyter Notebook used to **train** a custom YOLOv8 model on the road damage dataset.
2.  **`detect.py`**: A Python script that **uses the trained model** (`best.pt`) to run inference on a folder of test images, display the results, and print a summary report.

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **Ultralytics YOLOv8**: For model training and inference.
* **OpenCV (cv2)**: For image processing, drawing bounding boxes, and displaying results.
* **Pandas**: For creating a final analysis report of all detections.
* **Jupyter Notebook**: For the training environment.

---

## 🚀 How to Run

Follow these steps to set up and run the project.

### 1. Setup

**Clone the repository:**
```bash
git clone [YOUR_REPOSITORY_LINK]
cd [YOUR_PROJECT_FOLDER]
