# Vision_Track

**Vision_Track** is an AI-based navigation system designed to assist visually impaired individuals by detecting obstacles, segmenting lanes, and estimating depth. The system integrates DeepLabV3+ for lane segmentation, a pretrained YOLOv12-nano model for general object detection, and a YOLOv12-nano model trained on a custom dataset for pothole detection. Depth estimation is achieved using DepthAnythingV2 and LiDAR. Feedback is provided through voice and haptic responses.

This repository provides complete implementation of the **lane segmentation** and **pothole detection** modules.

## 🎯 Objectives

- Enable safe and efficient navigation in real-time  
- Detect lane boundaries and road hazards using deep learning  

## 📈 Model Performance

- **Lane Segmentation** (DeepLabV3+): 95.99% mean Intersection over Union (mIoU)  
- **Pothole Detection** (YOLOv12): 92.8% mean Average Precision (mAP)  

## 🗂️ Repository Structure

```
Vision_Track/
├── lane_segmentation_DeepLabV3+/   # Lane segmentation using DeepLabV3+
├── pothole_detection_yolov12/     # Pothole detection using YOLOv12-nano
├── README.md                      # Top-level documentation
├── .gitignore                     # Git ignore rules
```

## 🚀 Getting Started

### 🧰 Requirements

- Python 3.7+  
- PyTorch (for YOLOv12)  
- TensorFlow 1.x (for DeepLabV3+)  
- OpenCV, NumPy, and other standard packages  

To clone this repository:

```bash
git clone https://github.com/your-username/Vision_Track.git
cd Vision_Track
```

## 🕳️ Pothole Detection Module

This module uses a lightweight YOLOv12-nano model to detect potholes from video frames in real time.

📄 See [`pothole_detection_yolov12/README.md`](pothole_detection_yolov12/README.md) for full details.

## 🛣️ Lane Segmentation Module

Based on DeepLabV3+ with a MobileNetV2 backbone, this module segments lane markings to assist path tracking.

📄 See [`lane_segmentation_DeepLabV3+/README.md`](lane_segmentation_DeepLabV3+/README.md) for more information. 

## 📝 License

This code is for academic and research purposes only. Datasets may be proprietary and are not publicly released.

## 🙋 Acknowledgements

- Centre for Signal and Information Processing Research Lab at the University of Wollongong  
- TensorFlow and PyTorch development teams  
- DeepLab and YOLOv12 open-source communities  
