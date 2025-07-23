# Lane Defect Detection with YOLOv12-nano

This module detects potholes in real-time using a lightweight YOLOv12-nano object detection model. It was trained on a custom dataset annotated with bounding boxes for potholes, captured under various road and lighting conditions.

## 📌 Key Features

- Real-time detection using YOLOv12-nano  
- Lightweight and optimized for embedded deployment  
- Supports training, evaluation, and logging of results  

## 📁 Directory Structure

```
pothole_detection_yolov12/
├── pothole_dataset/        # Custom annotated dataset
├── validations/            # Stores validation outputs (e.g., predictions, metrics)
├── runs/                   # Logs, checkpoints, and training outputs
├── train.py                # Model training script
├── evaluate.py             # Model evaluation script
├── yolo12n.pt              # Pretrained YOLOv12 weights
└── README.md               # This file
```

## 📥 Dataset

The dataset was constructed using:

- [Kaggle - Pothole Detection](https://www.kaggle.com/datasets/andrewmvd/pothole-detection/data)  
- [Roboflow - Pothole Detection](https://public.roboflow.com/object-detection/pothole)

Note: The dataset has been converted into **YOLO format** for training and evaluation. However, **preprocessing steps** (e.g., resizing, augmentation, label formatting) used during this conversion are **not included** in this repository.

## 🚀 How to Use

### 1. 🏋️‍♂️ Train the model

```bash
python train.py
```

This command will start training on the dataset using YOLOv12-nano.

### 2. 📊 Evaluate the model

```bash
python evaluate.py
```

Results will be saved under `validations/`, including visualizations and performance metrics (val, val2, val3 folders).

## 📈 Performance

- Achieved **92.8% mean Average Precision (mAP)** on the test set  
- Inference time: <25 ms per image on NVIDIA T4 GPU  

## Notes

- The dataset is located in `pothole_dataset/`.
- Trained model checkpoints and logs are saved in the `runs/` directory.
- `yolo12n.pt` is the base YOLOv12 nano model used for transfer learning.
