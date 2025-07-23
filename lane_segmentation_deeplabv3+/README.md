# Lane Segmentation with DeepLabV3+

This module performs real-time lane segmentation using DeepLabV3+ with a MobileNetV2 backbone. It is based on TensorFlow 1.x and adapted from the official DeepLab implementation.

## 📌 Dataset Disclaimer

The dataset used for training and evaluation is **proprietary to the CSIP Research Lab UOW** and cannot be publicly shared. Only sample directory structure and processing instructions are provided.

## 📈 Performance

- Achieved **95.99% mean Intersection over Union (mIoU)** on the test set  
- Efficient inference using MobileNetV2 and optimized input pipeline  

## 📁 Directory Structure

```
lane_segmentation_deeplabv3/
├── deeplab/     # Core DeepLabV3+ implementation
├── slim/        # TensorFlow slim dependencies
└── README.md    # This file
```

## 📚 Table of Contents

- [Prerequisites](#prerequisites)  
- [Setup](#setup)  
- [Dataset Preparation](#dataset-preparation)  
- [Data Preprocessing](#data-preprocessing)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Inference](#inference)  
- [References](#references)

## 🛠️ Prerequisites

Install required Python packages:

```bash
pip install numpy Pillow tf_slim matplotlib
```

## ⚙️ Setup

Set `PYTHONPATH` for DeepLab:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

## 🗂️ Dataset Preparation

Organize your dataset as follows:

```
dataset/
├── JPEGImages/
│   ├── image1.jpg
│   ├── image2.jpg
├── SegmentationClass/
│   ├── image1.png
│   ├── image2.png
└── ImageSets/
    └── Segmentation/
        ├── train.txt
        ├── val.txt
        └── trainval.txt
```

## 🧼 Data Preprocessing

### 1. 🔄 Convert segmentation masks:

```bash
python deeplab/datasets/remove_gt_colormap.py \
  --original_gt_folder="dataset/SegmentationClass" \
  --output_dir="dataset/SegmentationClassRaw"
```

### 2. 📦 Create TFRecord:

```bash
python deeplab/datasets/build_voc2012_data.py \
  --image_folder="dataset/JPEGImages" \
  --semantic_segmentation_folder="dataset/SegmentationClassRaw" \
  --list_folder="dataset/ImageSets/Segmentation" \
  --image_format="jpg" \
  --output_dir="dataset/tfrecord"
```

Update `data_generator.py` to reflect your dataset's number of classes and total samples.

## 🏋️ Training

Download a pre-trained model checkpoint from the [DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md).

Start training:

```bash
python deeplab/train.py \
  --logtostderr \
  --training_number_of_steps=30000 \
  --train_split="train" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --train_crop_size="513,513" \
  --train_batch_size=1 \
  --dataset="pascal_voc_seg" \
  --tf_initial_checkpoint="pretrained_model/model.ckpt" \
  --train_logdir="training_log" \
  --dataset_dir="dataset/tfrecord"
```

## 📊 Evaluation

```bash
python deeplab/eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --eval_crop_size="513,513" \
  --dataset="pascal_voc_seg" \
  --checkpoint_dir="training_log" \
  --eval_logdir="eval_log" \
  --dataset_dir="dataset/tfrecord"
```

## 🖼️ Inference

Visualize predicted segmentation masks:

```bash
python deeplab/vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --vis_crop_size="513,513" \
  --dataset="pascal_voc_seg" \
  --checkpoint_dir="training_log" \
  --vis_logdir="vis_log" \
  --dataset_dir="dataset/tfrecord"
```

## 📚 References

- [DeepLab v3+ Custom Training Guide](https://rockyshikoku.medium.com/train-deeplab-v3-with-your-own-dataset-13f2af958a75)  
- [TensorFlow Models Repository](https://github.com/tensorflow/models)

## Notes

- The training logs and visualization outputs are located under:

```
lane_segmentation_deeplabv3/deeplab/datasets/custom/exp/train_on_trainval_set/
```

- For privacy reasons, only a few example outputs are retained in this repository.
