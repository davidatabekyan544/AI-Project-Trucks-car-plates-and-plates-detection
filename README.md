# AI-Based Truck and License Plate Detection System

## Overview
This project detects trucks and recognizes Armenian license plates using YOLOv8 and CRNN.

## Datasets
Truck dataset: universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2
Plate dataset: kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset
Plates recognition: https://roadpolice.am/hy/plate-number-search

## Models Used
- YOLOv8l (Truck Detection)
- YOLOv8n (Plate Detection)
- CRNN (Text Recognition)

## Results
- Truck Detection mAP@0.5: 80%
- Plate Detection mAP@0.5: 84%
- Text Recognition Accuracy: 82%
