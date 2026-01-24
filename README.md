# Deep Learning: From Classification to Detection

## Overview
This project explores the transition from image classification to object detection using deep learning and transfer learning techniques.  
Starting from pretrained classification backbones, the project progressively adapts them to perform object localization and detection tasks.

The focus is on architectural reasoning, system design, and empirical analysis rather than training models from scratch.

---

## Project Structure
The repository is organized into three main stages, each reflecting a different level of complexity:


Additional directories:
- `data/` – datasets (not tracked by git)
- `videos/` – input/output videos for inference and visualization

---

## Key Concepts
- Deep Learning with convolutional neural networks (CNNs)
- Transfer learning using pretrained classification backbones
- Transition from classification to object detection
- Bounding box regression and detection-specific loss functions
- Training monitoring and analysis using TensorBoard

---

## Workflow
1. Analyze a pretrained classification backbone and its architectural properties
2. Adapt the backbone for single-object detection
3. Extend the detector to support multiple objects and multiple classes
4. Train, evaluate, and visualize detection results on images and videos

---

## Technologies
- Python
- PyTorch
- Torchvision
- TensorBoard
- Jupyter Notebooks

---

## Notes
- Datasets, model checkpoints, and TensorBoard logs are excluded from version control.
- Jupyter notebooks are used for exploration and debugging, while training pipelines are implemented in Python scripts.

---

## License
This project is released under the MIT License.
