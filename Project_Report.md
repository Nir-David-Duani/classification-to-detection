From Image Classification to Object Detection  
Deep Learning Project Report  
  
Course: Introduction to Image Processing and Computer Vision  
Lecturer: Yoni Chechik  
  
Authors:  
- Nir David Duani — ID: 326277704  
  
Date: Feb 27, 2026  

---

## 1. Abstract
This project explores the transition from image classification to object detection using deep learning based convolutional neural networks. Starting from a pretrained classification backbone, the project examines how high-level semantic representations learned on large-scale datasets can be reused and adapted for detection tasks.  
The work is divided into three stages. First, a detailed analysis of a classification backbone architecture is presented, focusing on design motivations, residual learning, and optimization behavior. Next, the backbone is adapted for single-class, single-object detection through transfer learning, where the classification head is replaced with a localization-aware prediction module. Finally, the system is extended to handle multiple object classes and multiple objects per frame under a fixed-capacity constraint.  
Throughout the project, design decisions are motivated theoretically and empirically, with emphasis on training behavior, evaluation metrics, and inference on external videos. The results highlight both the strengths and limitations of classification backbones when repurposed for object detection tasks.

---

## 2. Project Links
- Project repo: (add link)  
- Video links: (add links)

---

## Part 1 — Classification Backbone Analysis (ResNet-18)

### 1. Backbone Selection and ID Calculation
The backbone architecture was selected in accordance with the assignment guidelines.  
The digits of the student ID number were summed digit by digit \((3+2+6+2+7+7+7+0+4)\), resulting in a total sum of 38.  
The final digit of this sum is 1, which corresponds to the selection of the ResNet-18 architecture.

### 2. Inference Using a Pretrained ResNet-18
To demonstrate the capabilities of the selected backbone, inference was performed using a ResNet-18 model pre-trained on ImageNet. The goal of this experiment is not to “prove” the model is correct (it is a known baseline), but to observe how a strong pre-trained classifier behaves on natural images and to connect these observations to architectural design choices.

For the inference demonstration, several images were sampled (preferably from diverse sources: natural scenes, indoor settings, different lighting, multiple objects in the same frame, etc.). Each image was preprocessed using the standard ImageNet normalization and resized as required by the model input. For each image, the model’s top predicted class (and confidence score) was recorded.

Observed strengths. ResNet-18 typically predicts the dominant object category correctly when the object is large, centered, and clearly visible. This behavior suggests that the pretrained network has learned robust high-level semantic features that generalize reasonably well beyond the original training set.

Observed limitations. In cases where the image contains multiple objects, strong background clutter, or objects at small scale, the classifier often predicts a single “dominant” category that may ignore other relevant objects. Additionally, errors often occur between visually similar categories, reflecting the fine-grained nature of the ImageNet label space.

A critical limitation of classification backbones is that they provide no explicit spatial output. The model produces a single category prediction for the entire image, but it does not identify where the object is located. This gap directly motivates the transition to object detection: we want to reuse the learned feature extractor while modifying the network to output localization information (bounding boxes) in later parts of the project.

### 3. Overview of the ResNet-18 Architecture
ResNet-18 is a deep convolutional neural network composed of an initial convolutional “stem”, followed by four sequential stages of residual blocks, and finally a global pooling and classification head. At a high level, the network is designed to gradually reduce spatial resolution while increasing the number of channels, enabling the extraction of richer semantic features as depth increases.

A typical ResNet-18 flow includes:
- An initial convolution + normalization + nonlinearity (stem)
- Four stages of residual blocks, where each stage operates at a specific spatial resolution
- Downsampling between stages (reducing spatial size while increasing channels)
- Global average pooling (converts a spatial feature map into a compact feature vector)
- A final fully connected layer producing class logits

From a systems perspective, ResNet-18 is attractive because it achieves strong accuracy with moderate computational cost. Compared to older architectures like VGG, ResNet-18 tends to be more parameter-efficient while remaining expressive, largely due to residual learning and the ability to scale depth without optimization collapse.

### 4. The Core Idea: Residual Learning
#### 4.1 Motivation: The Degradation Problem
A key insight from the original ResNet work is that making networks deeper is not only a representational question (“can the model represent the solution?”) but also an optimization question (“can gradient-based training actually find it?”). Even when vanishing gradients are mitigated by initialization and normalization, very deep “plain” networks (constructed by stacking layers without skip connections) may still become harder to optimize.

The degradation problem describes the empirical observation that beyond a certain depth, adding layers can increase training error. This is not a typical overfitting phenomenon: the deeper model has more capacity and should be able to fit the training set at least as well as the shallower model. In principle, a deeper model could replicate a shallower solution by setting the extra layers to behave like identity mappings. The fact that training fails to find such solutions reliably suggests that the optimization landscape becomes unfavorable as depth grows.

This sets the stage for residual learning: instead of expecting stacked layers to learn a full transformation from scratch, we structure the network so that “doing nothing” (identity) becomes easy.

#### 4.2 Residual Learning Formulation
Let the desired underlying mapping be denoted as \(H(x)\). Instead of learning this mapping directly, residual networks learn a residual function \(F(x)\) defined as:
\[
F(x) = H(x) - x
\]
The output of a residual block is then given by:
\[
y = x + F(x)
\]
In this formulation, the input is propagated forward through an identity shortcut connection, while the stacked convolutional layers learn a corrective update represented by the residual function.

#### 4.3 Optimization Perspective
In many practical scenarios, the optimal transformation is close to the identity mapping. In such cases, the residual function satisfies \(F(x) \approx 0\), which implies \(H(x) \approx x\). Learning a function that is close to zero is significantly easier for gradient-based optimization methods than learning an identity mapping through multiple nonlinear layers. As a result, residual blocks can effectively avoid modifying the input when no improvement is needed, preventing newly added layers from degrading network performance.

Furthermore, identity shortcut connections provide direct pathways for gradient propagation during backpropagation. Gradients can flow through these shortcuts without repeated multiplication by weights or nonlinearities, improving gradient flow and stabilizing training in deep architectures.

#### 4.4 Residual Blocks in ResNet-18
In ResNet-18, residual learning is implemented using basic residual blocks composed of two convolutional layers followed by an element-wise addition with the block input. When the spatial resolution or number of channels changes between stages, a projection operation, typically implemented using a \(1 \times 1\) convolution, is applied to the shortcut connection to ensure dimensional compatibility.

### 5. Strengths, Limitations, and Motivation for Detection
The analysis of ResNet-18 highlights several strengths. The architecture enables effective training of deep networks, learns transferable hierarchical features, and demonstrates strong performance as a pre-trained classification backbone. Its residual structure promotes stable optimization and feature reuse across layers.

At the same time, ResNet-18 remains a classification model and therefore lacks explicit spatial reasoning. It cannot localize objects, distinguish between multiple instances, or model object geometry. These limitations motivate extending the backbone toward object detection, where the pretrained feature extractor is reused while the network head is redesigned to predict object locations and object classes.

### 6. References (Part 1)
[1] K. He, X. Zhang, S. Ren, J. Sun. “Deep Residual Learning for Image Recognition.” CVPR 2016.  
[2] K. He, X. Zhang, S. Ren, J. Sun. “Identity Mappings in Deep Residual Networks.” ECCV 2016.

---

## Part 2 — Single-Class, Single-Object Detection (Safety Vest)

### 2.1 Objective and High-Level Approach
The goal of Part 2 is to convert a pretrained classifier (ResNet-18) into a detector that predicts **one axis-aligned bounding box per image**, for a dataset that contains **a single object class** and **exactly one object instance per frame**.

Conceptually, the task changes from:
- **Classification**: \(f(\text{image}) \rightarrow \text{class logits}\)
to
- **Regression**: \(f(\text{image}) \rightarrow \text{bounding box parameters}\)

We use **transfer learning**: the pretrained ResNet-18 convolutional trunk is reused as a backbone feature extractor, while the final classification head is replaced by a lightweight regression head that outputs 4 numbers.

---

### 2.2 Dataset Selection and Annotation Format
We used a Roboflow public dataset export (directory: `single_object_detection/data/saftey-vest/`) split into:
- `train/`
- `valid/`
- `test/`

Each split contains images and a COCO annotation file: `_annotations.coco.json`.

**Image resolution**: all images are \(224 \times 224\).  
**COCO bbox format**: \([x, y, w, h]\) in pixel coordinates (top-left origin).  
**Class**: the effective object class used in Part 2 is `Vest` (single-class).  

---

### 2.3 Data Sanity Checks (Stage 1)
Before implementing training, we validated the dataset integrity and the “single-object per image” assumption using the notebook:
- `single_object_detection/experiments.ipynb`

The sanity checks included:
- Verifying dataset split sizes (train/valid/test).
- Ensuring each image has **exactly one** annotation (no missing labels and no multiple objects).
- Ensuring bounding boxes are valid (positive area and inside image bounds).
- Visualizing random samples with drawn bounding boxes to confirm geometric correctness.
- Measuring bbox area statistics to detect annotation outliers (e.g., extremely small/large boxes).

**Observed bbox scale distribution (area fraction = \((w\cdot h)/(W\cdot H)\))**  
Train split summary (247 samples):
- median (p50): 0.0625
- p75: 0.1640
- p90: 0.3634
- p99: 0.9310
- max: 0.9955

Threshold view (fraction of samples):
- \(\le 0.20\): 0.806  
- \(> 0.80\): 0.049  

Interpretation: most vests occupy a small-to-moderate portion of the image, with a small tail of large close-ups. This distribution is plausible for real-world imagery and does not indicate a systematic annotation failure.

---

### 2.4 Bounding Box Representation (Stage 2 — Design Choice)
Although COCO stores bounding boxes as pixel \([x,y,w,h]\), we train the network to output a normalized representation:
\[
(c_x, c_y, w, h) \in [0,1]^4
\]
where:
- \(c_x = (x + w/2)/W\)
- \(c_y = (y + h/2)/H\)
- \(w = w/W\)
- \(h = h/H\)

This choice improves optimization stability because all target values lie in the same numerical range \([0,1]\), reducing scale sensitivity and making it easier to select learning rates and loss weights. For visualization and metrics such as IoU, predictions are converted back to pixel coordinates when needed.

---

### 2.5 Dataset Implementation (Stage 2 — `dataset.py`)
We implemented a dedicated PyTorch `Dataset`:
- `single_object_detection/dataset.py`

Key design points:
- The dataset loads `_annotations.coco.json`, builds an efficient index (image id → annotation), and returns a tuple:
  - `image_tensor`: float tensor of shape \((3,224,224)\) in \([0,1]\)
  - `bbox_tensor`: float tensor of shape \((4,)\) in normalized \((c_x,c_y,w,h)\)
- The dataset performs **early validation** by enforcing exactly one annotation per image (consistent with Part 2 assumptions).
- We explicitly documented that geometric augmentations (flip/resize/crop) require bbox-aware transformations; otherwise labels become inconsistent with images.

We validated this dataset in `experiments.ipynb` by loading samples, checking tensor shapes/ranges, and drawing the reconstructed pixel bbox on the returned image tensor.

---

### 2.6 Model Construction (Stage 3 — `model.py`)
We implemented a simple detector model:
- `single_object_detection/model.py`

**Backbone**: pretrained ResNet-18 (ImageNet).  
**Adaptation**: replace the classification layer `fc(512 → 1000)` with a regression head:
- `Linear(512 → 128) → ReLU → Linear(128 → 4)`

This converts the network output from class logits to bbox parameters. Since the target bbox representation is normalized to \([0,1]\), we apply a `Sigmoid` output activation, constraining predictions to the same range. This typically stabilizes early training behavior.

**Transfer learning strategy**: we support freezing and unfreezing the backbone.
- Start with a warm-up stage where the backbone is frozen and only the head is trained.
- Continue with fine-tuning by unfreezing later layers (e.g., `layer4`) and training with a smaller learning rate.

We performed a model sanity check inside `experiments.ipynb` to confirm:
- Input shape: \((B,3,224,224)\)
- Output shape: \((B,4)\)
- Output range: \([0,1]\) when using sigmoid

---

### 2.7 Training Workflow (Stage 4–5 — `train.py`) and Design Rationale
Training for Part 2 was implemented in:
- `single_object_detection/train.py`

The workflow was intentionally **debug-first**, since detection pipelines can fail silently due to mismatches between the dataset target format, model output format, and visualization / IoU calculations.

#### 2.7.1 Mini-overfit (Stage 4 — pipeline validation)
Before running full experiments, a mini-overfit routine was used to validate the end-to-end training pipeline:
- Train on a tiny subset (a few images) with **no geometric augmentation**
- Expectation: regression loss decreases strongly and IoU rises substantially

This step was critical to catch issues such as:
- wrong bbox representation (e.g., \((c_x,c_y,w,h)\) vs \((x_1,y_1,x_2,y_2)\))
- missing normalization / wrong output activation
- transform mismatches between train and eval

#### 2.7.2 Full training (Stage 5 — transfer learning schedule)
For fair comparisons between model heads, all runs shared the same training “assumptions”:
- **Optimizer**: AdamW with weight decay \(3\cdot 10^{-4}\)
- **Two-phase transfer learning**
  - Phase A: train head only (backbone frozen) for `head_only_epochs=10`
  - Phase B: unfreeze `layer4` and continue training with **two parameter groups**:
    - `layer4`: `lr_layer4`
    - `head`: `lr_head`
- **Scheduler**: `ReduceLROnPlateau` on validation mIoU (mode=max), with `patience=5`, `factor=0.5`
- **Early stopping**: disabled to allow full curve inspection in TensorBoard (`use_early_stopping=False`)

The main evaluation metric reported during training was **mean IoU (mIoU)** on the validation set, computed from the predicted and target boxes.

---

### 2.8 TensorBoard Monitoring
To compare architectures and training behavior, TensorBoard was used directly inside:
- `single_object_detection/experiments.ipynb`

All runs were logged under `single_object_detection/logs/`, enabling direct comparison of:
- train/val loss curves
- train/val mIoU curves
- learning rate evolution per parameter group (head vs `layer4`)

---

### 2.9 Architecture Sweep (5 Variants) and Experimental Findings
The primary goal of the sweep was to identify a head architecture that best preserves localization-relevant information for bbox regression.

All architectures share:
- the same pretrained ResNet-18 backbone
- the same output format: normalized \((c_x,c_y,w,h)\in[0,1]^4\) with sigmoid activation

The following head variants were tested (as implemented in `model.py` via `DetectorConfig.arch`):
- **Architecture 0 — `mlp_simple`**: a minimal baseline head (closest to the “original” simple regressor)
- **Architecture 1 — `mlp` (dropout)**: deeper MLP with dropout for regularization
- **Architecture 2 — `conv_head`**: convolutional head before pooling to preserve spatial structure
- **Architecture 3 — `two_head`**: separate branches for center \((c_x,c_y)\) and size \((w,h)\) with weighted loss
- **Architecture 4 — `mlp` + IoU-aware loss**: combines SmoothL1 with an IoU term

#### 2.9.1 Validation performance (qualitative summary)
The sweep showed a clear advantage for the convolutional head:
- **`conv_head` learned localization more effectively** and reached the highest validation mIoU.
- The deeper MLP variants improved optimization stability but tended to saturate earlier.
- The two-head setup was stable, but its best validation mIoU remained below the conv-head.
- The IoU-aware loss improved geometric alignment in some cases, but did not outperform conv-head overall.

#### 2.9.2 Selected model
Based on the sweep, the final model for Part 2 was:
- **Architecture 2 — Conv-head (`arch="conv_head"`)**

---

### 2.10 Test Set Evaluation (Final Model)
After selecting the best architecture, performance was evaluated on the held-out test split using the checkpoint:
- `single_object_detection/checkpoints/full_arch2_conv_head/best.pt`

The evaluation computed:
- SmoothL1 loss on normalized \((c_x,c_y,w,h)\)
- mIoU on reconstructed boxes

**Test results (102 samples):**
- **TEST SmoothL1 loss**: 0.001343  
- **TEST mIoU**: 0.7040

Practical note: the checkpoint loader explicitly used `torch.load(..., weights_only=False)` to support loading the stored configuration object under newer PyTorch defaults.

---

### 2.11 External Video Inference (Qualitative Evaluation + Debugging)
To validate real-world behavior beyond the dataset splits, the detector was applied to external videos:
- input clips: `single_object_detection/videos/clip1.mp4`, `clip2.mp4`
- outputs: `single_object_detection/videos_out/`

Key implementation details (to keep training and inference consistent):
- The model always receives a **fixed 224×224** input (matching the dataset images).
- Each video frame is resized to 224×224 for inference, and the predicted bbox is **scaled back** to the original (or resized) frame resolution before drawing.
- The preprocessing used the same normalization as training (ImageNet mean/std), and frames were converted to a PIL image before applying the evaluation transform to avoid type mismatches.
- The drawn label was fixed to a single string: **`safety vest`**.

An additional uncertainty-based filter (Monte Carlo dropout) was prototyped to suppress unreliable boxes, but it introduced visible temporal jitter and was ultimately removed in favor of stable rendering for the final video output.

---

### 2.12 Summary of Part 2 Outcome
Part 2 successfully converted a pretrained ResNet-18 classifier into a single-object detector by replacing the classification head with a bbox regression head and validating the system through:
- dataset sanity checks and bbox format normalization
- mini-overfit pipeline validation
- a controlled architecture sweep with TensorBoard comparison
- quantitative test-set evaluation (mIoU ≈ 0.70)
- qualitative external-video evaluation, including debugging of scaling / preprocessing mismatches

The final selected solution was the **conv-head architecture**, which demonstrated the strongest localization performance and produced clean, stable results in the final edited video.

---

## References (Part 2)
- Roboflow COCO export format documentation (dataset export conventions)  
- PyTorch / Torchvision model documentation for ResNet18

