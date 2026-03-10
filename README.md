# Deep Learning: From Classification to Detection

## Project Overview

This project explores how to transition from **image classification** to **object detection** using deep learning and transfer learning.

Starting from a **pretrained ResNet‑18 classifier**, the project evolves in three stages:

1. **Backbone analysis** – understanding ResNet‑18 as a pure classifier.
2. **Single‑object detection (Part 2)** – adapting ResNet‑18 to regress a single bounding box per image for one class (safety vest).
3. **Multi‑class, multi‑object detection (Part 3)** – extending the detector to handle multiple classes and multiple objects per frame under a **fixed‑capacity** constraint.

The focus is on:

- Architectural reasoning (how to modify a classifier into a detector).
- Designing clean target representations and loss functions for bounding boxes and classes.
- Analyzing training behavior and metrics (especially IoU).
- Applying the models to **external videos** and critically examining the results.

---

## Example Results

Below are short GIF snippets demonstrating the detectors in action.

### Single‑Object Detection (Part 2)

> Safety‑vest detector – exactly one object per frame.

![Single‑object detection demo](assets/single_object_demo.gif)

### Multi‑Object Detection (Part 3)

> Helmet / person / vest detector – up to 3 objects per frame using fixed slots.

![Multi‑object detection demo](assets/multi_object_demo.gif)

> Replace the `assets/...` paths above with the actual locations of your GIFs in the repo.

---

## Repository Structure

- `backbone_exploration/`  
  Analysis of the pretrained ResNet‑18 backbone:
  - Why residual connections help optimization.
  - How feature maps change across layers.
  - How the classifier behaves on natural images with multiple objects and clutter.

- `single_object_detection/`  
  Part 2 – **single‑class, single‑object detection**:
  - Dataset: safety‑vest images, exactly **one vest per image**.
  - Target: a single normalized bounding box \((c_x, c_y, w, h)\) in \([0, 1]\).
  - Model:
    - Reuse ResNet‑18 as a feature extractor.
    - Replace the classification head with a small regression head:
      - MLP variants.
      - Conv‑head that operates on the last conv feature map before pooling.
  - Goals:
    - Debug the full detection pipeline on a simpler problem.
    - Compare different heads on top of the same backbone.
    - Understand the effect of architecture on localization quality.

- `multi_object_detection/`  
  Part 3 – **multi‑class, multi‑object detection with fixed capacity**:

  ### Data and Representation

  - Dataset: construction safety dataset with three classes:
    - `helmet`, `person`, `vest`.
  - Constraint: at most **3 objects per image**.
  - Representation: **three fixed slots per image**:
    - Slot 0 → helmet  
    - Slot 1 → person  
    - Slot 2 → vest  
  - For each slot:
    - A normalized box `(cx, cy, w, h)` in `[0, 1]`.
    - A class ID in `{0, 1, 2, 3}` where `3 = background` (empty slot).
  - For each class we keep **at most one instance**: the largest object of that class (or a random instance during training), so multiple helmets/people/vests are not all modeled explicitly.

  This **slot‑based design** keeps the interface simple (fixed tensor sizes) and makes it easy to define losses and metrics, at the cost of:
  - Not modeling multiple instances of the **same** class in one image.
  - Assuming a fixed semantic meaning per slot.

  ### Architectures in `multi_object_detection/model.py`

  All models share a ResNet‑18 backbone and differ in the “head”:

  - **`mlp_shared`**  
    - ResNet‑18 → global average pooling → MLP → shared head predicting 3×(box + class logits).
    - Simple baseline head; works but typically saturates around mIoU ≈ 0.25 on validation.

  - **`conv_shared`** (main model)  
    - Uses the last conv feature map `(B, 512, 7, 7)`:
      - Conv 1×1 → Conv 3×3 → global pooling → linear heads for boxes and logits.
      - All three slots share the same 128‑dim pooled feature.
    - Preserves spatial structure longer and significantly improves localization over `mlp_shared`.
    - With tuned hyperparameters and IoU‑aware loss (see below), this head achieves the best validation and test performance in the project.

  - **`conv_per_slot`**  
    - Same conv tower, but the final linear heads are **per‑slot**:
      - Each slot has its own small linear head for box and logits.
    - Allows specialization per slot (helmet vs person vs vest), but in practice did not surpass the best `conv_shared` variants in this project.

  - **`conv_per_slot_l3`**  
    - Uses features from `layer3` (higher spatial resolution, e.g. `(B, 256, 14, 14)`) instead of `layer4`.
    - Intended to help with **small objects** (e.g. small helmets or vests far from the camera).
    - Shows some improvements over the simple MLP baseline, but still falls short of the top `conv_shared` configuration under the current settings.

  - **`conv_fpn`**  
    - Small feature pyramid network that fuses:
      - C2 (layer2), C3 (layer3), C4 (layer4).
    - Builds multi‑scale feature maps and then pools them into a single vector for prediction.
    - Conceptually similar to modern detectors (RetinaNet, Faster R‑CNN with FPN), but kept minimal.
    - In this project, the simpler `conv_shared` head remained superior.

  - **`grid_shared`**  
    - YOLO‑style grid head on the 7×7 feature map:
      - Predicts per‑cell objectness, box parameters, and class logits for each slot.
      - Aggregates per‑cell predictions into 3 slots via a softmax over objectness.
    - Bridges the gap between classic grid‑based detection and the fixed‑slot interface.

  Additional variants (`conv_shared_deep`, etc.) explore deeper conv towers and different dropouts / learning rates.

---

## Loss Functions and Metrics

### Slot‑Based Base Loss

Given:

- `pred_boxes` ∈ ℝ^(B×3×4) – normalized `(cx, cy, w, h)`.
- `pred_logits` ∈ ℝ^(B×3×4) – class logits.
- `gt_boxes`, `gt_class_ids` with the same slot structure.

The **base loss** is:

- **Bounding box loss**:
  - SmoothL1 between `pred_boxes` and `gt_boxes` **only for slots where `gt_class_ids != background`**.
- **Classification loss**:
  - CrossEntropy over all 3 slots (using class 3 as “background” for empty slots).

Total:

L_base = λ_bbox · L_SmoothL1 + L_class, with λ_bbox = 1.0.

### IoU‑Aware Loss

To better align training with the evaluation metric, we introduce an **IoU‑aware term**:

L = L_base + λ_IoU · (1 − mIoU).

where:

- `mIoU` is computed as in evaluation (see below), over all non‑background slots.
- `lambda_IoU` is a weight (e.g., 0.5) controlling the influence of the IoU term.

Effect:

- Encourages the model not only to predict “close” parameters, but to produce boxes that **actually overlap** well with the ground truth.
- For the best `conv_shared` configuration, adding this term improved:
  - Validation mIoU from ~0.39 → ~0.44.
  - Test mIoU from ~0.20–0.25 → ~0.36.

### Evaluation Metric

- **mIoU (mean Intersection over Union)**:
  - Convert normalized `(cx, cy, w, h)` to `(x1, y1, x2, y2)` in `[0,1]`.
  - Compute IoU per slot for all slots with `gt_class_ids != background`.
  - Average over all such slots in the batch.

This metric is reported per‑epoch on both train and validation sets, and finally on the held‑out test set.

---

## External Video Inference

The script:

- `multi_object_detection/inference.py`

provides reusable functions to:

- Load a model from a checkpoint (e.g. the best `conv_shared` + IoU‑loss model).
- Run detection on each frame of one or more `.mp4` videos in the `videos/` directory.
- Draw bounding boxes and class labels:
  - Color‑coded per slot / class (e.g. different colors for helmet, person, vest).
- Write annotated videos to `videos_out/...`.

This is used both for:

- **Qualitative evaluation**:
  - Are small helmets detected?
  - How often does the model hallucinate vests or people?
  - How stable are detections across consecutive frames?
- **Demonstrations**:
  - Creating short GIFs and demos for the report and README.

---

## Training and Analysis Highlights

Across Parts 2 and 3 the project emphasizes:

- **Mini‑overfit runs**:
  - Train on a tiny subset of images until the model overfits.
  - Verify that the loss decreases and IoU increases as expected (debugging data/model/loss wiring).

- **Full training schedule**:
  - Phase A: train only the detection head while the backbone is frozen.
  - Phase B: unfreeze `layer4` and optionally fine‑tune it with a smaller learning rate than the head.

- **Architecture sweeps**:
  - Single‑object:
    - Compare MLP vs Conv heads.
  - Multi‑object:
    - Compare `mlp_shared`, `conv_shared`, `conv_per_slot`, `conv_per_slot_l3`, `conv_fpn`, `grid_shared`, and deeper conv variants.

- **Effects of loss design**:
  - Base SmoothL1+CE vs IoU‑aware loss.
  - Experiments with Hungarian matching (set prediction) vs fixed slots.

- **Failure cases & limitations**:
  - Fixed‑slot assumption:
    - At most one object per class; multiple helmets or multiple people in the frame are partially represented.
  - Small, crowded objects and occlusions.
  - Sensitivity to hyperparameters (learning rates, dropout, weight decay).

---

## Technologies

- **Core**: Python, PyTorch, Torchvision
- **Experimentation**: Jupyter Notebooks, TensorBoard
- **Visualization**: Matplotlib, OpenCV (for video I/O and drawing detections)

---

## Notes

- Raw datasets, model checkpoints, and TensorBoard logs are not tracked by git.
- Jupyter notebooks are used for exploration, debugging, and documenting experiments.
- Training and evaluation pipelines are implemented in Python modules (`train.py`, `model.py`, `dataset.py`), making it easier to reproduce experiments and run them outside notebooks.

---

## License

This project is released under the MIT License.