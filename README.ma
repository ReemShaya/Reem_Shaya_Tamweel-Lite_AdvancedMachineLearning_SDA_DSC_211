# Tamweel Lite

An end-to-end machine-learning project for predicting financing defaults within 90 days of application, testing the model across different time periods and customers, choosing a decision threshold within a 12% review limit, checking model accuracy and interpretability, and creating a repeatable final policy for new data.

**Developer:** Reem Shaya
**Project type:** Individual capstone project

[![Open the executed notebook in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/almiyead-rgb/SafeSite-PPE-Vision-Meaad-Al-Marri/blob/main/SafeSite_Vision_Capstone_Meaad_Almarri.ipynb)

![SafeSite PPE Vision analytics dashboard](reports/SafeSite_Final_Dashboard_20260812_121457.png)

## Project overview

Tamweel Lite is an end-to-end machine-learning project for predicting synthetic financing defaults within 90 days of application. The complete workflow covers data preparation, baseline model comparison, leakage detection, time- and customer-aware validation, class-imbalance handling, cost-sensitive threshold selection under a 12% review capacity, model interpretation, calibration, ensemble evaluation, regional analysis, and final batch policy generation.

The project compares Logistic Regression, XGBoost, and LightGBM, then evaluates ensemble models to select the final approach. The final policy uses Logistic Regression with a calibrated decision threshold and applies a 12% review limit to new application batches.

All data is fully synthetic and the project is for educational use only.
The custom detector recognizes 11 classes:

`helmet`, `gloves`, `vest`, `boots`, `goggles`, `none`, `Person`, `no_helmet`, `no_goggle`, `no_gloves`, and `no_boots`.

## Key results

### Independent test-set evaluation

The custom checkpoint was evaluated with a real `model.val(split="test")` run on 141 held-out images containing 1,251 labeled instances.

| Metric | Result |
|---|---:|
| Precision | 0.62763 |
| Recall | 0.51427 |
| F1 score | 0.56532 |
| mAP50 | 0.52183 |
| mAP50-95 | 0.25838 |
| Inference time | 16.224 ms/image |

The strongest classes were `helmet`, `vest`, `Person`, `gloves`, and `boots`. The violation classes—especially `no_goggle`, `no_gloves`, `no_helmet`, and `no_boots`—were harder because of class imbalance, small objects, visual ambiguity, and limited positive examples.

### Confidence-threshold strategy

Ten thresholds were evaluated on the entire test set. The best overall F1 occurred at `0.35`. A separate safety threshold of `0.05` was selected by maximizing violation F2, intentionally prioritizing recall over precision for potential safety violations.

![Confidence-threshold analysis](reports/threshold_analysis/SafeSite_Confidence_Threshold_Analysis.png)

### Real construction-video tracking

Part 14 processes 200 consecutive frames from an eight-second real construction-site video using `model.track`, ByteTrack with `persist=True`, and an OpenCV capture-process-write pipeline.

| Metric | Result |
|---|---:|
| Processed frames | 200 |
| Processed duration | 8.00 seconds |
| Unique model-generated track IDs | 52 |
| Detection instances | 1,544 |
| Worker detections | 692 |
| PPE/equipment detections | 852 |
| Violation detections | 0 |
| Average inference | 37.77 ms/frame |
| Effective end-to-end rate | 21.24 FPS |

[Watch or download the real-video tracking result](outputs/real_video/SafeSite_real_construction_tracking.mp4).

![Real construction-video tracking preview](outputs/real_video/SafeSite_real_construction_tracking_preview.jpg)

The source clip is [Pexels video 8965526 by Mikael Blomkvist](https://www.pexels.com/video/construction-workers-walking-at-the-construction-site-8965526/) and is used under the [Pexels license](https://www.pexels.com/license/).

### Additional vision task: instance segmentation

The PPE dataset contains bounding-box labels rather than polygon masks. To satisfy and demonstrate a task beyond detection, the project applies the task-specific `yolo26n-seg.pt` weights as a supplementary instance-segmentation module. It produced 22 masks across 12 test images, including 19 person instances.

![Instance-segmentation evidence](reports/instance_segmentation/SafeSite_instance_segmentation_contact_sheet.png)

### Deployment and export

- A Gradio interface was created and smoke-tested on the custom detector: 17 detections were returned for the verification image, including 15 configured violation-class detections at the safety threshold.
- The custom checkpoint was exported with `model.export(format="onnx")`.
- `onnx.checker` validation passed.
- PyTorch and ONNX produced the same total number of detections—30—across five independent test images.

![PyTorch and ONNX comparison](reports/onnx/SafeSite_ONNX_PyTorch_Comparison.png)

## Technical pipeline

1. Download and validate the Construction-PPE dataset.
2. Inspect the 1,416 images, 11 classes, and 11,521 annotations.
3. Run baseline inference with pretrained YOLO26n.
4. Fine-tune YOLO26n for up to 100 epochs with early stopping; training completed 96 epochs and selected epoch 76.
5. Run independent test evaluation with `model.val`.
6. Analyze per-class false positives, false negatives, localization failures, and missed violations.
7. Evaluate ten confidence thresholds and select operating points for general PPE and safety violations.
8. Run pose estimation and task-specific instance segmentation.
9. Track objects across a real continuous video with ByteTrack and OpenCV.
10. Export and verify ONNX, then serve the custom detector through Gradio.

See [Technical Report](docs/TECHNICAL_REPORT.md) and [Rubric Compliance Matrix](docs/RUBRIC_COMPLIANCE.md) for the detailed evidence map.

## Repository structure

```text
SafeSite-PPE-Vision/
├── SafeSite_Vision_Capstone_Meaad_Almarri.ipynb
├── app.py
├── requirements.txt
├── model/
│   └── SafeSite_YOLO26n_best.pt
├── config/
│   ├── data.yaml
│   ├── environment.json
│   └── requirements.txt
├── docs/
│   ├── RUBRIC_COMPLIANCE.md
│   └── TECHNICAL_REPORT.md
├── outputs/
│   ├── real_video/
│   └── SafeSite_final_safety_tracking_20260812_120438.mp4
├── reports/
│   ├── instance_segmentation/
│   ├── onnx/
│   ├── threshold_analysis/
│   └── results_summary.json
└── submission_manifest.json
```

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/almiyead-rgb/SafeSite-PPE-Vision-Meaad-Al-Marri.git
cd SafeSite-PPE-Vision-Meaad-Al-Marri
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Gradio application

```bash
python app.py
```

Open `http://127.0.0.1:7860`, upload a construction-site image, adjust the thresholds if needed, and select **Analyze Safety**.

### 3. Run the executed notebook

Open `SafeSite_Vision_Capstone_Meaad_Almarri.ipynb` in Google Colab. The notebook already contains captured outputs from the completed run. To rerun it, mount Google Drive or adapt the configured paths, then execute the cells in order.

### 4. Direct Python inference

```python
from ultralytics import YOLO

model = YOLO("model/SafeSite_YOLO26n_best.pt")
results = model.predict(source="path/to/image.jpg", conf=0.35)
```

## Dataset and model

- Dataset: [Ultralytics Construction-PPE](https://docs.ultralytics.com/datasets/detect/construction-ppe/)
- Dataset split: 1,132 train images, 143 validation images, and 141 test images
- Custom checkpoint: `model/SafeSite_YOLO26n_best.pt`
- Input size: 640
- Batch size: 16
- Augmentation: HSV, horizontal flipping, translation, scale, mosaic, and final-epoch mosaic closure
- Training hardware: Tesla T4

The full dataset and generated training run directory are intentionally excluded from GitHub. The configuration and download source are recorded in `config/data.yaml`.

## Interpretation and limitations

- A detection is a model prediction, not a legally verified safety incident.
- The low `0.05` violation threshold improves recall but also increases false alerts; `0.35` is the stronger general-purpose operating point.
- A frame with zero detected violations is not proof that a site is completely safe.
- The 52 track IDs in the real video do not represent 52 unique workers; ID switches can occur when objects overlap, leave the frame, or have low-confidence detections.
- PPE compliance should be validated on representative local worksite data before operational deployment.
- The system must not be used for biometric identification or employee-performance decisions.

## Training-program attribution

This project was completed for the **Computer Vision for Developers with Ultralytics** capstone, delivered by **SDAIA Academy via Learning Space** as a five-day, on-site, 30-hour program. Session: **August 2026**.

Training-program reference: [SDAIA Academy on GitHub](https://github.com/SDAIAAcademy).
