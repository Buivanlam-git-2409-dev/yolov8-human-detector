# CLAUDE.md - YOLOv8 Human Detection Project

## Project
Human detection using YOLOv8. Train on custom dataset, deploy via Streamlit web app.

## Stack
- **Framework:** YOLOv8 (Ultralytics)
- **Web Framework:** Streamlit
- **Computer Vision:** OpenCV
- **Deep Learning:** PyTorch
- **Scientific Computing:** NumPy
- **Dependencies:**
  - ultralytics >= 8.0.0
  - streamlit >= 1.28.0
  - opencv-python >= 4.8.0
  - torch >= 2.0.0
  - numpy >= 1.24.0

## Project Structure

### .claude/
Contains Claude agent skills, subagents, and coding rules.
- **skills/** - Custom workflows
  - `train-yolo/` - Model training automation
  - `run-webapp/` - Streamlit web app launcher
  - `validate-data/` - Dataset format validation
- **agents/** - Subagents for specialized tasks
  - `dataset-expert.md` - YOLO dataset validation and augmentation
- **rules/** - Coding standards and guidelines
  - `code-style.md` - Python/code formatting conventions
  - `security.md` - Security best practices

### dataset/
Training and validation data in YOLO format.
- `data.yaml` - Dataset configuration (paths, classes, splits)
- **train/** - Training dataset
  - `images/` - Training images (class 0: human)
  - `labels/` - YOLO format labels (normalized coordinates)
- **val/** - Validation dataset
  - `images/` - Validation images
  - `labels/` - Validation labels

### models/
Trained model weights and exports.
- `best.pt` - Best trained weights (PyTorch format)
- `best.onnx` - Exported model (ONNX format for deployment)

### app/
Streamlit web application for inference.
- `app.py` - Main Streamlit application
- `utils.py` - Inference utilities and visualization helpers

### scripts/
Training and inference scripts.
- `train.py` - Model training script
- `predict.py` - CLI prediction tool for single images

### logs/ (optional)
Training logs and metrics storage.

## Commands

### Training
```bash
python scripts/train.py --data dataset/data.yaml --epochs 20
```

### Web Application
```bash
streamlit run app/app.py
```

### Single Image Prediction
```bash
python scripts/predict.py --model models/best.pt --source image.jpg
```

### Export to ONNX
```bash
yolo export model=models/best.pt format=onnx
```

## Conventions
- Use pathlib instead of os.path
- Log all errors with try-except blocks
- Add docstrings to all functions
- Follow PEP 8 style guide
- Validate model path existence before loading

## YOU MUST
- Always backup trained .pt files before retraining
- Do not commit dataset to git if > 100MB
- Never hardcode absolute paths (use pathlib)
- Add proper error handling for model loading and file uploads
- Log model training progress and metrics to CSV
- Validate model file exists: `Path("models/best.pt").exists()`
- Use `@st.cache_resource` for Streamlit model loading
- Never use `eval()` on user input

## Environment
Create a `.env` file in the project root with:
```
MODEL_PATH=models/best.pt
CONF_THRESHOLD=0.5
IOU_THRESHOLD=0.45
```

## Skills (Auto-invoked)
The following skills are automatically invoked based on context:

| When | Skill | Reference |
|------|-------|-----------|
| Training model | `train-yolo` | `.claude/skills/train-yolo/SKILL.md` |
| Running webapp | `run-webapp` | `.claude/skills/run-webapp/SKILL.md` |
| Validating dataset | `validate-data` | `.claude/skills/validate-data/SKILL.md` |

## External References
- **Dataset format:** `@dataset/data.yaml`
- **Code conventions:** `@.claude/rules/code-style.md`
- **Security rules:** `@.claude/rules/security.md`
- **Dataset expert:** `@.claude/agents/dataset-expert.md`

## Current State
> Last updated: May 6, 2026

- **Project Status:** Initial setup complete
- **Model Status:** Awaiting training data
- **Web App:** Ready for deployment
- **Next Steps:** Prepare dataset and run training

## Quick Reference

| Task | Command | Skill |
|------|---------|-------|
| Train model | `/train-yolo --epochs 50` | train-yolo |
| Run web app | `/run-webapp` | run-webapp |
| Validate dataset | `/validate-data` | validate-data |
| Dataset advice | `/dataset-expert "augmentation for human detection"` | dataset-expert |

## Key Decisions
- YOLOv8 for state-of-the-art object detection performance
- Streamlit for rapid UI prototyping and deployment
- Local model storage (.pt format) for easy portability
- PyTorch backend for GPU acceleration support