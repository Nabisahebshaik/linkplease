# Execution Log

## 2026-08-28

- Started implementation of the SIH 26038 diabetic-retinopathy screening pipeline.
- Inspected the repository and tagged assets. The current repository contains an unrelated LinkPlease FastAPI application; the tagged DR assets are located outside the repository under `C:\Users\shaik\OneDrive\Desktop\DR_Trails`.
- The requested `archive__4_.zip` was not found in the repository root during initial inspection. The pipeline therefore uses configurable dataset and CSV paths and reports missing data explicitly.
- Streamlit reference discovery could not run because the available `python3` interpreter is not installed. The Streamlit app is implemented using stable, documented APIs and keeps model loading lazy so the module can be inspected without TensorFlow installed.
- Implementing preprocessing, VGG16 model construction, training/evaluation, Grad-CAM, and Streamlit inference surfaces with deterministic seeds and explicit error reporting.
- Added root-level `src/` modules, `app.py`, and `verify_inference.py`. The model is loaded lazily by Streamlit and raises a visible error when `vgg16_model.h5` has not yet been trained.
- Added deterministic folder-label collection, optional `train.csv` cross-validation, replacement oversampling, stratified splitting, VGG16 preprocessing, callbacks, classification metrics, confusion-matrix output, and Grad-CAM overlays.
- The checkpoint is saved as a complete Keras model at `vgg16_model.h5`, allowing both the Streamlit app and verification script to load it without reconstructing architecture.
- Extended the pipeline with quality gating/enhancement, explainable vessel/lesion candidate masks, optic-disc localization, printable HTML reporting, and a transparent edge/cloud capacity estimate. The classical masks are screening aids, not validated clinical segmentation models.
- Updated command-line verification for the structured quality-aware inference result; syntax and whitespace checks passed after integration.
- Located the supplied DR1 dataset at `C:\Users\shaik\Downloads\DR1`: 3,662 labeled images and `train.csv`, with class counts No_DR=1,805, Mild=370, Moderate=999, Severe=193, Proliferate_DR=295. Training now accepts `--dataset-dir`/`--labels-csv` or `DR_DATASET_DIR`/`DR_LABELS_CSV` without copying the external dataset into the repository.
- Full training produced `vgg16_model.h5` and reached 91.38% validation accuracy by epoch 5. Initial inference verification found a Keras 3 nested-model Grad-CAM graph error; updated Grad-CAM now traces the nested VGG16 input explicitly.
- A follow-up verification error showed NumPy input was passed to `GradientTape.watch`; converted Grad-CAM inputs to tensors and removed the unnecessary explicit watch.
- Final Grad-CAM fallback applies the deserialized VGG16 pooling and dense head directly to convolution features, avoiding Keras 3 disconnected nested-graph gradients.
- Added `dr_cli.py` and README commands for a VS Code terminal-only workflow; Streamlit is no longer required for local development or verification.
- Added `evaluate_model.py` to build a CSV/PNG confusion matrix and JSON classification report without retraining.
- User supplied `C:\Users\shaik\Downloads\archive (5).zip` as the authoritative dataset source. Archive inspection found 3,663 entries: 3,662 colored fundus images and the label CSV.
- Extracted the archive into local `dataset/colored_images/` and `train.csv`. Cross-validation passed with 3,662 images and zero mismatches; generated dataset/model/metrics artifacts are ignored by Git to avoid accidentally committing large binaries.

- Training completed: epochs=15, test_loss=0.1932, test_accuracy=0.9483. Metrics saved under `metrics/`.
