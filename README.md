# Digit Model Lab - Machine Learning Assignment 2

An end-to-end multiclass classification project that trains multiple models on handwritten-digit data, compares the required metrics, and exposes the results in an interactive Streamlit application.

## a. Problem statement

The goal is to predict which digit (0-9) is represented by 64 numeric pixel features from an 8 x 8 grayscale image. The project implements all five classifiers explicitly listed in the assignment and adds an SVM because the PDF also says that six models are required. Every model is trained on the same stratified split and evaluated using Accuracy, multiclass AUC, weighted Precision, weighted Recall, weighted F1, and Matthews Correlation Coefficient (MCC).

## b. Dataset description

This project uses scikit-learn's bundled `load_digits` dataset, a copy of the test portion of the [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).

| Property | Value |
|---|---:|
| Problem type | Multiclass classification |
| Local dataset instances | 1,797 |
| Features | 64 integer pixel-intensity features (0-16) |
| Classes | 10 digits (0-9) |
| Image representation | 8 x 8 pixels, flattened to 64 columns |
| Train/test split | 80% / 20%, stratified |
| Random seed | 42 |

The dataset exceeds the assignment minimums of 500 instances and 12 features. It is loaded locally from scikit-learn, so training is reproducible without a network download.

## c. GitHub Repository Link

**[GitHub Repository](https://github.com/raseeler2024-code/ml-assgnmnt-2)**

## Live Streamlit App Link

**[Live Streamlit App](https://ml-assgnmnt-2.streamlit.app)**

## d. Models used and comparison

All metrics are calculated on the same untouched 20% holdout set. For this 10-class problem, Precision, Recall, and F1 use weighted averaging. AUC uses weighted one-vs-rest (OvR) probabilities.

<!-- MODEL_RESULTS_START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Support Vector Machine (Additional) | 0.9806 | 0.9937 | 0.9810 | 0.9806 | 0.9805 | 0.9784 |
| Logistic Regression | 0.9722 | 0.9991 | 0.9724 | 0.9722 | 0.9722 | 0.9692 |
| k-Nearest Neighbors | 0.9667 | 0.9950 | 0.9675 | 0.9667 | 0.9664 | 0.9631 |
| Random Forest (Ensemble) | 0.9667 | 0.9993 | 0.9673 | 0.9667 | 0.9664 | 0.9631 |
| Decision Tree | 0.8194 | 0.9105 | 0.8209 | 0.8194 | 0.8176 | 0.7999 |
| Gaussian Naive Bayes | 0.8111 | 0.9707 | 0.8480 | 0.8111 | 0.8151 | 0.7940 |
<!-- MODEL_RESULTS_END -->

### Model observations

<!-- MODEL_OBSERVATIONS_START -->
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline: 97.22% accuracy and a high 0.9991 AUC show that standardized pixel patterns separate the digit classes well. |
| Decision Tree | The single tree is easy to explain but has the weakest ranking quality (0.9105 AUC) and clear generalization loss from axis-aligned splits. |
| k-Nearest Neighbors | Performs very well (96.67% accuracy) because visually similar digits remain close after feature scaling; prediction cost grows with the training set. |
| Gaussian Naive Bayes | Its pixel-independence and Gaussian assumptions are too restrictive. Recall is 81.11%, although its 0.9707 AUC shows useful probability ranking. |
| Random Forest (Ensemble) | Matches kNN accuracy while producing a higher 0.9993 AUC. Ensembling greatly improves on the single decision tree and is robust without scaling. |
| Support Vector Machine (Additional) | **Overall winner:** the nonlinear RBF boundary gives the best Accuracy (98.06%), F1 (0.9805), and MCC (0.9784). |
<!-- MODEL_OBSERVATIONS_END -->

## Streamlit features

- CSV upload with the bundled `test_data.csv` as the default
- Model-selection dropdown for all six trained classifiers
- Side-by-side comparison of all six required evaluation metrics
- Confusion matrix and per-class classification report
- Row-level predictions, confidence scores, and downloadable results
- Validation for missing, non-numeric, or invalid input columns
- Customized responsive interface suitable for Streamlit Community Cloud

## Repository structure

```text
ml_assgmnt_2/
|-- app.py                       # Streamlit interface
|-- ml_utils.py                  # Shared data/model/evaluation functions
|-- train_models.py              # Reproducible training pipeline
|-- requirements.txt             # Deployment dependencies
|-- test_data.csv                # Stratified holdout data used by the app
|-- model/                        # Six saved model files and metadata
|-- reports/                      # Metrics and row-level test predictions
|-- tests/test_project.py         # Data/model integrity tests
|-- .streamlit/config.toml        # App theme and server settings
`-- README.md
```

## Run locally

Python 3.11 or 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python train_models.py               # Rebuild models and test outputs
python -m unittest discover -s tests -v
streamlit run app.py
```

## CSV format

The app expects exactly 64 numeric feature columns named `pixel_0_0` through `pixel_7_7`. Include an integer `target` column (0-9) to calculate metrics and the confusion matrix. If `target` is omitted, the app operates in prediction-only mode.

## Deployment on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud) with GitHub.
3. Create a new app and select the repository and branch.
4. Set the entry point to `app.py` and deploy.
5. Open the public URL, upload `test_data.csv`, and confirm all model results render.
6. Confirm the GitHub and live-app links near the top of this README open correctly.

## Assignment submission checklist

- [ ] Run the project on the BITS Virtual Lab and capture one screenshot.
- [ ] Push the complete project and meaningful commit history to GitHub.
- [ ] Deploy `app.py` to Streamlit Community Cloud.
- [x] Add the GitHub repository and live Streamlit app links to this README.
- [ ] Create a single submission PDF in the mandated order: GitHub link, live app link, BITS Lab screenshot, then this README content.
- [ ] Open every link in the final PDF before submitting it.
