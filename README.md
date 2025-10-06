Smart Data Explorer & Predictor (Advanced)

Overview

An interactive Streamlit app for exploratory data analysis, dimensionality reduction, clustering, and quick supervised learning (classification/regression) with model comparison and SHAP explainability.

Key features

- Upload any CSV and view dataset overview (head, shape, missing values)
- Data cleaning helpers: fill missing values, encode categorical variables
- Exploratory Data Analysis: summary statistics, scatter plots, Pearson correlation
- Dimensionality reduction with PCA and visualization
- KMeans clustering with adjustable number of clusters
- Auto-detects task (classification vs regression) and trains multiple models
- Model comparison table and downloadable predictions
- Optional SHAP-based feature importance for tree models

Tech stack

- Python, Streamlit (web UI)
- pandas, numpy for data handling
- scikit-learn for ML (train/test split, models, preprocessing)
- plotly for interactive visualizations
- shap for model explainability
- scipy for statistical tests

How to run (PowerShell)

# create and activate virtual env (Windows PowerShell)
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run app.py

Why this is CV-worthy

- Demonstrates end-to-end data product skills: data ingestion, cleaning, EDA, modeling, and explainability
- Uses modern Python data stack and interactive visualization
- Shows knowledge of ML workflows (preprocessing, train/test, metrics)

Suggested CV bullets (pick 2-3 short ones)

- Built an interactive Streamlit app for exploratory data analysis and predictive modeling using pandas, scikit-learn and Plotly; implemented PCA, KMeans clustering, and automatic task detection (classification vs regression).
- Implemented model comparison and explainability using Random Forests and SHAP to surface feature importance and generate downloadable prediction CSVs.
- Added data cleaning utilities (missing value imputation, categorical encoding) and inferential statistics (Pearson correlation) to support rapid data exploration.

Next improvements (recommended before public linking)

- Add a sample dataset and screenshots or a short demo video
- Add input validation, error handling, and unit tests
- Persist best models and add simple CI/CD to run tests and linting
- Sanitize uploads and limit file sizes for production deployments
- Consider packaging as a Docker image or publishing on Streamlit Community Cloud for a live demo

License

Add a LICENSE file if you intend to open-source this project.

Contact

Add your email or GitHub link here if you want people to reach out.

Sample dataset

I've included a small sample dataset `sample_data.csv` (Iris-like) so reviewers can try the app immediately without uploading files. To run the demo:

1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

2. In the app, use the sidebar to upload `sample_data.csv` (or the app will auto-work with it if you add a file upload step), explore the tabs, and try feature selection and modeling.

Reproducibility

`requirements.txt` now contains pinned dependency ranges to improve reproducibility; consider freezing exact versions with `pip freeze > requirements.txt` when you're ready to share the project widely.
