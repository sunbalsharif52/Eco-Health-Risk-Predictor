# Eco-Health Risk Predictor

### Machine Learning-Based Environmental Health Impact Classification

**Developed by Sunbal Sharif**

## Project Overview

Eco-Health Risk Predictor is a machine learning project that analyzes environmental measurements and predicts health-impact classes using a Random Forest Classifier.

The project combines exploratory data analysis (EDA), data preprocessing, model evaluation, and an interactive Streamlit dashboard to demonstrate an end-to-end machine learning workflow.

## Project Objectives

- Explore environmental and health-impact data.
- Identify missing values, duplicate records, and data-quality issues.
- Analyze the distribution of health-impact classes.
- Prepare suitable features for machine learning.
- Train a Random Forest classification model.
- Evaluate model performance using multiple classification metrics.
- Visualize classification results using a confusion matrix.
- Examine feature importance.
- Develop an interactive dashboard for generating model predictions.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical computing
- **Matplotlib** — Data visualization
- **Seaborn** — Statistical visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Jupyter Notebook** — Data analysis and model development
- **Streamlit** — Interactive dashboard development

## Dataset

The project uses the Air Quality and Health Impact Dataset obtained from Kaggle.

The dataset includes environmental measurements and health-impact information. Depending on the supplied dataset version, available variables may include:

- **AQI:** Air Quality Index
- **PM2.5 and PM10:** Particulate matter measurements
- **NO2, SO2, and O3:** Atmospheric gas measurements
- **Health-impact variables:** Recorded health-impact scores or classification labels

The actual target column and its class encoding must be verified against the dataset documentation before interpreting predictions.

The dataset file used by this project is:

`air_quality_health_impact_data.csv`

## Project Structure

```text
ECOHEALTH_RISK_PREDICTOR/
│
├── air_quality_health_impact_data.csv
├── app.py
├── EcoHealth_Analysis.ipynb
├── requirements.txt
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `EcoHealth_Analysis.ipynb` | Exploratory data analysis and machine learning workflow |
| `air_quality_health_impact_data.csv` | Dataset used for analysis and predictions |
| `requirements.txt` | Python package dependencies |
| `README.md` | Project documentation |


## Machine Learning Methodology

### 1. Data Loading and Exploration

Load the dataset into a Pandas DataFrame and inspect its dimensions, columns, data types, missing values, duplicate records, and statistical summaries.

### 2. Data Preprocessing

Prepare the dataset for modelling by identifying the target variable, handling missing values, and selecting appropriate predictors.

Identifier columns and potential target-leakage variables should be excluded when they would reveal the target or would not be available at prediction time.

### 3. Train-Test Split

Divide the data into training and testing subsets using an 80:20 split. Stratified splitting helps preserve the distribution of the target classes.

### 4. Model Training

Train a Random Forest Classifier using the training data. Where appropriate, class weighting can help the model account for imbalanced target classes.

### 5. Model Evaluation

Evaluate predictions using suitable classification metrics, including:

- Accuracy
- Balanced accuracy
- Precision
- Recall
- Macro F1-score
- Weighted F1-score
- Confusion matrix
- Classification report

A majority-class baseline can also be used to assess whether the trained model provides an improvement over a simple prediction strategy.

### 6. Feature Importance

Analyze the features that the fitted Random Forest model relies on most heavily when making predictions.

Feature importance describes model behaviour and does not establish causation.

## Streamlit Dashboard

The `app.py` application provides an interactive interface for exploring the dataset and examining model predictions.

Its intended components include:

- Dataset preview
- Model evaluation and confusion matrix
- Interactive environmental input controls
- Predicted health-impact class
- Feature-importance visualization, where implemented

The dashboard's functionality depends on the version of `app.py` currently saved in the repository.

## Installation and Setup

### Prerequisites

- Python
- Git, if cloning the repository
- The project dataset

### 1. Clone the Repository

```bash
git clone https://github.com/sunbalsharif52/Eco-Health-Risk-Predictor.git
cd Eco-Health-Risk-Predictor
```

### 2. Create a Virtual Environment

On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the packages listed in `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard

```powershell
python -m streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Open that URL in your browser to use the dashboard.

### 5. Run the Jupyter Notebook

Install Jupyter Notebook if necessary:

```powershell
python -m pip install notebook
```

Start Jupyter:

```powershell
python -m notebook
```

Open `EcoHealth_Analysis.ipynb` and execute the cells in order.

## Results and Outputs

The analysis can produce:

- Data-quality summaries
- Target-class distribution visualizations
- Model evaluation metrics
- Confusion matrix
- Classification report
- Feature-importance chart
- Test-set prediction results

Actual model performance should be reported only after executing the notebook successfully and verifying its results.

## Limitations

- Model performance depends on dataset quality, class balance, and feature selection.
- If the dataset is synthetic, its results cannot establish performance on real populations.
- Excluding obvious leakage variables does not guarantee that every source of leakage has been eliminated.
- Feature importance does not establish causal relationships.
- Independent validation on representative real-world data is necessary before considering practical public-health applications.

**Disclaimer:** This project is intended for educational, analytical, and portfolio purposes. It is not a medical diagnostic system or a validated public-health risk assessment tool.

## Future Improvements

- Compare Random Forest with Logistic Regression and other suitable classification algorithms.
- Perform cross-validation and hyperparameter tuning.
- Improve minority-class prediction performance.
- Add model explainability and probability calibration.
- Validate the model using independent data.
- Improve dashboard usability and reporting.

## Author

**Sunbal Sharif**

Machine Learning | Data Analysis | Environmental Health Analytics

GitHub: [sunbalsharif52](https://github.com/sunbalsharif52)

---

*This project demonstrates an end-to-end machine learning workflow, from data exploration and preprocessing to model evaluation and interactive visualization.*
