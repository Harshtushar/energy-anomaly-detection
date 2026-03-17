# Energy Anomaly Detection in Buildings

## Project Overview

This project detects **abnormal energy consumption patterns in buildings** using machine learning-based anomaly detection techniques. By analyzing historical energy meter readings, the system identifies unusual spikes or drops in energy usage that may indicate equipment malfunction, inefficient energy consumption, or abnormal building behavior.

The project uses the **Building Data Genome 2 (BDG2) dataset** and applies multiple anomaly detection algorithms to improve detection reliability.

---

## Objectives

The main objectives of this project are:

- Detect anomalies in building energy consumption
- Identify unusual spikes and drops in energy usage
- Apply unsupervised machine learning models
- Generate visual insights into energy anomalies
- Provide actionable insights for energy efficiency

---

## Dataset

The project uses the **Building Data Genome 2 dataset**, which contains:

- Timestamped energy meter readings
- Multiple building energy consumption records
- Weather data associated with buildings

The dataset contains **millions of time-series observations**, making it suitable for large-scale anomaly detection.

Due to the large size of the Building Data Genome 2 dataset,
only a small sample dataset is included in this repository.

To run the full project, download the dataset from:

https://www.kaggle.com/datasets

---

## Machine Learning Models Used

This project uses **three unsupervised anomaly detection models**:

1. **Isolation Forest**
2. **Local Outlier Factor (LOF)**
3. **Robust Covariance (Elliptic Envelope)**

The final anomaly classification is obtained using **ensemble voting**.

A data point is considered an anomaly if **at least two models classify it as anomalous**.

---

## Feature Engineering

Several features were engineered to improve anomaly detection:

### Temporal Features

- Hour of the day
- Day of the week
- Month
- Weekend indicator

### Statistical Features

- 7-day rolling mean
- 7-day rolling standard deviation
- Deviation from baseline consumption

### Lag Features

- Previous hour energy consumption (`lag1`)
- Previous day energy consumption (`lag24`)

---

## Visualization & Analysis

The project generates several visual insights:

- Feature importance
- Anomaly distribution
- Energy usage with anomaly highlights
- Monthly anomaly trends

Example outputs are stored in the **results folder**.

---

## Project Structure
energy-anomaly-detection
│
├── data/
│ └── BDG2/raw/
│
├── notebooks/
│ └── energy_anomaly_detection.ipynb
│
├── src/
│ ├── preprocessing.py
│ ├── feature_engineering.py
│ ├── anomaly_models.py
│ └── visualization.py
│
├── results/
│ ├── feature_importance.png
│ ├── anomaly_distribution.png
│ ├── energy_anomalies.png
│ └── monthly_anomalies.png
│
├── models/
│ └── isolation_forest_model.pkl
│
├── requirements.txt
├── README.md
└── .gitignore

---

## Installation

### Clone the repository:

```bash
git clone https://github.com/Harshtushar/energy-anomaly-detection.git
cd energy-anomaly-detection
```
### Create a virtual environment:

python -m venv venv

### Activate the environment.

Windows:

venv\Scripts\activate

### Install dependencies:

pip install -r requirements.txt


### Running the Project

Launch Jupyter Notebook:

jupyter notebook

Open:

notebooks/energy_anomaly_detection.ipynb

Run the notebook cells sequentially to execute the full anomaly detection pipeline.

## Results

The anomaly detection system identifies approximately 4–5% of observations as anomalies, which aligns with the expected anomaly rate defined during model configuration.

Visualizations help identify:

periods of abnormal energy usage

seasonal anomaly patterns

buildings with unusual consumption behavior

## Future Improvements

Possible future enhancements include:

Real-time anomaly detection system

Integration with IoT energy meters

Deep learning models for time-series anomaly detection

Interactive dashboards for monitoring anomalies

## Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Jupyter Notebook

## Author
Developed as part of a Machine Learning / Data Science internship project focused on energy analytics and anomaly detection.