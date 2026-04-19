# Heart Disease Prediction — ML Classification

End-to-end machine learning project predicting heart disease presence 
from clinical measurements using the UCI Heart Disease dataset.

## Problem Statement
Binary classification: predict whether a patient has heart disease (1) 
or not (0) based on 13 clinical features including age, cholesterol, 
chest pain type, and ST depression.

## Dataset
UCI Heart Disease Dataset (Cleveland subset) via Kaggle  
920 rows × 15 features | Target: `num` (binarized to 0/1)  
https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data

## Results
| Model               | CV Accuracy |
|---------------------|-------------|
| Logistic Regression | ~0.82       |
| Decision Tree       | ~0.73       |
| Random Forest       | ~0.82       |

Best model: **Random Forest** (less std, tuned with GridSearchCV)

## Project Structure
heart_disease_prediction.ipynb   ← full notebook
heart_disease_uci.csv            ← dataset
README.md

## How to Run
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook heart_disease_prediction.ipynb

## Key Techniques
- EDA: distributions, histograms, bargraphs
- Preprocessing: ColumnTransformer pipeline (imputation + scaling + encoding)
- Models: Logistic Regression, Decision Tree, Random Forest
- Evaluation: 5-fold cross-validation, classification report, Accuracy Score, Confusion Matrix Dispaly
- Tuning: GridSearchCV over n_estimators, max_depth, min_samples_split
