# Loan Approval Prediction

A machine learning classification project that predicts whether a loan application is likely to be **approved or rejected** using applicant, financial, credit-history, and property information.

The project includes exploratory data analysis, missing-value handling, categorical encoding, feature scaling, model training, evaluation, comparison, and a sample prediction. The complete implementation is provided as a Jupyter Notebook together with the dataset and project report.

## Project Overview

The dataset contains **614 records and 13 original columns**. `Loan_Status` is the prediction target, with loan approval represented as a binary classification problem.

Important input attributes include:

- Gender
- Marital status
- Dependents
- Education
- Self-employment status
- Applicant income
- Co-applicant income
- Loan amount
- Loan term
- Credit history
- Property area

`Loan_ID` is removed before model training because it is an identifier rather than a predictive feature.

## Machine Learning Workflow

```text
Dataset
   |
   v
Data Inspection & EDA
   |
   v
Missing-Value Handling
   |
   v
Categorical Encoding
   |
   v
Feature Scaling
   |
   v
80/20 Train-Test Split
   |
   +----------------------+
   |                      |
   v                      v
Logistic Regression    Random Forest
   |                      |
   +----------+-----------+
              |
              v
Accuracy, Confusion Matrix,
Precision, Recall & F1-score
              |
              v
Model Comparison & Prediction
```

## Data Preprocessing

The notebook performs the following preprocessing steps:

- Drops `Loan_ID`
- Fills missing categorical values using the mode
- Fills missing numerical values using the median
- Converts `Loan_Status` from `Y/N` to `1/0`
- Converts the `3+` dependents category to a numeric representation
- Applies one-hot encoding to categorical features with `drop_first=True`
- Standardizes features using `StandardScaler`
- Uses an 80/20 train-test split with `random_state=42`

## Exploratory Data Analysis

The notebook includes visual analysis of the dataset, including:

- Loan approval distribution
- Applicant income distribution
- Loan amount distribution
- Numeric correlation heatmap
- Correlation analysis after preprocessing
- Model evaluation visualizations
- Accuracy comparison between trained models

## Models

Two supervised classification models are evaluated:

1. **Logistic Regression**
2. **Random Forest Classifier**

## Results

| Model | Test Accuracy |
| --- | ---: |
| **Logistic Regression** | **78.86%** |
| Random Forest | 77.24% |

Logistic Regression achieved the higher test accuracy in this notebook.

### Logistic Regression

Confusion matrix:

```text
[[18, 25],
 [ 1, 79]]
```

Key class-wise metrics reported by the notebook include approximately:

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| Rejected (0) | 0.95 | 0.42 | 0.58 |
| Approved (1) | 0.76 | 0.99 | 0.86 |

### Random Forest

Confusion matrix:

```text
[[18, 25],
 [ 3, 77]]
```

The notebook also demonstrates a sample loan prediction using the Random Forest model, producing **Loan Approved** for the supplied example.

## Technologies Used

- Python
- Jupyter Notebook / Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Repository Structure

```text
Loan-Approval-Prediction/
├── notebook/
│   └── Loan_Approval_Prediction.ipynb
├── data/
│   └── Loan_Prediction_Dataset.csv
├── docs/
│   └── Loan_Approval_Prediction_Report.pdf
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Running the Project

### Google Colab

1. Open `notebook/Loan_Approval_Prediction.ipynb` in Google Colab.
2. Upload `data/Loan_Prediction_Dataset.csv` when required by the notebook.
3. Run the notebook cells in order.

### Local Jupyter

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `notebook/Loan_Approval_Prediction.ipynb` and run the cells in sequence.

## Notes and Limitations

- The project is an academic machine learning implementation and should not be used as a real lending decision system.
- The dataset is relatively small, so reported test performance should not be interpreted as production-level performance.
- In the submitted notebook, feature scaling is performed before the train-test split. A stricter production workflow would split the data first and fit preprocessing transformations only on the training data, ideally using a scikit-learn `Pipeline`.
- Accuracy alone does not capture the full behavior of the classifier; the confusion matrix, precision, recall, and F1-score are also important, particularly because performance differs between the approved and rejected classes.

## Documentation

The full project report is available at:

`docs/Loan_Approval_Prediction_Report.pdf`

## License

This project is released under the MIT License.
