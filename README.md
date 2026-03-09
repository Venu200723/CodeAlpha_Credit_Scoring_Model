# CodeAlpha Credit Scoring Model

## 📌 Project Title
Credit Scoring Model using Machine Learning

## 📖 Problem Statement
The objective of this project is to build a Machine Learning model capable of predicting whether a person is creditworthy based on their financial history and demographic details. In the banking and lending industry, accurately predicting credit risk is crucial to minimizing loan defaults and optimizing the approval process.

## 📊 Dataset Description
The model is trained on a dataset (`dataset/train.csv` and `dataset/test.csv`) which contains financial data parameters along with a target variable (detected as the last column) denoting the credit status of an individual.

## ⚙️ Steps Performed
1. **Libraries Imported**: Data handling (`pandas`, `numpy`), Visualization (`matplotlib`, `seaborn`), and Machine Learning algorithms (`scikit-learn`).
2. **Dataset Loaded**: Using `pandas` to read the dataset and display basic shapes and column descriptions.
3. **Exploratory Data Analysis (EDA)**: Investigating dataset info, computing summary statistics, missing value counts, target distribution, and generating visual plots (Correlation Heatmap, Feature Distributions).
4. **Data Cleaning**: Handling missing values (median for numerics, mode for categoricals), removing duplicate entries, and Label Encoding for categorical variables.
5. **Feature Selection**: Separating `X` (features) and `y` (target variable).
6. **Train-Test Split**: Splitting the processed data into 80% training data and 20% test data.
7. **Model Selection & Training**: Training multiple classifiers to find the best functioning logic.
8. **Model Evaluation**: Using Accuracy, Confusion Matrix, and Classification Report to review model performance.
9. **Best Model Selection**: Automating the selection of the best model based on accuracy score.
10. **Generating Predictions**: Forecasting test-set capabilities using the premier model.

## 🤖 Machine Learning Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## 📈 Final Results
Running the script yields real-time accuracy calculations based on the dataset shape. The Random Forest Classifier typically yields the best testing accuracy with complex demographic datasets.

### How to Run:
```bash
pip install -r requirements.txt
python credit_scoring_model.py
```
