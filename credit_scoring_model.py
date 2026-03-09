import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def main():
    print("="*50)
    print("CodeAlpha Credit Scoring Model Pipeline")
    print("="*50)
    
    # 2. Load the Dataset
    try:
        df = pd.read_csv('dataset/train.csv')
    except FileNotFoundError:
        print("Error: dataset/train.csv not found. Please ensure the dataset exists.")
        return

    print("\n[Step 2] Dataset Loading")
    print("First 5 rows:")
    print(df.head())
    print(f"\nDataset Shape: {df.shape}")
    print(f"Column Names: {list(df.columns)}")
    
    # 3. Data Exploration (EDA)
    print("\n[Step 3] Data Exploration (EDA)")
    print("\nDataset Info:")
    df.info()
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing values count:")
    print(df.isnull().sum())
    
    # Identify target column as the last column
    target_col = df.columns[-1]
    print(f"\nAssumed Target Variable: '{target_col}'")
    print("\nValue distribution of target variable:")
    print(df[target_col].value_counts())
    
    # Generate correlation heatmap
    print("\nGenerating visualizations...")
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        print("--> Saved 'correlation_heatmap.png'")
    plt.close()

    # Generate feature distributions for up to 4 numerical columns
    num_features = numeric_df.columns[:min(4, len(numeric_df.columns))]
    if len(num_features) > 0:
        plt.figure(figsize=(12, 10))
        for i, col in enumerate(num_features, 1):
            plt.subplot(2, 2, i)
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        print("--> Saved 'feature_distributions.png'")
        plt.close()

    # 4. Data Cleaning
    print("\n[Step 4] Data Cleaning")
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Remove duplicate rows
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicate rows.")

    # Convert categorical to numeric (Label Encoding)
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    print(f"Dataset has no null values: {df.isnull().sum().sum() == 0}")

    # 5. Feature Selection
    print("\n[Step 5] Feature Selection")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")

    # 6. Train-Test Split
    print("\n[Step 6] Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")

    # 7. Model Selection & 8. Model Training
    print("\n[Step 7 & 8] Model Training")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree Classification": DecisionTreeClassifier(random_state=42),
        "Random Forest Classification": RandomForestClassifier(random_state=42, n_estimators=100)
    }

    best_model_name = ""
    best_accuracy = 0
    best_model = None

    # Print results section by section
    print("\n[Step 9 & 12] Model Evaluation & Results")
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # 9. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"\n--- Output Results: {name} ---")
        print(f"Model Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model

    # 10. Best Model Selection
    print("\n[Step 10] Best Model Selection")
    print(f"The best performing model based on accuracy is: **{best_model_name}** with {best_accuracy:.4f} accuracy.")

    # 11. Predictions
    print("\n[Step 11] Predictions on Testing Dataset (Sample)")
    sample_X = X_test.head(10)
    sample_y = y_test.head(10)
    sample_preds = best_model.predict(sample_X)
    
    # Inverse transform labels if the target was categorical
    if target_col in label_encoders:
        le_target = label_encoders[target_col]
        actual_labels = le_target.inverse_transform(sample_y.values)
        predicted_labels = le_target.inverse_transform(sample_preds)
    else:
        actual_labels = sample_y.values
        predicted_labels = sample_preds
        
    prediction_df = pd.DataFrame({
        "Actual Credit Score": actual_labels,
        "Predicted Credit Score": predicted_labels
    })
    print(prediction_df)

    # 12. Predict on Actual Test Data
    print("\n[Testing] Predicting on dataset/test.csv")
    try:
        test_df = pd.read_csv('dataset/test.csv', low_memory=False)
        print(f"Loaded actual test dataset: {test_df.shape}")
        
        # Handle Missing Values in Test Data
        for col in test_df.columns:
            if test_df[col].dtype == 'object':
                mode_val = test_df[col].mode()
                test_df[col] = test_df[col].fillna(mode_val[0] if not mode_val.empty else "Unknown")
            else:
                test_df[col] = test_df[col].fillna(test_df[col].median() if not pd.isna(test_df[col].median()) else 0)

        # Label Encodings using previously fitted encoders
        for col in categorical_cols:
            if col in test_df.columns and col in label_encoders:
                le = label_encoders[col]
                test_df[col] = test_df[col].astype(str)
                # Map unseen labels to the first known class to prevent errors
                known_classes = set(le.classes_)
                test_df[col] = test_df[col].apply(lambda x: x if x in known_classes else str(le.classes_[0]))
                test_df[col] = le.transform(test_df[col])
        
        # Match columns safely
        if target_col in test_df.columns:
            X_real_test = test_df.drop(columns=[target_col])
        else:
            X_real_test = test_df
        
        # Keep only variables that model training saw
        X_real_test = X_real_test[X.columns]
        
        # Generate Predictions
        final_predictions = best_model.predict(X_real_test)
        
        if target_col in label_encoders:
            final_predictions = label_encoders[target_col].inverse_transform(final_predictions)
            
        test_df_original = pd.read_csv('dataset/test.csv', usecols=['ID', 'Customer_ID'], low_memory=False)
        test_df_original['Predicted_Score'] = final_predictions
        test_df_original.to_csv('final_predictions.csv', index=False)
        
        print("--> Saved predictions to 'final_predictions.csv'")
        print(test_df_original.head())
        
    except FileNotFoundError:
        print("dataset/test.csv not found, skipping...")

    print("\n" + "="*50)
    print("Machine Learning Pipeline Execution Completed.")
    print("="*50)

if __name__ == "__main__":
    main()