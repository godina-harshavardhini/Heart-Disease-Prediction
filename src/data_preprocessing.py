import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path="../data/Heart_disease.csv"):
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        
        # Display column names
        print("Columns in dataset:", df.columns.tolist())
        df.fillna(df.median(), inplace=True)

# Save the cleaned dataset
        df.to_csv("processed_heart_disease.csv", index=False)

        print("✅ Preprocessed dataset saved as 'processed_heart_disease.csv'")
        # Check if 'CVD' column exists
        if 'CVD' not in df.columns:
            raise KeyError("The column 'CVD' is missing in the dataset. Check the column names.")
        
        # Separate features and target
        X = df.drop(columns=['CVD'])
        y = df['CVD']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print("Data preprocessing completed successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler
    
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found. Check the file path.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except KeyError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data()
