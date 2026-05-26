import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Eksperimen_SML_Jessica-Leo")

mlflow.sklearn.autolog()

if __name__ == "__main__":
    df = pd.read_csv('diabetes_preprocessed.csv')
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        print("Training selesai, metrik dan artefak otomatis dicatat oleh autolog.")