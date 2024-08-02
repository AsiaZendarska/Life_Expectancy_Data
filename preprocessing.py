import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def preprocessing_data(file_path):
    df = pd.read_csv(file_path)

# zastępowanie spacji podłogą
    df.columns = df.columns.str.replace(" ","_")

# zastępowanie dużych liter małymi
    df.columns = df.columns.str.lower()

# podstawowe informacje
    print("Pierwsze kilka wierszy:")
    print(df.head())
    print("\nInformacje o DataFrame:")
    print(df.info())
    print("\nPodstawowe statystyki:")
    print(df.describe().T)

# sprawdzanie liczby brakujących wartości
    zero_count = df.isna().sum()
    print("\nLiczba brakujących wartości w każdej kolumnie:")
    print(zero_count)

# wyświetlanie czy zbiór jest zbalansowany za pomocą wykresu
    value_counts = df["status"].value_counts()
    print("Liczba wystąpień każdej unikalnej wartości w kolumnie 'Status':")
    print(value_counts)

# Usuwanie kolumny 'Country'
    df.drop(columns=['country'], inplace=True)
    
    if "status" in df.columns:
        plt.clf()
        ax = sns.countplot(x=df["status"])
        
        # Dodawanie wartości liczbowych wewnątrz prostokątów
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2, height / 2),
                        ha='center', va='center')

        plt.title('Zbiór danych')
        plt.show()

# konwersja kolumny 'Status' na wartości numeryczne za pomocą Label Encoding
    df["status"] = df["status"].map({"Developing": 0, "Developed": 1})

# Wypełnianie brakujących wartości średnią kolumny
    df.fillna(df.mean(), inplace=True)

    return df

def split_data(preprocessing_df, test_size=0.2, random_state=42):
    X = preprocessing_df.drop(columns=["status"], axis=1) 
    y = preprocessing_df["status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Classification Report:\n{report}")
    return accuracy, cm, report

class ModelRegLog:
    def __init__(self, file_path):
        self.file_path = file_path
        self.preprocessing_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None
        self.accuracy = None
        self.cm = None
        self.report = None

    def preper_dataset(self):
        self.preprocessing_df = preprocessing_data(self.file_path)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.preprocessing_df)
        self.model = train_model(self.X_train, self.y_train)
        self.accuracy, self.cm, self.report = evaluate_model(self.model, self.X_test, self.y_test)



    