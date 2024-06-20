import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

def load_data(file_path):
    data = pd.read_excel(file_path)

    # Convertendo variáveis categóricas para numéricas (Sex, Smoker/não Smoker)
    data['Sex'] = data['Sex'].map({'Female': 0, 'Male': 1})
    #data['Smoker'] = data['Smoker'].map({'FALSE': 0, 'TRUE': 1})
    data['Smoker'] = data['Smoker'].replace({'FALSO': 0, 'VERDADEIRO': 1})

    # Separando variáveis de predição e alvo
    X = data[['Sex', 'Age', 'Weight', 'Smoker']].values
    y = data['BloodPressure_1'].values
    
    return X, y

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def create_folds(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    folds = []

    for train_index, test_index in kf.split(X):
        X_treino, X_teste = X[train_index], X[test_index]
        y_treino, y_teste = y[train_index], y[test_index]
        folds.append((X_treino, y_treino, X_teste, y_teste))

    return folds
