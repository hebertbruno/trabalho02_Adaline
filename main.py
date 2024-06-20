import numpy as np
from data import load_data, create_folds, preprocess_data
from treinamento import train_model
from teste import test_model
from metricas import media



def min_max_normalization(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    X_normalized = (X - min_vals) / (max_vals - min_vals)
    return X_normalized

def main():
    # Carregar e preparar os dados
    file_path = 'hospital.xls'
    X, y = load_data(file_path)
    
    # Normalizar os dados
    #print("X normal:", X)
    X_normalized = min_max_normalization(X)
    #print(X_normalized)
    X_float = X_normalized.astype(float)
    #print ("agora float:", X_float)
    
    
    # Criar as pastas
    folds = create_folds(X_float, y)

    # Listas para armazenar as métricas
    acuracias = []
    coeficientes_pearson = []
    mse_values = []

    # Validação cruzada
    for index, fold in enumerate(folds, start=1):
        X_treino, y_treino, X_teste, y_teste = fold
       # print (X_teste)
       # print (y_teste)
        # Treinar o modelo
        model = train_model(X_treino, y_treino)
        
        # Testar o modelo
        acuracia, pearson, mse = test_model(model, X_teste, y_teste)
        
        # Armazenar as métricas
        acuracias.append(acuracia)
        coeficientes_pearson.append(pearson)
        mse_values.append(mse)
    
    # Calcular métricas médias
    media_acuracia = media(acuracias)
    media_pearson = media(coeficientes_pearson)
    media_mse = media(mse_values)
    

    print("Acurácias:", acuracias)
    print("Acurácia Média:", media_acuracia)
    print("Coeficientes de Pearson:", coeficientes_pearson)
    print("Coeficiente de Pearson Médio:", media_pearson)
    print("Valores do Erro Médio Quadrático:", mse_values)
    print("MSE Médio:", media_mse)
    
    
if __name__ == "__main__":
    main()
