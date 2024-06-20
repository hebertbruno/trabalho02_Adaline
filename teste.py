from metricas import calcular_pearson2, calcular_mse2, calc_acuracia

def test_model(model, X_test, y_test):
    y_pred = (model.predict(X_test)).astype(int)
    #print ("Caracteristicas:", X_test)
    print ("valores previstos", y_pred)
    print ("valores testados", y_test)
    acuracia = calc_acuracia(y_test, y_pred)
    pearson = calcular_pearson2(y_test, y_pred)
    mse = calcular_mse2(y_test, y_pred)
    return acuracia, pearson, mse
