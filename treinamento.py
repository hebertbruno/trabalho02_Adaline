from adaline import Adaline

def train_model(X_train, y_train, learning_rate=0.01, n_epochs=1000):
    adaline = Adaline(learning_rate=learning_rate, n_epochs=n_epochs)
    adaline.fit(X_train, y_train)
    return adaline
