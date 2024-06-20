import numpy as np

class Adaline:
    # Inicialização do Adaline recebendo a taxa de aprendizado e número de épocas
    def __init__(self, learning_rate, n_epochs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
     
    def fit(self, X, y):
        # Vetor de pesos
        self.weights_ = np.zeros(X.shape[1] + 1) # X.shape[1] = num colunas em X, ou seja, características
        
        self.cost_ = [] # Lista de custo em cada época
        
        for _ in range(self.n_epochs):
            net_input = self.net_input(X) # Calcula a entrada líquida
            output = self.purelin(net_input) # A saída é a própria entrada líquida, função de ativação
            errors = y - output # Cálculo dos erros saída desejada - saída prevista
            # Atualiza os pesos usando a regra de aprendizado do gradiente descendente
            self.weights_[1:] += self.learning_rate * X.T.dot(errors) 
            self.weights_[0] += self.learning_rate * errors.sum()
            cost = (errors**2).sum() / 2.0 
            self.cost_.append(cost) # Armazena o custo da época atual

        return self
    
    def net_input(self, X):
        # Produto escalar
        produto_escalar = (np.dot(X, self.weights_[1:]) + self.weights_[0]) # X*w + b)
        return produto_escalar
        
    def purelin(self, X): # Função de ativação f(x) = x
        return X
    
    def predict(self, X):
        return self.purelin(self.net_input(X)) # Faz a previsão usando a entrada líquida e a função de ativação

