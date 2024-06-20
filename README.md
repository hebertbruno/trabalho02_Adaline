# Classificadores de Regressão Linear: ADALINE 

## Introdução

Este projeto implementa um modelo ADALINE (Adaptive Linear Neuron) para prever a pressão arterial com base em dados de características de pacientes de um hospital. O algoritmo ADALINE é um modelo de aprendizado supervisionado que utiliza uma função de ativação linear para minimizar o erro quadrático entre as previsões e os valores reais. Este estudo utiliza validação cruzada para avaliar o desempenho do modelo.

## Estrutura do Projeto

O projeto está organizado nos seguintes arquivos:

- `main.py`: Contém a função principal que carrega os dados, normaliza, cria as pastas para validação cruzada, treina o modelo e avalia o desempenho.
- `data.py`: Contém funções para carregar, pré-processar os dados e criar pastas para validação cruzada.
- `treinamento.py`: Contém a função de chamada para treinar o modelo ADALINE.
- `teste.py`: Contém a função para testar o modelo ADALINE e calcular métricas de desempenho.
- `adaline.py:` Contém as formulas usadas no treinamento e predição do Adaline
- `metricas.py`: Contém funções para calcular métricas como a média.

## Requisitos

Certifique-se de ter o Python 3.x instalado em seu sistema. As seguintes bibliotecas Python são necessárias:

- `numpy`
- `pandas`
- `scikit-learn`

Você pode instalar todas as bibliotecas necessárias usando o seguinte comando:

```bash
pip install numpy pandas scikit-learn
````

## Como Executar o Código 
- Clone o repositório para o seu diretório local.
- Navegue até o diretório do projeto.
- Coloque o arquivo hospital.xls contendo os dados no mesmo diretório do projeto.
- Execute o arquivo main.py para treinar e avaliar o modelo

```bash
python main.py
````
# Interpretação dos Resultados
## Métricas
- `Coeficiente de Pearson:` Mede a correlação linear entre os valores previstos e os valores reais. Valores próximos de 1 indicam uma forte correlação positiva.
- `Erro Médio Quadrático (MSE):` Mede a média dos quadrados dos erros. Valores menores indicam melhor desempenho do modelo.
## Resultados Obtidos
- `Coeficientes de Pearson:` 0.773, 0.544, 0.749, 0.783, 0.598
- `Coeficiente de Pearson Médio:` 0.690
- `Valores do Erro Médio Quadrático (MSE):` 27.2,25.3,24.0,22.9,20.3
- `MSE Médio:` 23.94

Estes resultados indicam que o modelo tem uma correlação moderada a alta entre os valores previstos e os valores reais, com um erro médio quadrático relativamente baixo, sugerindo um desempenho aceitável do modelo para esta aplicação.
