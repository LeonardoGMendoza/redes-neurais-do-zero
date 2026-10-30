# Redes Neurais do Zero (NumPy)

Primeiro passo de estudos de redes neurais artificiais, rumo a um app de IA que seja "os olhos" de pessoas cegas.

## O que tem aqui
`rede_neural_do_zero.py`: rede neural implementada só com NumPy, com cada linha comentada em português.

- **Parte A**: perceptron (1 neurônio) aprendendo a porta AND
- **Parte B**: rede com camada oculta aprendendo XOR
- **Parte C**: a mesma rede decidindo se deve avisar a pessoa sobre um obstáculo (dados simulados)

Conceitos praticados: pesos, bias, função de ativação (sigmoid), forward pass, função de perda, backpropagation e gradiente descendente.

## Como rodar
```bash
pip install numpy
python rede_neural_do_zero.py
```
Ou cole o código no Google Colab.

## Próximos passos
1. Desafios no fim do arquivo (ReLU, tamanho da camada oculta, nova entrada, dados de teste)
2. PyTorch com o dataset MNIST
3. CNNs para imagens, rumo à detecção de obstáculos com a câmera do celular
