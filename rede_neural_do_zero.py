# =============================================================================
# EXERCÍCIO 1 — Rede neural do ZERO, só com NumPy
# Objetivo: entender pesos, bias, ativação, forward, perda, backprop e
# gradiente descendente antes de usar PyTorch.
#
# Parte A: Perceptron (1 neurônio) aprendendo a porta AND
# Parte B: Rede com camada oculta aprendendo XOR (o "problema clássico")
# Parte C: A mesma rede decidindo "AVISAR OBSTÁCULO?" — primeiro passo
#          na direção do app que será o "olho" de quem não enxerga.
#
# Como rodar: python rede_neural_do_zero.py   (ou cole no Google Colab)
# =============================================================================

import numpy as np

np.random.seed(42)  # semente fixa: os resultados ficam iguais toda vez que rodar


# -----------------------------------------------------------------------------
# Funções de ativação
# -----------------------------------------------------------------------------
def sigmoid(z):
    # Transforma qualquer número em um valor entre 0 e 1 (parece uma "probabilidade")
    return 1 / (1 + np.exp(-z))


def sigmoid_derivada(a):
    # Derivada da sigmoid, escrita em função da SAÍDA a = sigmoid(z).
    # O backprop precisa dela para saber "quanto" cada neurônio influenciou o erro.
    return a * (1 - a)


# =============================================================================
# PARTE A — PERCEPTRON: um único neurônio aprendendo AND
# =============================================================================
print("=" * 60)
print("PARTE A — Perceptron aprendendo AND")
print("=" * 60)

# Entradas: cada linha é um exemplo com 2 valores (x1, x2)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Saída esperada do AND: só dá 1 quando as duas entradas são 1
y_and = np.array([[0], [0], [0], [1]])

# Pesos (um para cada entrada) e bias (um "ajuste" que desloca a decisão)
w = np.random.randn(2, 1)   # matriz 2x1 com números aleatórios pequenos
b = np.zeros((1, 1))        # bias começa em zero

taxa_aprendizado = 0.5      # tamanho do "passo" a cada correção

for epoca in range(5000):
    # FORWARD: soma ponderada + ativação
    z = X @ w + b            # @ = multiplicação de matrizes
    saida = sigmoid(z)

    # PERDA: erro quadrático médio (quanto a rede errou, em média)
    erro = saida - y_and
    perda = np.mean(erro ** 2)

    # BACKPROP (aqui é simples porque só há 1 camada):
    # regra da cadeia -> dPerda/dz = erro * derivada da ativação
    delta = erro * sigmoid_derivada(saida)

    # GRADIENTE DESCENDENTE: anda no sentido contrário ao gradiente
    w -= taxa_aprendizado * (X.T @ delta)
    b -= taxa_aprendizado * np.sum(delta, axis=0, keepdims=True)

    if epoca % 1000 == 0:
        print(f"época {epoca:5d} | perda = {perda:.4f}")

print("\nResultado AND (arredondado):")
print(np.round(sigmoid(X @ w + b), 3).ravel(), "-> esperado", y_and.ravel())

# Curiosidade: tente trocar y_and por XOR ([0,1,1,0]) aqui.
# O perceptron NÃO consegue aprender — XOR não é separável por uma reta.
# Por isso precisamos de uma CAMADA OCULTA (Parte B).


# =============================================================================
# PARTE B — Rede com 1 camada oculta aprendendo XOR
# =============================================================================
class RedeNeural:
    """Rede de 2 camadas: entrada -> oculta -> saída."""

    def __init__(self, n_entradas, n_ocultos, n_saidas):
        # Camada 1 (entrada -> oculta)
        self.W1 = np.random.randn(n_entradas, n_ocultos)
        self.b1 = np.zeros((1, n_ocultos))
        # Camada 2 (oculta -> saída)
        self.W2 = np.random.randn(n_ocultos, n_saidas)
        self.b2 = np.zeros((1, n_saidas))

    def forward(self, X):
        # Guarda os valores intermediários: o backprop vai precisar deles
        self.a1 = sigmoid(X @ self.W1 + self.b1)          # ativação da camada oculta
        self.a2 = sigmoid(self.a1 @ self.W2 + self.b2)    # saída final
        return self.a2

    def backward(self, X, y, lr):
        n = X.shape[0]  # número de exemplos

        # 1) Erro na saída
        delta2 = (self.a2 - y) * sigmoid_derivada(self.a2)

        # 2) Erro "propagado para trás" até a camada oculta (regra da cadeia)
        delta1 = (delta2 @ self.W2.T) * sigmoid_derivada(self.a1)

        # 3) Atualiza pesos e bias de cada camada (gradiente descendente)
        self.W2 -= lr * (self.a1.T @ delta2) / n
        self.b2 -= lr * np.sum(delta2, axis=0, keepdims=True) / n
        self.W1 -= lr * (X.T @ delta1) / n
        self.b1 -= lr * np.sum(delta1, axis=0, keepdims=True) / n

    def treinar(self, X, y, epocas=10000, lr=1.0, mostrar_a_cada=2000):
        for epoca in range(epocas):
            saida = self.forward(X)
            perda = np.mean((saida - y) ** 2)
            self.backward(X, y, lr)
            if epoca % mostrar_a_cada == 0:
                print(f"época {epoca:5d} | perda = {perda:.4f}")


print("\n" + "=" * 60)
print("PARTE B — Rede com camada oculta aprendendo XOR")
print("=" * 60)

y_xor = np.array([[0], [1], [1], [0]])   # XOR: 1 quando as entradas são diferentes

rede_xor = RedeNeural(n_entradas=2, n_ocultos=4, n_saidas=1)
rede_xor.treinar(X, y_xor, epocas=10000, lr=1.0)

print("\nResultado XOR:")
for entrada, previsto in zip(X, rede_xor.forward(X)):
    print(f"  {entrada} -> {previsto[0]:.3f}")


# =============================================================================
# PARTE C — Na direção do app: "devo AVISAR a pessoa sobre um obstáculo?"
# =============================================================================
# Dados SIMULADOS (no app real, viriam da câmera/sensor do celular):
#   distancia: metros até o obstáculo (0 a 5)
#   no_caminho: 1 se o obstáculo está na frente, 0 se está ao lado
# Regra "real" que a rede precisa descobrir sozinha:
#   avisar = 1 se o obstáculo está no caminho E a menos de 2 metros
print("\n" + "=" * 60)
print("PARTE C — Rede decidindo se deve AVISAR sobre obstáculo")
print("=" * 60)

n_exemplos = 400
distancia = np.random.uniform(0, 5, n_exemplos)
no_caminho = np.random.randint(0, 2, n_exemplos)
avisar = ((distancia < 2) & (no_caminho == 1)).astype(int).reshape(-1, 1)

# Normalização: deixar as entradas na mesma escala (0 a 1) ajuda muito o treino
X_obs = np.column_stack([distancia / 5, no_caminho])

rede_obs = RedeNeural(n_entradas=2, n_ocultos=8, n_saidas=1)
rede_obs.treinar(X_obs, avisar, epocas=10000, lr=2.0)

acuracia = np.mean((rede_obs.forward(X_obs) > 0.5) == avisar)
print(f"\nAcurácia nos dados de treino: {acuracia * 100:.1f}%")

# Testando situações novas, como se fossem leituras da câmera
testes = [(0.8, 1), (1.5, 1), (3.0, 1), (0.5, 0), (4.5, 0)]
print("\nSimulação do app:")
for d, c in testes:
    prob = rede_obs.forward(np.array([[d / 5, c]]))[0, 0]
    onde = "à frente" if c else "ao lado"
    acao = "🔊 'Cuidado! Obstáculo à frente.'" if prob > 0.5 else "(silêncio)"
    print(f"  {d:.1f} m, {onde:8s} -> prob={prob:.2f}  {acao}")

# =============================================================================
# DESAFIOS para você fazer sozinho:
#  1. Troque sigmoid por ReLU na camada oculta (lembre de mudar a derivada).
#  2. Mude n_ocultos para 2, 4, 16 e veja o efeito na perda.
#  3. Na Parte C, adicione uma 3ª entrada: "velocidade da pessoa".
#  4. Separe 20% dos dados para TESTE e meça a acurácia só neles.
# =============================================================================
