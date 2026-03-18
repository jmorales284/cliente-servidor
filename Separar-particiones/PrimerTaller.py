import numpy as np
import matplotlib.pyplot as plt
import time
from keras.datasets import mnist
import copy

# Cargar MNIST (entrenamiento y prueba)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Aplanar y normalizar
x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
x_test = test_images.reshape(test_images.shape[0], -1) / 255.0

# One-hot encoding
def one_hot(y, num_classes=10):
    one_hot_matrix = np.zeros((len(y), num_classes))
    one_hot_matrix[np.arange(len(y)), y] = 1
    return one_hot_matrix

y_train = one_hot(train_labels)
y_test = one_hot(test_labels)

def split_data(x, y, k):
    m = x.shape[0]
    indices = np.arange(m)
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    split_sizes = [m // k] * k
    for i in range(m % k):
        split_sizes[i] += 1
    splits = []
    start = 0
    for size in split_sizes:
        end = start + size
        splits.append((x_shuffled[start:end], y_shuffled[start:end]))
        start = end
    return splits

# Parámetros fijos
INPUT_SIZE = 784
HIDDEN_SIZE = 72
OUTPUT_SIZE = 10

def initialize_parameters():
    np.random.seed(42)
    W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0 / INPUT_SIZE)
    b1 = np.zeros((1, HIDDEN_SIZE))
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0 / HIDDEN_SIZE)
    b2 = np.zeros((1, OUTPUT_SIZE))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

def backward(X, y, params, cache):
    m = X.shape[0]
    W2 = params['W2']
    a1, z1, a2 = cache['a1'], cache['z1'], cache['a2']
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(params, grads, learning_rate):
    params['W1'] -= learning_rate * grads['dW1']
    params['b1'] -= learning_rate * grads['db1']
    params['W2'] -= learning_rate * grads['dW2']
    params['b2'] -= learning_rate * grads['db2']
    return params

def average_parameters(params_list):
    avg_params = {}
    for key in params_list[0].keys():
        avg_params[key] = np.mean([p[key] for p in params_list], axis=0)
    return avg_params

def train_post_average(partitions, epochs, learning_rate):
    """
    Entrena un modelo por partición de forma independiente durante todas las épocas,
    luego promedia los parámetros finales.
    Retorna: modelo promediado, pérdidas de cada partición, tiempo total.
    """
    num_partitions = len(partitions)
    params_list = [initialize_parameters() for _ in range(num_partitions)]
    losses_per_partition = [[] for _ in range(num_partitions)]
    start_time = time.time()

    for i in range(num_partitions):
        x_part, y_part = partitions[i]
        print(f"\nEntrenando partición {i+1} de forma independiente...")
        for epoch in range(epochs):
            a2, cache = forward(x_part, params_list[i])
            loss = compute_loss(a2, y_part)
            losses_per_partition[i].append(loss)
            grads = backward(x_part, y_part, params_list[i], cache)
            params_list[i] = update_parameters(params_list[i], grads, learning_rate)
            if epoch % 10 == 0:
                print(f"  Época {epoch}, pérdida partición {i+1}: {loss:.4f}")

    elapsed = time.time() - start_time
    avg_params = average_parameters(params_list)
    return avg_params, losses_per_partition, elapsed

def train_iterative_average(partitions, epochs, learning_rate, x_test, y_test_labels):
    """
    En cada época, entrena cada partición por separado y luego promedia los parámetros.
    El promedio se usa como inicio para la siguiente época.
    Retorna: modelo final, pérdidas de cada partición, precisiones en test por época, tiempo.
    """
    num_partitions = len(partitions)
    params_list = [initialize_parameters() for _ in range(num_partitions)]
    losses_per_partition = [[] for _ in range(num_partitions)]
    test_accuracies = []  # Nueva lista para almacenar precisión en test por época
    start_time = time.time()

    for epoch in range(epochs):
        # Entrenar cada partición una época
        for i in range(num_partitions):
            x_part, y_part = partitions[i]
            a2, cache = forward(x_part, params_list[i])
            loss = compute_loss(a2, y_part)
            losses_per_partition[i].append(loss)
            grads = backward(x_part, y_part, params_list[i], cache)
            params_list[i] = update_parameters(params_list[i], grads, learning_rate)

        # Promediar parámetros
        avg_params = average_parameters(params_list)
        for i in range(num_partitions):
            params_list[i] = avg_params.copy()

        # Evaluar el modelo promediado en test
        acc = accuracy(x_test, y_test_labels, avg_params)
        test_accuracies.append(acc)

        if epoch % 10 == 0:
            avg_loss = np.mean([losses_per_partition[i][-1] for i in range(num_partitions)])
            print(f"Época {epoch}, pérdida promedio: {avg_loss:.4f}, precisión test: {acc*100:.2f}%")

    elapsed = time.time() - start_time
    final_params = params_list[0]
    return final_params, losses_per_partition, test_accuracies, elapsed

def predict(X, params):
    a2, _ = forward(X, params)
    return np.argmax(a2, axis=1)

def accuracy(X, y_labels, params):
    preds = predict(X, params)
    return np.mean(preds == y_labels)

# Hiperparámetros
epochs = 100
learning_rate = 0.1
k = 6
partitions = split_data(x_train, y_train, k)

print("=" * 50)
print("MÉTODO 1: Promedio al final")
print("=" * 50)
model_post, losses_post, time_post = train_post_average(partitions, epochs, learning_rate)
acc_post = accuracy(x_test, test_labels, model_post)
print(f"\nPrecisión en test (promedio al final): {acc_post * 100:.2f}%")
print(f"Tiempo de entrenamiento: {time_post:.2f} s")

print("\n" + "=" * 50)
print("MÉTODO 2: Promedio por época")
print("=" * 50)
model_iter, losses_iter, acc_iter_epoch, time_iter = train_iterative_average(partitions, epochs, learning_rate, x_test, test_labels)
acc_iter = accuracy(x_test, test_labels, model_iter)
print(f"\nPrecisión en test (promedio por época): {acc_iter * 100:.2f}%")
print(f"Tiempo de entrenamiento: {time_iter:.2f} s")

# ---------------------------
# Gráficas comparativas
# ---------------------------
plt.figure(figsize=(16, 10))

# 1. Pérdidas de entrenamiento (promedio por época para iterativo, individuales para post)
plt.subplot(2, 3, 1)
# Para el método iterativo, calculamos la pérdida promedio por época
loss_iter_avg = [np.mean([losses_iter[j][i] for j in range(k)]) for i in range(epochs)]
plt.plot(loss_iter_avg, label='Iterativo (promedio)', linewidth=2)
# Para el método post, graficamos las pérdidas de cada partición (con transparencia)
for i in range(k):
    plt.plot(losses_post[i], alpha=0.5, linewidth=1, label=f'Partición {i+1}' if i==0 else "")
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida en entrenamiento')
plt.legend()
plt.grid(True)

# 2. Precisión en test por época (solo iterativo)
plt.subplot(2, 3, 2)
plt.plot(acc_iter_epoch, label='Iterativo', linewidth=2, color='orange')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión en test por época')
plt.legend()
plt.grid(True)

# 3. Tiempo de entrenamiento (barras)
plt.subplot(2, 3, 3)
plt.bar(['Promedio al final', 'Promedio por época'], [time_post, time_iter], color=['blue', 'orange'])
plt.ylabel('Tiempo (s)')
plt.title('Tiempo de entrenamiento')

# 4. Precisión final (barras)
plt.subplot(2, 3, 4)
plt.bar(['Promedio al final', 'Promedio por época'], [acc_post*100, acc_iter*100], color=['blue', 'orange'])
plt.ylabel('Precisión (%)')
plt.title('Precisión final en test')

# 5. Comparación de pérdidas (ambos métodos en misma escala)
plt.subplot(2, 3, 5)
plt.plot(loss_iter_avg, label='Iterativo (promedio)', linewidth=2)
# Para el método post, mostramos la pérdida de la primera partición como ejemplo (o promedio)
loss_post_avg = np.mean(losses_post, axis=0)  # promedio de particiones
plt.plot(loss_post_avg, label='Post (promedio particiones)', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Comparación de pérdidas (promedio)')
plt.legend()
plt.grid(True)

# 6. Diferencia de precisión (si hubiera un centralizado, pero aquí no)
plt.subplot(2, 3, 6)
# No hay centralizado, mostramos la precisión del iterativo
plt.plot(acc_iter_epoch, color='green', linewidth=2)
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión del método iterativo')
plt.grid(True)

plt.tight_layout()
plt.savefig('comparacion_metodos.png', dpi=150)
plt.show()

print("\nGráfica guardada como 'comparacion_metodos.png'")