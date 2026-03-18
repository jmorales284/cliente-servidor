import numpy as np
import multiprocessing as mp
import copy
import time
from keras.datasets import mnist
import matplotlib.pyplot as plt

# ---------------------------
# 1. Carga y preprocesamiento de datos
# ---------------------------
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Aplanar y normalizar
x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
x_test = test_images.reshape(test_images.shape[0], -1) / 255.0

def one_hot(y, num_classes=10):
    one_hot_matrix = np.zeros((len(y), num_classes))
    one_hot_matrix[np.arange(len(y)), y] = 1
    return one_hot_matrix

y_train = one_hot(train_labels)
y_test = one_hot(test_labels)

def split_data(x, y, k):
    """Divide los datos en k particiones mezcladas."""
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

# ---------------------------
# 2. Funciones de la red neuronal
# ---------------------------
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

# ---------------------------
# 4. Método paralelo con multiprocessing
# ---------------------------
# Variables globales para los workers (se fijan en init_worker)
_PARTITIONS = None
_PARTITION_INDEX = None

def init_worker(partitions, index_queue):
    """Inicializa cada worker con su partición fija."""
    global _PARTITIONS, _PARTITION_INDEX
    _PARTITIONS = partitions
    _PARTITION_INDEX = index_queue.get()   # índice exclusivo

def worker_train(params, learning_rate):
    """
    Worker que entrena una época con su partición fija.
    Devuelve (parámetros_actualizados, pérdida).
    """
    global _PARTITIONS, _PARTITION_INDEX
    x_part, y_part = _PARTITIONS[_PARTITION_INDEX]

    # Copia local para no modificar los parámetros originales
    local_params = copy.deepcopy(params)

    a2, cache = forward(x_part, local_params)
    loss = compute_loss(a2, y_part)

    grads = backward(x_part, y_part, local_params, cache)
    local_params = update_parameters(local_params, grads, learning_rate)

    return local_params, loss

def train_iterative_average_parallel(partitions, epochs, learning_rate, num_workers=None, x_test=None, y_test_labels=None):
    """
    Versión paralela de train_iterative_average.
    partitions: lista de tuplas (x_i, y_i)
    epochs: número de épocas
    learning_rate: tasa de aprendizaje
    num_workers: número de procesos (por defecto, igual al número de particiones)
    x_test: datos de test (opcional, para evaluar)
    y_test_labels: etiquetas de test en formato one-hot (opcional)
    Retorna: (modelo_final, lista_pérdidas_promedio_por_época, lista_precisiones_por_época, tiempo)
    """
    if num_workers is None:
        num_workers = len(partitions)

    index_queue = mp.Queue()
    for i in range(len(partitions)):
        index_queue.put(i)

    pool = mp.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(partitions, index_queue)
    )

    global_params = initialize_parameters()
    avg_losses = []
    test_accuracies = []  # Nueva lista para precisiones
    start_time = time.time()

    try:
        for epoch in range(epochs):
            tasks = [(global_params, learning_rate)] * len(partitions)
            results = pool.starmap(worker_train, tasks)

            params_list = [r[0] for r in results]
            losses = [r[1] for r in results]

            global_params = average_parameters(params_list)
            avg_loss = np.mean(losses)
            avg_losses.append(avg_loss)

            # Evaluar en test si se proporcionaron datos
            if x_test is not None and y_test_labels is not None:
                acc = accuracy(x_test, y_test_labels, global_params)
                test_accuracies.append(acc)
            else:
                test_accuracies.append(0.0)

            if epoch % 10 == 0:
                if x_test is not None:
                    print(f"Época {epoch}, pérdida promedio: {avg_loss:.4f}, precisión test: {acc*100:.2f}%")
                else:
                    print(f"Época {epoch}, pérdida promedio: {avg_loss:.4f}")

    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - start_time
    return global_params, avg_losses, test_accuracies, elapsed

# ---------------------------
# 5. Evaluación
# ---------------------------
def predict(X, params):
    a2, _ = forward(X, params)
    return np.argmax(a2, axis=1)

def accuracy(X, y_labels, params):
    preds = predict(X, params)
    return np.mean(preds == y_labels)

# ---------------------------
# 6. Ejecución principal
# ---------------------------
if __name__ == "__main__":
    # Hiperparámetros
    epochs = 100
    learning_rate = 0.1
    k = 2   # número de particiones

    partitions = split_data(x_train, y_train, k)
    print(f"Particiones: {len(partitions)}")
    for i, (x_part, y_part) in enumerate(partitions):
        print(f"Partición {i+1}: {x_part.shape[0]} ejemplos")

    # Método: Promedio por época (paralelo)
    print("\n" + "=" * 50)
    print("MÉTODO: Promedio por época (paralelo con multiprocessing)")
    print("=" * 50)
    model_parallel, losses_parallel, acc_parallel_epoch, time_parallel = train_iterative_average_parallel(
        partitions, epochs, learning_rate, num_workers=k, x_test=x_test, y_test_labels=test_labels
    )
    acc_parallel = accuracy(x_test, test_labels, model_parallel)
    print(f"\nPrecisión en test final: {acc_parallel * 100:.2f}%")
    print(f"Tiempo de entrenamiento: {time_parallel:.2f} s")

    # ---------------------------
    # Gráficas de evaluación
    # ---------------------------
    plt.figure(figsize=(16, 10))

    # 1. Pérdida promedio por época
    plt.subplot(2, 3, 1)
    plt.plot(losses_parallel, label='Pérdida promedio', linewidth=2, color='blue')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de entrenamiento (promedio por época)')
    plt.legend()
    plt.grid(True)

    # 2. Precisión en test por época
    plt.subplot(2, 3, 2)
    plt.plot(acc_parallel_epoch, label='Precisión test', linewidth=2, color='orange')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión en test por época')
    plt.legend()
    plt.grid(True)

    # 3. Tiempo de entrenamiento (barra)
    plt.subplot(2, 3, 3)
    plt.bar(['Paralelo'], [time_parallel], color='green')
    plt.ylabel('Tiempo (s)')
    plt.title('Tiempo de entrenamiento')

    # 4. Precisión final (barra)
    plt.subplot(2, 3, 4)
    plt.bar(['Paralelo'], [acc_parallel * 100], color='green')
    plt.ylabel('Precisión (%)')
    plt.title('Precisión final en test')

    # 5. Pérdida en escala logarítmica
    plt.subplot(2, 3, 5)
    plt.semilogy(losses_parallel, label='Pérdida', linewidth=2, color='blue')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (log)')
    plt.title('Pérdida en escala logarítmica')
    plt.legend()
    plt.grid(True)

    # 6. (Opcional) Diferencia con algún otro método, aquí no aplica, mostramos la precisión
    plt.subplot(2, 3, 6)
    plt.plot(acc_parallel_epoch, color='purple', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Evolución de la precisión')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('evaluacion_paralelo.png', dpi=150)
    plt.show()
    print("\nGráfica guardada como 'evaluacion_paralelo.png'")
    print("\n¡Entrenamiento completado!")