import numpy as np
import socket
import pickle
import struct
import argparse
import time
from keras.datasets import mnist
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parámetros de la red y funciones compartidas
# ------------------------------------------------------------
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

def average_parameters(params_list):
    avg_params = {}
    for key in params_list[0].keys():
        avg_params[key] = np.mean([p[key] for p in params_list], axis=0)
    return avg_params

def predict(X, params):
    W1, b1, W2, b2 = params['W1'], params['b1'], params['W2'], params['b2']
    z1 = np.dot(X, W1) + b1
    a1 = np.maximum(0, z1)
    z2 = np.dot(a1, W2) + b2
    exp_z = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
    a2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
    return np.argmax(a2, axis=1)

def accuracy(X, y_labels, params):
    preds = predict(X, params)
    return np.mean(preds == y_labels)

def one_hot(y, num_classes=10):
    one_hot_matrix = np.zeros((len(y), num_classes))
    one_hot_matrix[np.arange(len(y)), y] = 1
    return one_hot_matrix

# ------------------------------------------------------------
# Comunicación
# ------------------------------------------------------------
def send_message(sock, obj):
    data = pickle.dumps(obj)
    length = len(data)
    sock.sendall(struct.pack('!I', length))
    sock.sendall(data)

def recv_all(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def recv_message(sock):
    raw_length = recv_all(sock, 4)
    if not raw_length:
        return None
    length = struct.unpack('!I', raw_length)[0]
    data = recv_all(sock, length)
    return pickle.loads(data)

# ------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Servidor de parámetros con particiones fijas')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Dirección IP de escucha (por defecto 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Puerto (por defecto 5000)')
    parser.add_argument('--workers', type=int, default=3, help='Número de workers esperados (por defecto 3)')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas (por defecto 100)')
    parser.add_argument('--lr', type=float, default=0.1, help='Tasa de aprendizaje (por defecto 0.1)')
    return parser.parse_args()

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
args = parse_args()
HOST = args.host
PORT = args.port
NUM_WORKERS = args.workers
EPOCHS = args.epochs
LEARNING_RATE = args.lr

# Cargar datos
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
y_train = one_hot(train_labels)
x_test = test_images.reshape(test_images.shape[0], -1) / 255.0
y_test_labels = test_labels

# Crear particiones fijas
def create_fixed_partitions(x, y, k):
    m = x.shape[0]
    indices = np.random.permutation(m)
    partitions = []
    split_sizes = [m // k] * k
    for i in range(m % k):
        split_sizes[i] += 1
    start = 0
    for size in split_sizes:
        end = start + size
        part_indices = indices[start:end]
        partitions.append(part_indices)
        start = end
    return partitions

partitions_indices = create_fixed_partitions(x_train, y_train, NUM_WORKERS)

# ------------------------------------------------------------
# Esperar conexiones
# ------------------------------------------------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_WORKERS)
print(f"Servidor escuchando en {HOST}:{PORT}")

worker_sockets = []
for i in range(NUM_WORKERS):
    client_sock, addr = server_socket.accept()
    print(f"Worker {i+1} conectado desde {addr}")
    send_message(client_sock, partitions_indices[i])
    send_message(client_sock, LEARNING_RATE)
    worker_sockets.append(client_sock)

print("Todos los workers conectados. Comenzando entrenamiento...\n")

# ------------------------------------------------------------
# Inicializar parámetros globales
# ------------------------------------------------------------
global_params = initialize_parameters()

# Métricas
train_losses_per_epoch = []
test_accuracies_per_epoch = []

# Medir tiempo
start_time = time.time()

# ------------------------------------------------------------
# Bucle de épocas
# ------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    print(f"Época {epoch}/{EPOCHS}")

    for sock in worker_sockets:
        send_message(sock, global_params)

    received_params = []
    epoch_loss_sum = 0.0
    total_samples_epoch = 0

    for idx, sock in enumerate(worker_sockets):
        respuesta = recv_message(sock)
        if respuesta is None:
            print(f"Worker {idx+1} desconectado")
            break
        params_actualizados, loss, num_samples = respuesta
        received_params.append(params_actualizados)
        epoch_loss_sum += loss * num_samples
        total_samples_epoch += num_samples

    if len(received_params) != NUM_WORKERS:
        print("No se recibieron todos los parámetros. Abortando.")
        break

    global_params = average_parameters(received_params)
    avg_epoch_loss = epoch_loss_sum / total_samples_epoch
    train_losses_per_epoch.append(avg_epoch_loss)

    acc = accuracy(x_test, y_test_labels, global_params)
    test_accuracies_per_epoch.append(acc)

    print(f"  Pérdida promedio: {avg_epoch_loss:.4f} | Precisión test: {acc*100:.2f}%\n")

# Tiempo total
elapsed_time = time.time() - start_time

# Cerrar conexiones
for sock in worker_sockets:
    sock.close()
server_socket.close()

print(f"\nTiempo total de entrenamiento: {elapsed_time:.2f} segundos")

# ------------------------------------------------------------
# Gráficas mejoradas
# ------------------------------------------------------------
if len(train_losses_per_epoch) > 0:
    epochs_range = range(1, len(train_losses_per_epoch) + 1)

    plt.figure(figsize=(16, 10))

    # 1. Pérdida de entrenamiento
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_losses_per_epoch, 'b-o', markersize=3, label='Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.title('Pérdida de entrenamiento por época')
    plt.grid(True)
    plt.legend()

    # 2. Precisión en test
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, test_accuracies_per_epoch, 'g-s', markersize=3, label='Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Precisión en test por época')
    plt.grid(True)
    plt.legend()

    # 3. Tiempo de entrenamiento (barra)
    plt.subplot(2, 3, 3)
    plt.bar(['Distribuido'], [elapsed_time], color='orange')
    plt.ylabel('Tiempo (s)')
    plt.title('Tiempo de entrenamiento')

    # 4. Precisión final (barra)
    plt.subplot(2, 3, 4)
    plt.bar(['Distribuido'], [test_accuracies_per_epoch[-1]*100], color='green')
    plt.ylabel('Precisión (%)')
    plt.title('Precisión final en test')

    # 5. Pérdida en escala logarítmica
    plt.subplot(2, 3, 5)
    plt.semilogy(epochs_range, train_losses_per_epoch, 'b-o', markersize=3, label='Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida (log)')
    plt.title('Pérdida en escala logarítmica')
    plt.grid(True)
    plt.legend()

    # 6. Precisión (repetida para mantener formato)
    plt.subplot(2, 3, 6)
    plt.plot(epochs_range, test_accuracies_per_epoch, 'purple', linewidth=2)
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.title('Evolución de la precisión')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('metricas_distribuidas.png', dpi=150)
    plt.show()
    print("Gráfica guardada como 'metricas_distribuidas.png'")