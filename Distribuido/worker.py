import numpy as np
import socket
import pickle
import struct
from keras.datasets import mnist

# ------------------------------------------------------------
# Funciones de la red neuronal (deben coincidir con el servidor)
# ------------------------------------------------------------
INPUT_SIZE = 784
HIDDEN_SIZE = 72
OUTPUT_SIZE = 10

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
# Conectar al servidor y recibir la partición
# ------------------------------------------------------------
PINGGY_HOST = 'sqntz-2800-e2-627f-fbf9-7ea7-c26d-c88f-885c.a.free.pinggy.link'
PINGGY_PORT = 36559

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((PINGGY_HOST, PINGGY_PORT))
print("Conectado al servidor. Recibiendo partición...")

# Recibir índices de la partición
indices_particion = recv_message(sock)
# Recibir learning rate
learning_rate = recv_message(sock)
print(f"Partición recibida: {len(indices_particion)} ejemplos")

# ------------------------------------------------------------
# Cargar datos completos (para extraer la partición)
# ------------------------------------------------------------
(train_images, train_labels), (_, _) = mnist.load_data()
x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
y_train = one_hot(train_labels)

# Extraer la partición asignada
x_part = x_train[indices_particion]
y_part = y_train[indices_particion]

print(f"Datos cargados. Partición de tamaño: {x_part.shape[0]}")

# ------------------------------------------------------------
# Bucle principal: recibir parámetros, entrenar una época, devolver
# ------------------------------------------------------------
while True:
    # Recibir parámetros globales
    params = recv_message(sock)
    if params is None:
        print("Servidor cerró conexión.")
        break

    # Entrenar una época completa sobre la partición (puedes usar mini-batches internamente)
    # Aquí implementamos un entrenamiento sencillo: una época = un solo paso con todos los datos
    # (Equivalente a batch gradient descent en la partición)
    # Si quieres usar mini-batches, deberías hacer un bucle interno.

    # Forward con todos los datos de la partición
    a2, cache = forward(x_part, params)
    loss = compute_loss(a2, y_part)

    # Backward
    grads = backward(x_part, y_part, params, cache)

    # Actualizar parámetros localmente (una época = una actualización con todos los datos)
    params_actualizados = update_parameters(params, grads, learning_rate)

    # Enviar los parámetros actualizados al servidor, junto con la pérdida y número de muestras
    send_message(sock, (params_actualizados, loss, len(x_part)))

sock.close()