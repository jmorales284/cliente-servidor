import argparse
import io
import pickle
import socket
import struct
import time
from typing import Any, List

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ---------------------------
# Funciones de red (igual que en el servidor, usar send_tensor/recv_tensor)
# ---------------------------
def send_tensor(sock: socket.socket, obj: Any) -> None:
    import io
    if isinstance(obj, torch.Tensor):
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data = buffer.getvalue()
    else:
        data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)

def recv_tensor(sock: socket.socket) -> Any:
    def recvall(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Conexión cerrada")
            buf.extend(chunk)
        return bytes(buf)
    raw_len = recvall(8)
    (length,) = struct.unpack("!Q", raw_len)
    data = recvall(length)
    try:
        buffer = io.BytesIO(data)
        return torch.load(buffer)
    except:
        return pickle.loads(data)

send_obj = send_tensor
recv_obj = recv_tensor

# ---------------------------
# Modelo (idéntico)
# ---------------------------
class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ---------------------------
# DataLoader
# ---------------------------
def build_dataloader(rank, world_size, batch_size):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    return loader, sampler

# ---------------------------
# Worker principal con acumulación local
# ---------------------------
def run_worker(server_host, server_port, rank, world_size, batch_size, device_str=None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Worker {rank}] Usando dispositivo: {device}")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[Worker {rank}] Conectado a PS {server_host}:{server_port}")

    # Registro
    send_obj(sock, {"type": "register", "rank": rank, "world_size": world_size})
    cfg = recv_obj(sock)
    assert cfg["type"] == "config"
    param_names = cfg["param_names"]
    epochs = cfg["epochs"]
    steps_per_epoch = cfg["steps_per_epoch"]    # número de actualizaciones (envíos) por época
    lr = cfg["lr"]
    server_step = cfg["step"]
    accumulation_steps = cfg.get("accumulation_steps", 1)   # cuántos batches acumular localmente

    model = Cifar10CNN().to(device)
    state_dict_cpu = cfg["state_dict"]
    model.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})
    criterion = nn.CrossEntropyLoss()

    train_loader, sampler = build_dataloader(rank, world_size, batch_size)
    data_iter = iter(train_loader)

    total_updates = epochs * steps_per_epoch   # número de veces que enviaremos gradientes
    local_batches_processed = 0

    # Para acumulación local: guardamos gradientes acumulados y suma de pérdidas
    accumulated_grads = [torch.zeros_like(p, device="cpu") for p in model.parameters()]
    accumulated_samples = 0
    accumulated_loss = 0.0
    accumulation_counter = 0

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        data_iter = iter(train_loader)

        for update_step in range(steps_per_epoch):
            # Acumular accumulation_steps batches localmente
            for _ in range(accumulation_steps):
                try:
                    x, y = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    x, y = next(data_iter)

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                model.train()
                # Zero gradientes del modelo ANTES del backward (no usar optimizer, manual)
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad = None

                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                # Acumular gradientes (en CPU) y pérdida
                with torch.no_grad():
                    for i, p in enumerate(model.parameters()):
                        if p.grad is not None:
                            accumulated_grads[i] += p.grad.detach().cpu() * x.size(0)
                    accumulated_samples += x.size(0)
                    accumulated_loss += loss.item() * x.size(0)
                accumulation_counter += 1

            # Después de accumulation_steps, enviamos los gradientes acumulados al servidor
            # Convertir gradientes acumulados a lista (en CPU)
            grads_to_send = [g.clone() for g in accumulated_grads]

            # Enviar mensaje con gradientes y pérdida acumulada
            avg_loss = accumulated_loss / accumulated_samples
            send_obj(sock, {
                "type": "gradients",
                "worker": rank,
                "step": server_step,
                "batch_size": accumulated_samples,   # tamaño efectivo del batch enviado
                "grads": grads_to_send,
                "loss": avg_loss
            })

            # Resetear acumuladores locales
            for g in accumulated_grads:
                g.zero_()
            accumulated_samples = 0
            accumulated_loss = 0.0
            accumulation_counter = 0

            # Esperar respuesta del servidor
            resp = recv_obj(sock)
            rtype = resp.get("type")

            if rtype == "update":
                state_cpu = resp["state_dict"]
                model.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})
                server_step = resp["step"]
                local_batches_processed += accumulation_steps
                if local_batches_processed % (steps_per_epoch * accumulation_steps) == 0:
                    print(f"[Worker {rank}] Época {epoch+1} completada")
            elif rtype == "resync":
                state_cpu = resp["state_dict"]
                model.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})
                server_step = resp["step"]
                print(f"[Worker {rank}] Resync, reintentando step")
                # No incrementamos local_batches_processed, reintentamos el mismo step
                continue
            elif rtype == "stop":
                print(f"[Worker {rank}] Detenido por PS")
                send_obj(sock, {"type": "done"})
                sock.close()
                return
            else:
                raise RuntimeError(f"Respuesta desconocida: {rtype}")

    # Fin del entrenamiento
    print(f"[Worker {rank}] Entrenamiento completado")
    send_obj(sock, {"type": "done"})
    sock.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    run_worker(
        server_host=args.server_host,
        server_port=args.server_port,
        rank=args.rank,
        world_size=args.world_size,
        batch_size=args.batch_size,
        device_str=args.device
    )

if __name__ == "__main__":
    main()
