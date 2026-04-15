import argparse
import csv
import math
import os
import pickle
import socket
import struct
import threading
import time
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T

try:
    import psutil
except Exception:
    psutil = None


# ---------------------------
# Utilidades de red
# ---------------------------

def send_obj(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Conexión cerrada")
        buf.extend(chunk)
    return bytes(buf)


def recv_obj(sock: socket.socket) -> Any:
    raw_len = recvall(sock, 8)
    (length,) = struct.unpack("!Q", raw_len)
    data = recvall(sock, length)
    return pickle.loads(data)


# ---------------------------
# Modelo
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
# Parameter Server
# ---------------------------

class ParameterServer:
    def __init__(self, host, port, num_workers, epochs, steps_per_epoch, lr):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

        self.device = torch.device("cpu")
        self.model = Cifar10CNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_list = [p for _, p in self.model.named_parameters()]

        self.global_step = 0
        self.total_steps = self.epochs * self.steps_per_epoch

        self.lock = threading.Lock()
        self.train_finished = False
        self.workers_registered = set()

        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self._proc = psutil.Process() if psutil else None

        # métricas
        self._t0_train = None
        self._t_end_train = None
        self._epoch_times = []
        self._epoch_start_t = None
        self._epoch_eval_threads = []

        self.history = []

        self.reset_aggregation_state()

    def reset_aggregation_state(self):
        self.agg_grads_sum = [torch.zeros_like(p, device="cpu") for p in self.param_list]
        self.agg_samples = 0
        self.agg_loss_sum = 0.0
        self.waiting_socks = []
        self.contributors = set()

    def start(self):
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.num_workers)
        print(f"[PS] Escuchando en {self.host}:{self.port}")

        while True:
            conn, addr = self.server_sock.accept()
            print(f"[PS] Nueva conexión desde {addr}")
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def handle_client(self, conn: socket.socket):
        try:
            while True:
                msg = recv_obj(conn)
                mtype = msg.get("type")

                if mtype == "register":
                    rank = msg["rank"]
                    self.workers_registered.add(rank)
                    print(f"[PS] Worker {rank} registrado desde {conn.getpeername()}")

                    state_dict_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

                    config_msg = {
                        "type": "config",
                        "param_names": self.param_names,
                        "epochs": self.epochs,
                        "steps_per_epoch": self.steps_per_epoch,
                        "lr": self.lr,
                        "step": self.global_step,
                        "state_dict": state_dict_cpu
                    }
                    send_obj(conn, config_msg)
                    print(f"[PS] Configuración enviada al Worker {rank}")

                elif mtype == "gradients":
                    with self.lock:
                        if self.train_finished:
                            send_obj(conn, {"type": "stop"})
                            return

                        worker_rank = msg["worker"]
                        worker_step = msg.get("step", -1)

                        if self._t0_train is None:
                            self._t0_train = time.perf_counter()
                            self._epoch_start_t = self._t0_train

                        # Si el worker está desincronizado, le mandamos resync
                        if worker_step != self.global_step:
                            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                            send_obj(conn, {
                                "type": "resync",
                                "state_dict": state,
                                "step": self.global_step
                            })
                            print(f"[PS] Resync enviado a worker {worker_rank}: worker_step={worker_step}, global_step={self.global_step}")
                            continue

                        # Evitar doble contribución del mismo worker al mismo step
                        if worker_rank in self.contributors:
                            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                            send_obj(conn, {
                                "type": "resync",
                                "state_dict": state,
                                "step": self.global_step
                            })
                            print(f"[PS] Worker {worker_rank} intentó contribuir dos veces al step {self.global_step}")
                            continue

                        grads = msg["grads"]
                        bs = msg["batch_size"]
                        loss_value = float(msg.get("loss", 0.0))

                        for i, g in enumerate(grads):
                            self.agg_grads_sum[i] += g * bs

                        self.agg_samples += bs
                        self.agg_loss_sum += loss_value * bs
                        self.waiting_socks.append(conn)
                        self.contributors.add(worker_rank)

                        if len(self.waiting_socks) == self.num_workers:
                            avg_grads = [g / self.agg_samples for g in self.agg_grads_sum]

                            for p, g in zip(self.param_list, avg_grads):
                                p.grad = g.to(self.device)

                            self.optimizer.step()
                            self.optimizer.zero_grad()

                            self.global_step += 1
                            avg_loss = self.agg_loss_sum / self.agg_samples

                            state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

                            for s in self.waiting_socks:
                                send_obj(s, {
                                    "type": "update",
                                    "state_dict": state,
                                    "step": self.global_step
                                })

                            print(f"[PS] Step {self.global_step}/{self.total_steps} completado | Train Loss={avg_loss:.4f}")

                            # FIN DE ÉPOCA
                            if self.global_step % self.steps_per_epoch == 0:
                                epoch = self.global_step // self.steps_per_epoch

                                now = time.perf_counter()
                                epoch_time = now - self._epoch_start_t
                                self._epoch_times.append(epoch_time)
                                self._epoch_start_t = now

                                ram = self._sample_ram()

                                t = threading.Thread(
                                    target=self._eval_epoch,
                                    args=(state, epoch, epoch_time, ram, avg_loss),
                                    daemon=True
                                )
                                t.start()
                                self._epoch_eval_threads.append(t)

                            # Reset agregación del siguiente step
                            self.reset_aggregation_state()

                            if self.global_step >= self.total_steps:
                                self.train_finished = True
                                self._t_end_train = time.perf_counter()

                                # Espera un poco para que los workers reciban el último update
                                threading.Thread(target=self.evaluate_and_report, daemon=True).start()
                                return

                elif mtype == "done":
                    print("[PS] Worker notificó finalización.")
                    return

                else:
                    print(f"[PS] Mensaje desconocido: {mtype}")

        except Exception as e:
            print(f"[PS] Error con cliente: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------------------
    # RAM
    # ---------------------------

    def _sample_ram(self):
        if not psutil:
            return {"ram_used": None}

        vm = psutil.virtual_memory()
        return {
            "ram_used": round((vm.total - vm.available) / (1024 ** 3), 3)
        }

    # ---------------------------
    # Evaluación
    # ---------------------------

    def _eval_epoch(self, state, epoch, epoch_time, ram, avg_loss):
        acc = self._evaluate(state)

        row = {
            "epoch": epoch,
            "global_step": self.global_step,
            "train_loss": round(avg_loss, 6),
            "accuracy": round(acc, 4),
            "time": round(epoch_time, 4),
            "ram_used": ram["ram_used"]
        }
        self.history.append(row)

        print(
            f"[PS] Epoch {epoch} | "
            f"Train Loss={avg_loss:.4f} | "
            f"Acc={acc:.2f}% | "
            f"Time={epoch_time:.2f}s | "
            f"RAM={ram['ram_used']} GB"
        )

    def _evaluate(self, state):
        eval_model = Cifar10CNN().to(self.device)
        eval_model.load_state_dict({k: v.to(self.device) for k, v in state.items()})
        eval_model.eval()

        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])

        test = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transform
        )
        loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False)

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                out = eval_model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        return 100.0 * correct / total

    # ---------------------------
    # GRÁFICAS
    # ---------------------------

    def plot_metrics(self):
        if not self.history:
            print("[PS] No hay métricas para graficar.")
            return

        os.makedirs("plots", exist_ok=True)

        epochs = [h["epoch"] for h in self.history]
        acc = [h["accuracy"] for h in self.history]
        time_ = [h["time"] for h in self.history]
        ram = [h["ram_used"] if h["ram_used"] is not None else 0 for h in self.history]
        loss = [h["train_loss"] for h in self.history]

        plt.figure()
        plt.plot(epochs, acc, marker="o")
        plt.title("Accuracy por época")
        plt.xlabel("Época")
        plt.ylabel("Accuracy (%)")
        plt.grid(True)
        plt.savefig("plots/accuracy.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(epochs, loss, marker="o")
        plt.title("Train Loss por época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("plots/loss.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(epochs, time_, marker="o")
        plt.title("Tiempo por época")
        plt.xlabel("Época")
        plt.ylabel("Segundos")
        plt.grid(True)
        plt.savefig("plots/time.png", bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(epochs, ram, marker="o")
        plt.title("Uso de RAM por época")
        plt.xlabel("Época")
        plt.ylabel("RAM usada (GB)")
        plt.grid(True)
        plt.savefig("plots/ram.png", bbox_inches="tight")
        plt.close()

        print("[PS] Gráficas guardadas en la carpeta plots/")

    def save_csv(self):
        if not self.history:
            print("[PS] No hay historial para guardar.")
            return

        with open("metrics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

        print("[PS] Métricas guardadas en metrics.csv")

    # ---------------------------
    # FINAL
    # ---------------------------

    def evaluate_and_report(self):
        for t in self._epoch_eval_threads:
            t.join()

        print("\n[PS] ENTRENAMIENTO FINALIZADO")

        if self._t0_train is not None and self._t_end_train is not None:
            total_train_time = self._t_end_train - self._t0_train
            print(f"[PS] Tiempo total entrenamiento: {total_train_time:.2f} s")

        self.plot_metrics()
        self.save_csv()

        print("[PS] Proceso terminado.")


# ---------------------------
# MAIN
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    steps_per_epoch = math.floor((50000 / args.num_workers) / args.batch_size)

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        lr=args.lr
    )

    ps.start()


if __name__ == "__main__":
    main()
