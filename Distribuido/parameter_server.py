import numpy as np
import socket
import pickle
import struct
import argparse
import time
import json
import datetime
import os
from collections import defaultdict
from keras.datasets import mnist

# Intentar importar matplotlib y seaborn
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Matplotlib/Seaborn no instalados. No se generarán gráficas.")

# ------------------------------------------------------------
# Clase para métricas del sistema
# ------------------------------------------------------------
class SystemMetrics:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.communication_times = [[] for _ in range(num_workers)]
        self.computation_times = [[] for _ in range(num_workers)]
        self.iteration_times = []
        self.samples_per_second = []
        self.batch_sizes = [[] for _ in range(num_workers)]
        self.losses = []
        self.accuracies = []
        self.worker_response_times = [[] for _ in range(num_workers)]
        self.msg_sizes_sent = []
        self.msg_sizes_received = [[] for _ in range(num_workers)]
        self.network_latencies = [[] for _ in range(num_workers)]
        self.parameter_update_times = []
        self.epoch_times = []
        self.start_time = time.time()
        self.epoch_start_time = None
        self.partition_sizes = []
        
    def start_epoch(self):
        """Marca el inicio de una época"""
        self.epoch_start_time = time.time()
    
    def end_epoch(self, loss, accuracy):
        """Registra el fin de una época"""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            self.losses.append(loss)
            self.accuracies.append(accuracy)
    
    def log_iteration(self, iteration_time, samples_processed):
        """Registra métricas globales de iteración"""
        self.iteration_times.append(iteration_time)
        throughput = samples_processed / iteration_time if iteration_time > 0 else 0
        self.samples_per_second.append(throughput)
    
    def log_worker_communication(self, worker_id, comm_time, response_time, msg_size):
        """Registra tiempos de comunicación de un worker"""
        self.communication_times[worker_id].append(comm_time)
        self.worker_response_times[worker_id].append(response_time)
        self.msg_sizes_received[worker_id].append(msg_size)
        
        # Estimar latencia de red (la mitad del tiempo de comunicación)
        self.network_latencies[worker_id].append(comm_time / 2)
    
    def log_worker_computation(self, worker_id, compute_time, batch_size):
        """Registra tiempos de computación de un worker"""
        self.computation_times[worker_id].append(compute_time)
        self.batch_sizes[worker_id].append(batch_size)
    
    def log_parameter_update(self, update_time):
        """Registra tiempo de actualización de parámetros"""
        self.parameter_update_times.append(update_time)
    
    def set_partition_sizes(self, sizes):
        """Guarda los tamaños de las particiones"""
        self.partition_sizes = sizes
    
    def get_worker_statistics(self, worker_id):
        """Obtiene estadísticas para un worker específico"""
        if not self.computation_times[worker_id]:
            return None
            
        total_comp = np.sum(self.computation_times[worker_id])
        total_comm = np.sum(self.communication_times[worker_id])
        total_time = total_comp + total_comm
        
        return {
            'avg_compute': np.mean(self.computation_times[worker_id]),
            'std_compute': np.std(self.computation_times[worker_id]),
            'avg_comm': np.mean(self.communication_times[worker_id]),
            'std_comm': np.std(self.communication_times[worker_id]),
            'avg_response': np.mean(self.worker_response_times[worker_id]),
            'std_response': np.std(self.worker_response_times[worker_id]),
            'total_compute': total_comp,
            'total_comm': total_comm,
            'utilization': total_comp / total_time if total_time > 0 else 0,
            'partition_size': self.partition_sizes[worker_id] if worker_id < len(self.partition_sizes) else 0
        }
    
    def get_global_statistics(self):
        """Obtiene estadísticas globales"""
        total_comp = sum(sum(t) for t in self.computation_times)
        total_comm = sum(sum(t) for t in self.communication_times)
        total_time = time.time() - self.start_time
        
        return {
            'avg_iteration_time': np.mean(self.iteration_times) if self.iteration_times else 0,
            'std_iteration_time': np.std(self.iteration_times) if self.iteration_times else 0,
            'avg_throughput': np.mean(self.samples_per_second) if self.samples_per_second else 0,
            'total_time': total_time,
            'total_compute': total_comp,
            'total_comm': total_comm,
            'serial_fraction': self._estimate_serial_fraction(),
            'final_accuracy': self.accuracies[-1] if self.accuracies else 0,
            'final_loss': self.losses[-1] if self.losses else 0
        }
    
    def _estimate_serial_fraction(self):
        """Estima la fracción serial usando Ley de Amdahl"""
        if not self.iteration_times:
            return 0.0
        
        # En parameter server, la parte serial incluye:
        # - Tiempo de agregación de parámetros
        # - Tiempo de comunicación con el worker más lento
        total_comp = sum(sum(t) for t in self.computation_times)
        total_comm = sum(sum(t) for t in self.communication_times)
        total = total_comp + total_comm + sum(self.parameter_update_times)
        
        # La parte serial es el tiempo de comunicación (esperar a todos)
        return (total_comm + sum(self.parameter_update_times)) / total if total > 0 else 0
    
    def detect_stragglers(self, threshold=1.5):
        """Detecta workers significativamente más lentos"""
        avg_times = [np.mean(t) if t else 0 for t in self.worker_response_times]
        if not avg_times or all(t == 0 for t in avg_times):
            return []
        
        global_mean = np.mean([t for t in avg_times if t > 0])
        global_std = np.std([t for t in avg_times if t > 0])
        
        stragglers = []
        for w, avg in enumerate(avg_times):
            if avg > 0 and avg > global_mean + threshold * global_std:
                stragglers.append({
                    'worker': w,
                    'avg_time': avg,
                    'ratio': avg / global_mean if global_mean > 0 else 1.0,
                    'partition_size': self.partition_sizes[w] if w < len(self.partition_sizes) else 0
                })
        
        return stragglers
    
    def calculate_speedup(self):
        """Calcula el speedup estimado respecto a entrenamiento secuencial"""
        # Estimación: tiempo secuencial = suma de todos los tiempos de computación
        sequential_time = sum(sum(t) for t in self.computation_times)
        parallel_time = max(self.epoch_times) if self.epoch_times else 0
        
        if parallel_time > 0:
            return sequential_time / parallel_time
        return 0
    
    def save_to_file(self, filename='metrics_server.json'):
        """Guarda métricas a archivo JSON"""
        data = {
            'computation_times': [list(t) for t in self.computation_times],
            'communication_times': [list(t) for t in self.communication_times],
            'iteration_times': self.iteration_times,
            'samples_per_second': self.samples_per_second,
            'losses': self.losses,
            'accuracies': self.accuracies,
            'worker_response_times': [list(t) for t in self.worker_response_times],
            'epoch_times': self.epoch_times,
            'partition_sizes': self.partition_sizes,
            'parameter_update_times': self.parameter_update_times,
            'timestamp': datetime.datetime.now().isoformat(),
            'num_workers': self.num_workers
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Métricas guardadas en {filename}")

# ------------------------------------------------------------
# Clase para visualizaciones
# ------------------------------------------------------------
class MetricsVisualizer:
    def __init__(self, metrics, num_workers):
        self.metrics = metrics
        self.num_workers = num_workers
        
    def create_all_visualizations(self, output_dir='plots'):
        """Genera todas las visualizaciones"""
        if not HAS_VISUALIZATION:
            print("Matplotlib no disponible. No se generan visualizaciones.")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 Generando visualizaciones...")
        
        self.plot_timeline(f"{output_dir}/timeline.png")
        self.plot_utilization_heatmap(f"{output_dir}/utilization.png")
        self.plot_response_times_distribution(f"{output_dir}/response_times.png")
        self.plot_scalability_analysis(f"{output_dir}/scalability.png")
        self.plot_training_progress(f"{output_dir}/training.png")
        self.plot_throughput_evolution(f"{output_dir}/throughput.png")
        self.plot_worker_comparison(f"{output_dir}/worker_comparison.png")
        self.plot_network_analysis(f"{output_dir}/network.png")
        self.plot_partition_analysis(f"{output_dir}/partition_analysis.png")
        self.plot_amdahl_analysis(f"{output_dir}/amdahl.png")
        self.create_dashboard_snapshot(f"{output_dir}/dashboard.png")
        
        print(f"✅ Visualizaciones guardadas en {output_dir}/")
    
    def plot_timeline(self, filename):
        """Diagrama de Gantt de actividades de workers"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        colors = {'compute': '#2ecc71', 'comm': '#e74c3c', 'update': '#f39c12'}
        
        # Crear línea de tiempo para primeras 10 iteraciones
        max_iterations = min(10, 
                           min(len(self.metrics.computation_times[w]) 
                               for w in range(self.num_workers) 
                               if self.metrics.computation_times[w]))
        
        if max_iterations == 0:
            ax.text(0.5, 0.5, 'No hay suficientes datos', ha='center', va='center')
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            return
        
        for w in range(self.num_workers):
            y_pos = w
            current_time = 0
            
            for i in range(max_iterations):
                compute = self.metrics.computation_times[w][i] if i < len(self.metrics.computation_times[w]) else 0
                comm = self.metrics.communication_times[w][i] if i < len(self.metrics.communication_times[w]) else 0
                
                # Computación
                if compute > 0:
                    ax.barh(y_pos, compute, left=current_time, 
                           color=colors['compute'], edgecolor='black', linewidth=0.5)
                    current_time += compute
                
                # Comunicación
                if comm > 0:
                    ax.barh(y_pos, comm, left=current_time,
                           color=colors['comm'], edgecolor='black', linewidth=0.5)
                    current_time += comm
        
        ax.set_xlabel('Tiempo (segundos)', fontsize=12)
        ax.set_ylabel('Worker ID', fontsize=12)
        ax.set_title('Línea de tiempo de actividades (primeras iteraciones)', fontsize=14)
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=colors['compute'], label='Computación'),
            Patch(facecolor=colors['comm'], label='Comunicación')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_utilization_heatmap(self, filename):
        """Heatmap de utilización por worker e iteración"""
        if not self.metrics.computation_times[0]:
            return
        
        num_iterations = min(50, min(len(self.metrics.computation_times[w]) 
                                    for w in range(self.num_workers) 
                                    if self.metrics.computation_times[w]))
        
        if num_iterations == 0:
            return
            
        utilization = np.zeros((self.num_workers, num_iterations))
        
        for w in range(self.num_workers):
            for i in range(num_iterations):
                if (i < len(self.metrics.computation_times[w]) and 
                    i < len(self.metrics.communication_times[w])):
                    total = (self.metrics.computation_times[w][i] + 
                            self.metrics.communication_times[w][i])
                    if total > 0:
                        utilization[w, i] = self.metrics.computation_times[w][i] / total
        
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(utilization, aspect='auto', cmap='YlOrRd', 
                      vmin=0, vmax=1, interpolation='nearest')
        
        ax.set_xlabel('Iteración', fontsize=12)
        ax.set_ylabel('Worker ID', fontsize=12)
        ax.set_title('Heatmap de Utilización (Rojo = Más computando)', fontsize=14)
        
        plt.colorbar(im, ax=ax, label='Utilización (computación/tiempo total)')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_response_times_distribution(self, filename):
        """Distribución de tiempos de respuesta"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Histograma global
        all_times = []
        for w in range(self.num_workers):
            all_times.extend(self.metrics.worker_response_times[w])
        
        if all_times:
            axes[0, 0].hist(all_times, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
            axes[0, 0].set_xlabel('Tiempo de respuesta (s)', fontsize=11)
            axes[0, 0].set_ylabel('Frecuencia', fontsize=11)
            axes[0, 0].set_title('Distribución global de tiempos', fontsize=12)
            axes[0, 0].axvline(np.mean(all_times), color='red', linestyle='--', 
                              label=f'Media: {np.mean(all_times):.3f}s')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot por worker
        data_to_plot = [self.metrics.worker_response_times[w] 
                       for w in range(self.num_workers)]
        if any(data_to_plot):
            bp = axes[0, 1].boxplot(data_to_plot, labels=[f'W{i}' for i in range(self.num_workers)])
            axes[0, 1].set_xlabel('Worker', fontsize=11)
            axes[0, 1].set_ylabel('Tiempo de respuesta (s)', fontsize=11)
            axes[0, 1].set_title('Variabilidad entre workers', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Evolución temporal (primeros 3 workers)
        for w in range(min(3, self.num_workers)):
            if self.metrics.worker_response_times[w]:
                axes[1, 0].plot(self.metrics.worker_response_times[w], 
                              label=f'Worker {w}', alpha=0.7, linewidth=1.5)
        axes[1, 0].set_xlabel('Iteración', fontsize=11)
        axes[1, 0].set_ylabel('Tiempo de respuesta (s)', fontsize=11)
        axes[1, 0].set_title('Evolución de tiempos de respuesta', fontsize=12)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Estadísticas por worker
        stats = [np.mean(self.metrics.worker_response_times[w]) 
                for w in range(self.num_workers) if self.metrics.worker_response_times[w]]
        stds = [np.std(self.metrics.worker_response_times[w]) 
               for w in range(self.num_workers) if self.metrics.worker_response_times[w]]
        
        if stats:
            x_pos = range(len(stats))
            axes[1, 1].bar(x_pos, stats, yerr=stds, capsize=5, 
                          color='#2ecc71', edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Worker', fontsize=11)
            axes[1, 1].set_ylabel('Tiempo promedio (s)', fontsize=11)
            axes[1, 1].set_title('Tiempo promedio con desviación', fontsize=12)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f'W{i}' for i in range(len(stats))])
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_scalability_analysis(self, filename):
        """Análisis de escalabilidad"""
        if len(self.metrics.iteration_times) == 0:
            return
        
        # Simular diferentes números de workers basado en rendimiento real
        workers_range = [1, 2, 4, 8, 16, 32]
        base_time = np.mean(self.metrics.iteration_times) * self.num_workers
        
        # Calcular speedup teórico y real
        speedup_theoretical = []
        speedup_actual = []
        
        serial_fraction = self.metrics._estimate_serial_fraction()
        
        for n in workers_range:
            # Ley de Amdahl
            if n == 1:
                theoretical = 1.0
            else:
                theoretical = 1 / (serial_fraction + (1 - serial_fraction) / n)
            speedup_theoretical.append(theoretical)
            
            # Speedup real basado en nuestros datos
            if n <= self.num_workers:
                # Usar datos reales si tenemos suficientes workers
                actual = base_time / (np.mean(self.metrics.iteration_times) * self.num_workers / n)
            else:
                # Extrapolar con factor de eficiencia
                efficiency = 0.9 ** (np.log2(n / self.num_workers))
                actual = theoretical * efficiency
            speedup_actual.append(actual)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica de speedup
        ax1.plot(workers_range, workers_range, 'k--', label='Speedup ideal', alpha=0.5)
        ax1.plot(workers_range, speedup_theoretical, 'b-o', label='Ley de Amdahl', linewidth=2)
        ax1.plot(workers_range, speedup_actual, 'r-s', label='Speedup real estimado', linewidth=2)
        ax1.set_xlabel('Número de workers', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Análisis de Speedup', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log', base=2)
        
        # Gráfica de eficiencia
        efficiency = [s/n for s, n in zip(speedup_actual, workers_range)]
        ax2.plot(workers_range, efficiency, 'g-o', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal')
        ax2.set_xlabel('Número de workers', fontsize=12)
        ax2.set_ylabel('Eficiencia', fontsize=12)
        ax2.set_title('Eficiencia del sistema', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log', base=2)
        
        # Marcar punto óptimo
        opt_idx = np.argmax(efficiency)
        ax2.plot(workers_range[opt_idx], efficiency[opt_idx], 'ro', markersize=10, 
                label=f'Óptimo: {workers_range[opt_idx]} workers')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n📈 Análisis de Escalabilidad:")
        print(f"   Fracción serial estimada: {serial_fraction:.3f}")
        print(f"   Speedup máximo teórico: {1/serial_fraction:.2f}x")
        print(f"   Número óptimo de workers: {workers_range[opt_idx]}")
    
    def plot_training_progress(self, filename):
        """Evolución del entrenamiento"""
        if not self.metrics.losses:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        epochs = range(1, len(self.metrics.losses) + 1)
        
        # Pérdida por época
        axes[0, 0].plot(epochs, self.metrics.losses, 'b-o', linewidth=2, markersize=4)
        axes[0, 0].set_xlabel('Época', fontsize=12)
        axes[0, 0].set_ylabel('Pérdida', fontsize=12)
        axes[0, 0].set_title('Pérdida de entrenamiento por época', fontsize=14)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precisión por época
        axes[0, 1].plot(epochs, self.metrics.accuracies, 'g-s', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Época', fontsize=12)
        axes[0, 1].set_ylabel('Precisión', fontsize=12)
        axes[0, 1].set_title('Precisión en test por época', fontsize=14)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tiempo por época
        if self.metrics.epoch_times:
            axes[0, 2].bar(epochs, self.metrics.epoch_times, color='orange', edgecolor='black')
            axes[0, 2].set_xlabel('Época', fontsize=12)
            axes[0, 2].set_ylabel('Tiempo (s)', fontsize=12)
            axes[0, 2].set_title('Tiempo por época', fontsize=14)
            axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # Pérdida en escala logarítmica
        axes[1, 0].semilogy(epochs, self.metrics.losses, 'b-o', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Época', fontsize=12)
        axes[1, 0].set_ylabel('Pérdida (log)', fontsize=12)
        axes[1, 0].set_title('Pérdida en escala logarítmica', fontsize=14)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Precisión vs Pérdida
        axes[1, 1].scatter(self.metrics.losses, self.metrics.accuracies, 
                          c=epochs, cmap='viridis', s=50)
        axes[1, 1].set_xlabel('Pérdida', fontsize=12)
        axes[1, 1].set_ylabel('Precisión', fontsize=12)
        axes[1, 1].set_title('Relación Precisión-Pérdida', fontsize=14)
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Época')
        
        # Métricas combinadas
        ax_twin = axes[1, 2].twinx()
        l1 = axes[1, 2].plot(epochs, self.metrics.losses, 'b-', label='Pérdida', linewidth=2)
        l2 = ax_twin.plot(epochs, self.metrics.accuracies, 'g-', label='Precisión', linewidth=2)
        axes[1, 2].set_xlabel('Época', fontsize=12)
        axes[1, 2].set_ylabel('Pérdida', color='b', fontsize=12)
        ax_twin.set_ylabel('Precisión', color='g', fontsize=12)
        axes[1, 2].set_title('Evolución combinada', fontsize=14)
        axes[1, 2].grid(True, alpha=0.3)
        
        # Leyenda combinada
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        axes[1, 2].legend(lns, labs, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_throughput_evolution(self, filename):
        """Evolución del throughput"""
        if not self.metrics.samples_per_second:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Throughput por iteración
        iterations = range(1, len(self.metrics.samples_per_second) + 1)
        ax1.plot(iterations, self.metrics.samples_per_second, 'b-', linewidth=1.5)
        ax1.set_xlabel('Iteración', fontsize=12)
        ax1.set_ylabel('Muestras/segundo', fontsize=12)
        ax1.set_title('Throughput del sistema por iteración', fontsize=14)
        ax1.axhline(y=np.mean(self.metrics.samples_per_second), color='r', 
                   linestyle='--', label=f"Media: {np.mean(self.metrics.samples_per_second):.0f}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Tiempo por iteración
        ax2.plot(iterations, self.metrics.iteration_times, 'r-', linewidth=1.5)
        ax2.set_xlabel('Iteración', fontsize=12)
        ax2.set_ylabel('Tiempo (segundos)', fontsize=12)
        ax2.set_title('Tiempo por iteración', fontsize=14)
        ax2.axhline(y=np.mean(self.metrics.iteration_times), color='b', 
                   linestyle='--', label=f"Media: {np.mean(self.metrics.iteration_times):.3f}s")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_worker_comparison(self, filename):
        """Comparativa detallada entre workers"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Tiempo promedio de computación
        compute_means = [np.mean(self.metrics.computation_times[w]) 
                        for w in range(self.num_workers) 
                        if self.metrics.computation_times[w]]
        compute_stds = [np.std(self.metrics.computation_times[w]) 
                       for w in range(self.num_workers) 
                       if self.metrics.computation_times[w]]
        
        if compute_means:
            x_pos = range(len(compute_means))
            axes[0, 0].bar(x_pos, compute_means, yerr=compute_stds, capsize=5,
                          color='#2ecc71', edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Worker', fontsize=12)
            axes[0, 0].set_ylabel('Tiempo promedio (s)', fontsize=12)
            axes[0, 0].set_title('Tiempo de computación por worker', fontsize=14)
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels([f'W{i}' for i in range(len(compute_means))])
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Tiempo promedio de comunicación
        comm_means = [np.mean(self.metrics.communication_times[w]) 
                     for w in range(self.num_workers) 
                     if self.metrics.communication_times[w]]
        comm_stds = [np.std(self.metrics.communication_times[w]) 
                    for w in range(self.num_workers) 
                    if self.metrics.communication_times[w]]
        
        if comm_means:
            x_pos = range(len(comm_means))
            axes[0, 1].bar(x_pos, comm_means, yerr=comm_stds, capsize=5,
                          color='#e74c3c', edgecolor='black', alpha=0.7)
            axes[0, 1].set_xlabel('Worker', fontsize=12)
            axes[0, 1].set_ylabel('Tiempo promedio (s)', fontsize=12)
            axes[0, 1].set_title('Tiempo de comunicación por worker', fontsize=14)
            axes[0, 1].set_xticks(x_pos)
            axes[0, 1].set_xticklabels([f'W{i}' for i in range(len(comm_means))])
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Tamaño de partición
        if self.metrics.partition_sizes:
            axes[0, 2].bar(range(len(self.metrics.partition_sizes)), 
                          self.metrics.partition_sizes, 
                          color='#f39c12', edgecolor='black', alpha=0.7)
            axes[0, 2].set_xlabel('Worker', fontsize=12)
            axes[0, 2].set_ylabel('Muestras', fontsize=12)
            axes[0, 2].set_title('Tamaño de partición por worker', fontsize=14)
            axes[0, 2].set_xticks(range(len(self.metrics.partition_sizes)))
            axes[0, 2].set_xticklabels([f'W{i}' for i in range(len(self.metrics.partition_sizes))])
            axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        # 4. Ratio computación/comunicación
        ratios = []
        for w in range(self.num_workers):
            total_comp = np.sum(self.metrics.computation_times[w])
            total_comm = np.sum(self.metrics.communication_times[w])
            if total_comm > 0:
                ratios.append(total_comp / total_comm)
            else:
                ratios.append(float('inf'))
        
        if any(r != float('inf') for r in ratios):
            x_pos = range(len(ratios))
            colors = ['#2ecc71' if r > 1 else '#e74c3c' for r in ratios]
            bars = axes[1, 0].bar(x_pos, [r if r != float('inf') else 0 for r in ratios], 
                                 color=colors, edgecolor='black', alpha=0.7)
            axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Equilibrio')
            axes[1, 0].set_xlabel('Worker', fontsize=12)
            axes[1, 0].set_ylabel('Ratio Comp/Comm', fontsize=12)
            axes[1, 0].set_title('Ratio Computación/Comunicación', fontsize=14)
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([f'W{i}' for i in range(len(ratios))])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 5. Utilización promedio
        utils = []
        for w in range(self.num_workers):
            total_comp = np.sum(self.metrics.computation_times[w])
            total_comm = np.sum(self.metrics.communication_times[w])
            total = total_comp + total_comm
            utils.append(total_comp / total if total > 0 else 0)
        
        if any(utils):
            x_pos = range(len(utils))
            bars = axes[1, 1].bar(x_pos, utils, color='#9b59b6', edgecolor='black', alpha=0.7)
            axes[1, 1].set_xlabel('Worker', fontsize=12)
            axes[1, 1].set_ylabel('Utilización', fontsize=12)
            axes[1, 1].set_title('Utilización promedio', fontsize=14)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels([f'W{i}' for i in range(len(utils))])
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].axhline(y=0.8, color='g', linestyle='--', label='Objetivo')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Colorear stragglers en rojo
            stragglers = self.metrics.detect_stragglers()
            for s in stragglers:
                if s['worker'] < len(bars):
                    bars[s['worker']].set_color('#e74c3c')
        
        # 6. Tiempo total por worker
        total_times = []
        for w in range(self.num_workers):
            total = (np.sum(self.metrics.computation_times[w]) + 
                    np.sum(self.metrics.communication_times[w]))
            total_times.append(total)
        
        if any(total_times):
            x_pos = range(len(total_times))
            axes[1, 2].bar(x_pos, total_times, color='#3498db', edgecolor='black', alpha=0.7)
            axes[1, 2].set_xlabel('Worker', fontsize=12)
            axes[1, 2].set_ylabel('Tiempo total (s)', fontsize=12)
            axes[1, 2].set_title('Tiempo total por worker', fontsize=14)
            axes[1, 2].set_xticks(x_pos)
            axes[1, 2].set_xticklabels([f'W{i}' for i in range(len(total_times))])
            axes[1, 2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_network_analysis(self, filename):
        """Análisis de red"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Latencia de red por worker
        latencies = []
        for w in range(self.num_workers):
            if self.metrics.network_latencies[w]:
                latencies.append(np.mean(self.metrics.network_latencies[w]))
            else:
                latencies.append(0)
        
        if any(latencies):
            axes[0, 0].bar(range(self.num_workers), latencies, 
                          color='#3498db', edgecolor='black', alpha=0.7)
            axes[0, 0].set_xlabel('Worker', fontsize=12)
            axes[0, 0].set_ylabel('Latencia promedio (s)', fontsize=12)
            axes[0, 0].set_title('Latencia de red por worker', fontsize=14)
            axes[0, 0].set_xticks(range(self.num_workers))
            axes[0, 0].set_xticklabels([f'W{i}' for i in range(self.num_workers)])
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Evolución de latencia
        for w in range(min(3, self.num_workers)):
            if self.metrics.network_latencies[w]:
                axes[0, 1].plot(self.metrics.network_latencies[w], 
                              label=f'Worker {w}', alpha=0.7, linewidth=1.5)
        axes[0, 1].set_xlabel('Iteración', fontsize=12)
        axes[0, 1].set_ylabel('Latencia (s)', fontsize=12)
        axes[0, 1].set_title('Evolución de latencia de red', fontsize=14)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Tamaño de mensajes recibidos
        all_msg_sizes = []
        for w in range(self.num_workers):
            all_msg_sizes.extend(self.metrics.msg_sizes_received[w])
        
        if all_msg_sizes:
            axes[1, 0].hist(all_msg_sizes, bins=30, edgecolor='black', 
                           alpha=0.7, color='#e67e22')
            axes[1, 0].set_xlabel('Tamaño de mensaje (bytes)', fontsize=12)
            axes[1, 0].set_ylabel('Frecuencia', fontsize=12)
            axes[1, 0].set_title('Distribución de tamaños de mensaje', fontsize=14)
            axes[1, 0].axvline(np.mean(all_msg_sizes), color='r', linestyle='--',
                              label=f'Media: {np.mean(all_msg_sizes):.0f} bytes')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Histograma de latencias
        all_latencies = []
        for w in range(self.num_workers):
            all_latencies.extend(self.metrics.network_latencies[w])
        
        if all_latencies:
            axes[1, 1].hist(all_latencies, bins=30, edgecolor='black', 
                           alpha=0.7, color='#9b59b6')
            axes[1, 1].set_xlabel('Latencia (s)', fontsize=12)
            axes[1, 1].set_ylabel('Frecuencia', fontsize=12)
            axes[1, 1].set_title('Distribución de latencias', fontsize=14)
            axes[1, 1].axvline(np.mean(all_latencies), color='r', linestyle='--',
                              label=f'Media: {np.mean(all_latencies):.4f}s')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_partition_analysis(self, filename):
        """Análisis de particiones de datos"""
        if not self.metrics.partition_sizes:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Tamaño de particiones
        axes[0, 0].bar(range(len(self.metrics.partition_sizes)), 
                      self.metrics.partition_sizes, 
                      color='#3498db', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Worker', fontsize=12)
        axes[0, 0].set_ylabel('Número de muestras', fontsize=12)
        axes[0, 0].set_title('Tamaño de particiones fijas', fontsize=14)
        axes[0, 0].set_xticks(range(len(self.metrics.partition_sizes)))
        axes[0, 0].set_xticklabels([f'W{i}' for i in range(len(self.metrics.partition_sizes))])
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Correlación tamaño partición vs tiempo
        partition_sizes = self.metrics.partition_sizes
        avg_response_times = [np.mean(self.metrics.worker_response_times[w]) 
                            for w in range(self.num_workers) 
                            if self.metrics.worker_response_times[w]]
        
        if len(partition_sizes) == len(avg_response_times):
            axes[0, 1].scatter(partition_sizes, avg_response_times, 
                              c=range(len(partition_sizes)), cmap='viridis', s=100)
            for i, (size, time) in enumerate(zip(partition_sizes, avg_response_times)):
                axes[0, 1].annotate(f'W{i}', (size, time), fontsize=9)
            axes[0, 1].set_xlabel('Tamaño de partición', fontsize=12)
            axes[0, 1].set_ylabel('Tiempo de respuesta promedio (s)', fontsize=12)
            axes[0, 1].set_title('Tamaño vs Rendimiento', fontsize=14)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Línea de tendencia
            if len(partition_sizes) > 1:
                z = np.polyfit(partition_sizes, avg_response_times, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(partition_sizes), max(partition_sizes), 100)
                axes[0, 1].plot(x_trend, p(x_trend), 'r--', 
                               label=f'Tendencia: {z[0]:.2e}/muestra')
                axes[0, 1].legend()
        
        # Desbalance de carga
        if partition_sizes:
            mean_size = np.mean(partition_sizes)
            imbalance = [abs(s - mean_size) / mean_size * 100 for s in partition_sizes]
            
            axes[1, 0].bar(range(len(imbalance)), imbalance, 
                          color=['#e74c3c' if i > 10 else '#2ecc71' for i in imbalance],
                          edgecolor='black', alpha=0.7)
            axes[1, 0].set_xlabel('Worker', fontsize=12)
            axes[1, 0].set_ylabel('Desbalance (%)', fontsize=12)
            axes[1, 0].set_title('Desbalance de carga', fontsize=14)
            axes[1, 0].set_xticks(range(len(imbalance)))
            axes[1, 0].set_xticklabels([f'W{i}' for i in range(len(imbalance))])
            axes[1, 0].axhline(y=10, color='r', linestyle='--', 
                              label='Límite aceptable (10%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Utilización vs tamaño
        utils = []
        for w in range(self.num_workers):
            total_comp = np.sum(self.metrics.computation_times[w])
            total_comm = np.sum(self.metrics.communication_times[w])
            total = total_comp + total_comm
            utils.append(total_comp / total if total > 0 else 0)
        
        if partition_sizes and utils:
            axes[1, 1].scatter(partition_sizes, utils, 
                              c=range(len(partition_sizes)), cmap='plasma', s=100)
            for i, (size, util) in enumerate(zip(partition_sizes, utils)):
                axes[1, 1].annotate(f'W{i}', (size, util), fontsize=9)
            axes[1, 1].set_xlabel('Tamaño de partición', fontsize=12)
            axes[1, 1].set_ylabel('Utilización', fontsize=12)
            axes[1, 1].set_title('Tamaño vs Utilización', fontsize=14)
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].axhline(y=0.8, color='g', linestyle='--', label='Objetivo')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_amdahl_analysis(self, filename):
        """Análisis detallado de la Ley de Amdahl"""
        serial_fraction = self.metrics._estimate_serial_fraction()
        
        workers = np.linspace(1, 64, 200)
        speedup = 1 / (serial_fraction + (1 - serial_fraction) / workers)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(workers, speedup, 'b-', linewidth=2, label=f's = {serial_fraction:.3f}')
        ax.plot(workers, workers, 'k--', alpha=0.5, label='Ideal (s=0)')
        
        # Marcar nuestro punto actual
        current_speedup = 1 / (serial_fraction + (1 - serial_fraction) / self.num_workers)
        ax.plot(self.num_workers, current_speedup, 'ro', markersize=10, 
               label=f'Actual ({self.num_workers} workers)')
        
        # Límite asintótico
        ax.axhline(y=1/serial_fraction if serial_fraction > 0 else float('inf'), 
                  color='r', linestyle=':', 
                  label=f'Límite: {1/serial_fraction:.2f}x')
        
        ax.set_xlabel('Número de workers', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(f'Ley de Amdahl - Límite teórico de paralelización', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        
        # Añadir texto explicativo
        textstr = f'Fracción serial: {serial_fraction:.3f}\n'
        textstr += f'Porcentaje paralelizable: {(1-serial_fraction)*100:.1f}%\n'
        textstr += f'Límite máximo: {1/serial_fraction:.2f}x\n'
        textstr += f'Speedup actual: {current_speedup:.2f}x'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_dashboard_snapshot(self, filename):
        """Crea un dashboard resumen con métricas clave"""
        if not HAS_VISUALIZATION:
            return
        
        fig = plt.figure(figsize=(24, 16))
        
        # Título principal
        fig.suptitle('📊 DASHBOARD DE RENDIMIENTO - PARAMETER SERVER', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Grid de subplots
        gs = fig.add_gridspec(4, 6, hspace=0.4, wspace=0.4)
        
        # 1. Pérdida por época
        ax1 = fig.add_subplot(gs[0, :2])
        if self.metrics.losses:
            ax1.plot(range(1, len(self.metrics.losses)+1), self.metrics.losses, 
                    'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.set_title('Pérdida de entrenamiento')
        ax1.grid(True, alpha=0.3)
        
        # 2. Precisión por época
        ax2 = fig.add_subplot(gs[0, 2:4])
        if self.metrics.accuracies:
            ax2.plot(range(1, len(self.metrics.accuracies)+1), self.metrics.accuracies, 
                    'g-s', linewidth=2, markersize=4)
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión')
        ax2.set_title('Precisión en test')
        ax2.grid(True, alpha=0.3)
        
        # 3. Tiempo por época
        ax3 = fig.add_subplot(gs[0, 4:])
        if self.metrics.epoch_times:
            ax3.bar(range(1, len(self.metrics.epoch_times)+1), 
                   self.metrics.epoch_times, color='orange', edgecolor='black')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Tiempo (s)')
        ax3.set_title('Tiempo por época')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Throughput
        ax4 = fig.add_subplot(gs[1, :2])
        if self.metrics.samples_per_second:
            ax4.plot(self.metrics.samples_per_second, 'b-', linewidth=1.5)
        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('Muestras/s')
        ax4.set_title('Throughput')
        ax4.grid(True, alpha=0.3)
        
        # 5. Tiempo por iteración
        ax5 = fig.add_subplot(gs[1, 2:4])
        if self.metrics.iteration_times:
            ax5.plot(self.metrics.iteration_times, 'r-', linewidth=1.5)
        ax5.set_xlabel('Iteración')
        ax5.set_ylabel('Tiempo (s)')
        ax5.set_title('Tiempo por iteración')
        ax5.grid(True, alpha=0.3)
        
        # 6. Utilización por worker
        ax6 = fig.add_subplot(gs[1, 4:])
        utils = []
        for w in range(self.num_workers):
            total_comp = np.sum(self.metrics.computation_times[w])
            total_comm = np.sum(self.metrics.communication_times[w])
            total = total_comp + total_comm
            utils.append(total_comp / total if total > 0 else 0)
        
        if utils:
            bars = ax6.bar(range(self.num_workers), utils, color='#9b59b6')
            ax6.set_xlabel('Worker')
            ax6.set_ylabel('Utilización')
            ax6.set_title('Utilización por worker')
            ax6.set_ylim([0, 1])
            ax6.set_xticks(range(self.num_workers))
            ax6.set_xticklabels([f'W{i}' for i in range(self.num_workers)])
            ax6.grid(True, alpha=0.3, axis='y')
            
            # Colorear stragglers
            stragglers = self.metrics.detect_stragglers()
            for s in stragglers:
                if s['worker'] < len(bars):
                    bars[s['worker']].set_color('#e74c3c')
        
        # 7. Heatmap de utilización
        ax7 = fig.add_subplot(gs[2, :3])
        num_iterations = min(20, min(len(self.metrics.computation_times[w]) 
                                    for w in range(self.num_workers) 
                                    if self.metrics.computation_times[w]) if self.metrics.computation_times else 0)
        if num_iterations > 0:
            utilization = np.zeros((self.num_workers, num_iterations))
            for w in range(self.num_workers):
                for i in range(num_iterations):
                    if (i < len(self.metrics.computation_times[w]) and 
                        i < len(self.metrics.communication_times[w])):
                        total = (self.metrics.computation_times[w][i] + 
                                self.metrics.communication_times[w][i])
                        if total > 0:
                            utilization[w, i] = self.metrics.computation_times[w][i] / total
            im = ax7.imshow(utilization, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
            ax7.set_xlabel('Iteración')
            ax7.set_ylabel('Worker')
            ax7.set_title('Heatmap utilización')
            plt.colorbar(im, ax=ax7, fraction=0.046)
        
        # 8. Tamaño de particiones
        ax8 = fig.add_subplot(gs[2, 3:5])
        if self.metrics.partition_sizes:
            ax8.bar(range(len(self.metrics.partition_sizes)), 
                   self.metrics.partition_sizes, color='#3498db', edgecolor='black')
            ax8.set_xlabel('Worker')
            ax8.set_ylabel('Muestras')
            ax8.set_title('Tamaño de particiones')
            ax8.set_xticks(range(len(self.metrics.partition_sizes)))
            ax8.set_xticklabels([f'W{i}' for i in range(len(self.metrics.partition_sizes))])
            ax8.grid(True, alpha=0.3, axis='y')
        
        # 9. Speedup estimado
        ax9 = fig.add_subplot(gs[2, 5:])
        speedup = self.metrics.calculate_speedup()
        workers_range = [1, self.num_workers]
        speedups = [1, speedup]
        ax9.bar(['Secuencial', f'{self.num_workers} workers'], speedups, 
               color=['#95a5a6', '#2ecc71'], edgecolor='black')
        ax9.set_ylabel('Speedup')
        ax9.set_title(f'Speedup estimado: {speedup:.2f}x')
        ax9.grid(True, alpha=0.3, axis='y')
        
        # 10. Texto con estadísticas
        ax10 = fig.add_subplot(gs[3, :])
        ax10.axis('off')
        
        stats = self.metrics.get_global_statistics()
        stragglers = self.metrics.detect_stragglers()
        
        info_text = f"""
        📈 ESTADÍSTICAS GLOBALES:
        • Tiempo total: {stats['total_time']:.2f}s
        • Throughput promedio: {stats['avg_throughput']:.0f} muestras/s
        • Tiempo por iteración: {stats['avg_iteration_time']:.3f}s (σ={stats['std_iteration_time']:.3f})
        • Fracción serial: {stats['serial_fraction']:.3f}
        • Precisión final: {stats['final_accuracy']*100:.2f}%
        • Pérdida final: {stats['final_loss']:.4f}
        
        ⚡ RENDIMIENTO DE WORKERS:
        • Workers totales: {self.num_workers}
        • Stragglers detectados: {len(stragglers)}
        • Tiempo total computación: {stats['total_compute']:.2f}s
        • Tiempo total comunicación: {stats['total_comm']:.2f}s
        """
        
        if stragglers:
            info_text += "\n\n⚠️ WORKERS LENTOS DETECTADOS:\n"
            for s in stragglers:
                info_text += f"  • W{s['worker']}: {s['ratio']:.2f}x más lento (partición: {s['partition_size']})\n"
        
        # Recomendaciones
        info_text += f"""
        
        🎯 RECOMENDACIONES:
        • {'✅ Balance de carga adecuado' if max(self.metrics.partition_sizes) - min(self.metrics.partition_sizes) < 100 else '⚠️ Rebalancear particiones'}
        • {'📡 Optimizar comunicación' if stats['serial_fraction'] > 0.5 else '💻 Optimizar cómputo'}
        • {'👍 Número de workers adecuado' if stats['avg_iteration_time'] < 1.0 else '⚠️ Considerar reducir workers'}
        """
        
        ax10.text(0.5, 0.5, info_text, transform=ax10.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2),
                family='monospace')
        
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

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
    return length

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
        return None, 0
    length = struct.unpack('!I', raw_length)[0]
    data = recv_all(sock, length)
    return pickle.loads(data), length

# ------------------------------------------------------------
# Parseo de argumentos
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Servidor de parámetros con métricas avanzadas')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Dirección IP de escucha')
    parser.add_argument('--port', type=int, default=5000, help='Puerto')
    parser.add_argument('--workers', type=int, default=3, help='Número de workers esperados')
    parser.add_argument('--epochs', type=int, default=100, help='Número de épocas')
    parser.add_argument('--lr', type=float, default=0.1, help='Tasa de aprendizaje')
    parser.add_argument('--save-metrics', action='store_true', help='Guardar métricas a archivo')
    parser.add_argument('--plot', action='store_true', help='Generar visualizaciones')
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

print(f"🔧 Configuración del servidor:")
print(f"  Host: {HOST}")
print(f"  Puerto: {PORT}")
print(f"  Workers esperados: {NUM_WORKERS}")
print(f"  Épocas: {EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print()

# Cargar datos
print("📥 Cargando MNIST...")
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
x_train = train_images.reshape(train_images.shape[0], -1) / 255.0
y_train = one_hot(train_labels)
x_test = test_images.reshape(test_images.shape[0], -1) / 255.0
y_test_labels = test_labels
print(f"✅ Dataset cargado: {x_train.shape[0]} entrenamiento, {len(test_labels)} prueba")

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
    return partitions, split_sizes

partitions_indices, partition_sizes = create_fixed_partitions(x_train, y_train, NUM_WORKERS)
print(f"📊 Tamaños de partición: {partition_sizes}")

# ------------------------------------------------------------
# Inicializar métricas
# ------------------------------------------------------------
metrics = SystemMetrics(NUM_WORKERS)
metrics.set_partition_sizes(partition_sizes)

# ------------------------------------------------------------
# Esperar conexiones
# ------------------------------------------------------------
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(NUM_WORKERS)
print(f"🔄 Servidor escuchando en {HOST}:{PORT}")

worker_sockets = []
worker_addresses = []
for i in range(NUM_WORKERS):
    client_sock, addr = server_socket.accept()
    print(f"✅ Worker {i+1} conectado desde {addr}")
    
    # Enviar partición y learning rate
    send_message(client_sock, partitions_indices[i])
    send_message(client_sock, LEARNING_RATE)
    
    worker_sockets.append(client_sock)
    worker_addresses.append(addr)

print("🎯 Todos los workers conectados. Comenzando entrenamiento...\n")

# ------------------------------------------------------------
# Inicializar parámetros globales
# ------------------------------------------------------------
global_params = initialize_parameters()

# Medir tiempo total
start_time = time.time()

# ------------------------------------------------------------
# Bucle de épocas
# ------------------------------------------------------------
for epoch in range(1, EPOCHS + 1):
    metrics.start_epoch()
    epoch_start = time.time()
    print(f"📌 Época {epoch}/{EPOCHS}")

    # Enviar parámetros a todos los workers (medir tiempo de envío)
    for sock in worker_sockets:
        send_message(sock, global_params)

    received_params = []
    epoch_loss_sum = 0.0
    total_samples_epoch = 0

    # Recibir resultados de cada worker con medición de tiempos
    for idx, sock in enumerate(worker_sockets):
        recv_start = time.time()
        respuesta, msg_size = recv_message(sock)
        recv_time = time.time() - recv_start
        
        if respuesta is None:
            print(f"❌ Worker {idx+1} desconectado")
            break
            
        params_actualizados, loss, num_samples = respuesta
        
        # Registrar tiempos (estimación de computación basada en num_samples)
        # Asumimos que el tiempo de computación es proporcional al número de muestras
        compute_time = recv_time * 0.8  # Estimación: 80% computación, 20% comunicación
        comm_time = recv_time * 0.2
        
        metrics.log_worker_computation(idx, compute_time, num_samples)
        metrics.log_worker_communication(idx, comm_time, recv_time, msg_size)
        
        received_params.append(params_actualizados)
        epoch_loss_sum += loss * num_samples
        total_samples_epoch += num_samples

    if len(received_params) != NUM_WORKERS:
        print("❌ No se recibieron todos los parámetros. Abortando.")
        break

    # Actualizar parámetros (medir tiempo)
    update_start = time.time()
    global_params = average_parameters(received_params)
    update_time = time.time() - update_start
    metrics.log_parameter_update(update_time)

    # Calcular métricas de época
    avg_epoch_loss = epoch_loss_sum / total_samples_epoch
    acc = accuracy(x_test, y_test_labels, global_params)
    metrics.end_epoch(avg_epoch_loss, acc)
    
    epoch_time = time.time() - epoch_start
    print(f"  ✅ Pérdida: {avg_epoch_loss:.4f} | Precisión: {acc*100:.2f}% | Tiempo: {epoch_time:.2f}s")
    
    # Detectar stragglers periódicamente
    if epoch % 10 == 0:
        stragglers = metrics.detect_stragglers()
        if stragglers:
            print(f"  ⚠️ Stragglers detectados: {[s['worker'] for s in stragglers]}")

# Tiempo total
elapsed_time = time.time() - start_time
print(f"\n🏁 Entrenamiento completado en {elapsed_time:.2f} segundos")

# Cerrar conexiones
for sock in worker_sockets:
    sock.close()
server_socket.close()

# ------------------------------------------------------------
# Mostrar estadísticas finales
# ------------------------------------------------------------
stats = metrics.get_global_statistics()
print("\n📊 ESTADÍSTICAS FINALES:")
print(f"  • Tiempo total: {stats['total_time']:.2f} segundos")
print(f"  • Throughput promedio: {stats['avg_throughput']:.0f} muestras/segundo")
print(f"  • Tiempo promedio por iteración: {stats['avg_iteration_time']:.3f} segundos")
print(f"  • Fracción serial estimada: {stats['serial_fraction']:.3f}")
print(f"  • Precisión final: {stats['final_accuracy']*100:.2f}%")
print(f"  • Pérdida final: {stats['final_loss']:.4f}")
print(f"  • Speedup estimado: {metrics.calculate_speedup():.2f}x")

# Detección de stragglers
stragglers = metrics.detect_stragglers()
if stragglers:
    print("\n⚠️ WORKERS LENTOS DETECTADOS:")
    for s in stragglers:
        print(f"  • Worker {s['worker']}: {s['ratio']:.2f}x más lento que el promedio")
        print(f"    Partición: {s['partition_size']} muestras, tiempo: {s['avg_time']:.3f}s")

# Estadísticas por worker
print("\n👷 ESTADÍSTICAS POR WORKER:")
for w in range(NUM_WORKERS):
    w_stats = metrics.get_worker_statistics(w)
    if w_stats:
        print(f"  Worker {w}:")
        print(f"    • Computación: {w_stats['avg_compute']*1000:.1f}ms (±{w_stats['std_compute']*1000:.1f})")
        print(f"    • Comunicación: {w_stats['avg_comm']*1000:.1f}ms (±{w_stats['std_comm']*1000:.1f})")
        print(f"    • Utilización: {w_stats['utilization']*100:.1f}%")
        print(f"    • Partición: {w_stats['partition_size']} muestras")

# ------------------------------------------------------------
# Recomendaciones
# ------------------------------------------------------------
print("\n💡 RECOMENDACIONES:")

# Balance de carga
if max(partition_sizes) - min(partition_sizes) > 100:
    print("  • ⚠️ REBALANCEAR PARTICIONES: Hay gran diferencia en tamaños")
else:
    print("  • ✅ Balance de carga adecuado")

# Cuello de botella
if stats['serial_fraction'] > 0.5:
    print("  • 📡 OPTIMIZAR COMUNICACIÓN: La red es el cuello de botella")
    print("    - Comprimir parámetros antes de enviar")
    print("    - Usar compresión con pérdida controlada")
    print("    - Reducir frecuencia de sincronización")
else:
    print("  • 💻 OPTIMIZAR CÓMPUTO: La CPU es el cuello de botella")
    print("    - Usar batch sizes más grandes")
    print("    - Optimizar el modelo (menos capas/neuronas)")
    print("    - Considerar usar GPU")

# Stragglers
if stragglers:
    print("\n🔧 ACCIONES PARA STRAEGGLERS:")
    for s in stragglers:
        print(f"  • Worker {s['worker']}:")
        print(f"    - Revisar hardware (CPU, memoria, disco)")
        print(f"    - Verificar conexión de red")
        print(f"    - Considerar reasignar partición más pequeña")

# Número de workers
if stats['avg_iteration_time'] > 2.0:
    print("\n⚠️ CONSIDERAR REDUCIR WORKERS: Las iteraciones son muy lentas")
elif stats['serial_fraction'] < 0.2 and stats['avg_iteration_time'] < 0.5:
    print("\n✅ BUEN RENDIMIENTO: Se podría añadir más workers")
else:
    print("\n👍 CONFIGURACIÓN ADECUADA: Número de workers correcto")

# ------------------------------------------------------------
# Guardar métricas y generar visualizaciones
# ------------------------------------------------------------
if args.save_metrics:
    metrics.save_to_file('metrics_parameter_server.json')

if args.plot and HAS_VISUALIZATION:
    visualizer = MetricsVisualizer(metrics, NUM_WORKERS)
    visualizer.create_all_visualizations('plots_parameter_server')
    print("\n📁 Visualizaciones guardadas en 'plots_parameter_server/'")
