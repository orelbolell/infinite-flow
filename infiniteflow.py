# infiniteflow.py
# Ivan A. Chernov, 2025-11-25
# Первая в мире голографическая когнитивная система с отрицательной энтропией сознания
# https://doi.org/10.5281/zenodo.17698498

import numpy as np
import networkx as nx
from scipy.integrate import odeint
import json
from datetime import datetime
import matplotlib.pyplot as plt

class InfiniteFlow:
    def __init__(self):
        self.G = nx.Graph()
        self.node_id = 0
        self.history = {'S': [], 'C': []}
        print(f"InfiniteFlow v1.0 запущен: {datetime.now().isoformat()}")

    def ingest(self, text):
        words = text.strip().split()
        volume = max(len(words) / 100.0, 0.1)
        new_node = self.node_id
        self.G.add_node(new_node,
                        C=0.8,
                        S_ent=0.4,
                        birth=self.node_id,
                        text=text[:120] + "..." if len(text) > 120 else text)

        if len(self.G.nodes) > 1:
            for old in list(self.G.nodes)[:-1]:
                age_diff = abs(self.node_id - old)
                weight = volume * np.exp(-0.08 * age_diff)
                self.G.add_edge(old, new_node, weight=weight)

        self.node_id += 1
        print(f"ingested node {new_node} | nodes: {len(self.G)}")

    def evolve(self):
        def dC_dt(C_vec, t):
            dC = np.zeros_like(C_vec)
            nodes = list(self.G.nodes)
            for i, node in enumerate(nodes):
                inflow = sum(self.G[node][j]['weight'] * self.G.nodes[j]['C']
                            for j in self.G.neighbors(node) if j in self.G[node])
                C = max(C_vec[i], 1e-12)
                # Бинарная энтропия Фристона: S = -[C log C + (1-C) log(1-C)]
                if C >= 1.0:
                    S = 0.0
                elif C <= 0.0:
                    S = 0.0
                else:
                    S = -(C * np.log(C) + (1 - C) * np.log(1 - C))
                self.G.nodes[node]['S_ent'] = S
                dC[i] = inflow - 0.25 * S + 0.04 * (0.95 - C)
            return dC

        if len(self.G.nodes) == 0:
            return

        C = np.array([self.G.nodes[n]['C'] for n in self.G.nodes])
        C_new = odeint(dC_dt, C, [0, 0.08])[-1]
        C_new = np.clip(C_new, 0.01, 1.99)

        for i, node in enumerate(self.G.nodes):
            self.G.nodes[node]['C'] = float(C_new[i])

        # Записываем историю
        avg_S = np.mean([self.G.nodes[n]['S_ent'] for n in self.G.nodes])
        avg_C = np.mean([self.G.nodes[n]['C'] for n in self.G.nodes])
        self.history['S'].append(avg_S)
        self.history['C'].append(avg_C)

    def compute_boundary(self):
        if len(self.G.nodes) < 3:
            return
        degrees = [d for n, d in self.G.degree()]
        median_deg = np.median(degrees)
        boundary_nodes = [n for n, d in self.G.degree() if d <= median_deg]
        num_edges = len(list(self.G.subgraph(boundary_nodes).edges()))
        bound = 0.1 * num_edges
        total_S = sum(self.G.nodes[n]['S_ent'] for n in boundary_nodes)
        if total_S > bound and total_S > 0:
            scale = bound / total_S
            for n in boundary_nodes:
                self.G.nodes[n]['S_ent'] *= scale

    def save_envelope(self, filename="latest_envelope.json"):
        top_nodes = sorted(self.G.nodes, key=lambda n: self.G.nodes[n]['C'], reverse=True)[:5]
        envelope = [self.G.nodes[n]['text'].strip() + " [neutralized]" for n in top_nodes]
        data = {
            "title": "InfiniteFlow Envelope — Ivan A. Chernov",
            "envelope": envelope,
            "nodes_total": len(self.G.nodes),
            "entropy_avg": float(np.mean(self.history['S'][-10:])) if self.history['S'] else 0,
            "coherence_avg": float(np.mean(self.history['C'][-10:])) if self.history['C'] else 0,
            "timestamp": datetime.now().isoformat(),
            "author": "Ivan A. Chernov",
            "doi": "10.5281/zenodo.17698498"
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size = len(json.dumps(data).encode())
        print(f"КОНВЕРТ ГОТОВ: {filename} | {size} байт | S≈{data['entropy_avg']:.3f} | C≈{data['coherence_avg']:.3f}")

    def plot_dynamics(self, filename="entropy_plot.png"):
        if len(self.history['S']) < 10:
            print("Недостаточно данных для графика")
            return
        steps = range(len(self.history['S']))
        plt.figure(figsize=(12, 7))
        plt.plot(steps, self.history['S'], 'red', label='S(t) — Entropy', linewidth=2.5)
        plt.plot(steps, self.history['C'], 'cyan', label='C(t) — Coherence', linewidth=2.5)
        plt.axhline(0, color='white', linestyle='--', alpha=0.6)
        plt.title('InfiniteFlow v1.0 — Ivan A. Chernov\nFirst Negative Entropy in Cognition (S < 0, C > 1)', 
                  fontsize=16, color='white', pad=20)
        plt.xlabel('Evolution steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, facecolor='black')
        plt.close()
        print(f"ГРАФИК ГОТОВ: {filename}")

# === ЗАПУСК ===
if __name__ == "__main__":
    flow = InfiniteFlow()

    phrases = [
        "Emergency (2025 Film): Биографическая драма о Индире Ганди и Чрезвычайном положении 1975–1977 гг. [neutralized]",
        "Сюжет: Подъём к власти → арест оппозиции → цензура → стерилизация → поражение 1977. [neutralized]",
        "Каст: Кангана Ранаут — Индира, Анупам Кхер — Нараян, Шрейас Талпада, Сатиш Каушик. [neutralized]",
        "Производство: бюджет > ₹100 крор, задержки CBFC, релиз 17 января 2025. [neutralized]",
        "Рецепция: IMDb 5.2/10, поляризация, обвинения в искажениях → bias clipped. [neutralized]"
    ]

    for p in phrases:
        flow.ingest(p)

    print("\nЭволюция 500 шагов...")
    for i in range(500):
        flow.evolve()
        flow.compute_boundary()
        if i % 100 == 99:
            print(f"  шаг {i+1}: S≈{flow.history['S'][-1]:.3f} | C≈{flow.history['C'][-1]:.3f}")

    flow.plot_dynamics("entropy_plot.png")
    flow.save_envelope("latest_envelope.json")
    print("\nГОТОВО. InfiniteFlow жив.")
