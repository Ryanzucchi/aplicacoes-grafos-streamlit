import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
import io
import heapq
from community import community_louvain # Para detecção de comunidades
import scipy
import os
from collections import deque
import random
from matplotlib.patches import ArrowStyle
import pandas as pd # Importar pandas para exibir dataframes de centralidade

# Configuração da página
st.set_page_config(page_title="Aplicações de Grafos", layout="wide")

# Cores personalizadas para diferentes tipos de grafos
GRAPH_COLORS = {
    "GPS": {
        "node": "#4CAF50",
        "edge": "#A5D6A7",
        "path_node": "#FF5722",
        "path_edge": "#FF9800",
        "visited": "#2196F3",
        "exploring": "#9C27B0",
        "arrow": "#FF0000"
    },
    "Communication": {
        "node": "#3F51B5",
        "edge": "#9FA8DA",
        "path_node": "#FF5722",
        "path_edge": "#FF9800",
        "visited": "#00BCD4",
        "exploring": "#E91E63",
        "arrow": "#FF0000"
    },
    "Social": {
        "node": "#9C27B0",
        "edge": "#CE93D8",
        "path_node": "#FF5722",
        "path_edge": "#FF9800",
        "visited": "#FFC107",
        "exploring": "#4CAF50",
        "arrow": "#FF0000"
    },
    "Finance": {
        "node": "#FF5722",
        "edge": "#FFCCBC",
        "path_node": "#9C27B0",
        "path_edge": "#7B1FA2",
        "visited": "#4CAF50",
        "exploring": "#2196F3",
        "arrow": "#FF0000"
    },
    "Random": { # Cores para grafos aleatórios genéricos
        "node": "#607D8B",
        "edge": "#B0BEC5",
        "path_node": "#FF5722",
        "path_edge": "#FF9800",
        "visited": "#2196F3",
        "exploring": "#9C27B0",
        "arrow": "#FF0000"
    }
}

# --- Funções de Geração de Grafos ---

def generate_random_graph(num_nodes, num_edges, is_directed=False, min_weight=1, max_weight=10):
    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Adicionar nós
    nodes = [f"N{i+1}" for i in range(num_nodes)]
    G.add_nodes_from(nodes)

    # Adicionar arestas aleatórias
    possible_edges = []
    if is_directed:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    possible_edges.append((nodes[i], nodes[j]))
    else:
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                possible_edges.append((nodes[i], nodes[j]))
    
    # Garantir que num_edges não exceda o número máximo de arestas possíveis
    num_edges = min(num_edges, len(possible_edges))
    
    # Selecionar arestas únicas aleatoriamente
    selected_edges = random.sample(possible_edges, num_edges)

    for u, v in selected_edges:
        weight = random.randint(min_weight, max_weight)
        G.add_edge(u, v, weight=weight)
        if not is_directed and not G.has_edge(v, u): # Para grafos não direcionados, NetworkX já lida com isso, mas explicitamente para garantir
             G.add_edge(v, u, weight=weight) # Adiciona a aresta inversa com o mesmo peso para consistência

    return G

# --- Funções de Plotagem e Animação ---

def plot_graph(G, pos=None, edge_labels=None, graph_type="Random", highlight_path=None, communities=None, current_edge=None, title="Grafo"):
    plt.figure(figsize=(8, 6))
    pos = pos or nx.spring_layout(G, seed=42) # Usar seed para layout reproduzível
    colors = GRAPH_COLORS.get(graph_type, GRAPH_COLORS["Random"]) # Pega cores do tipo ou default

    node_colors = []
    if highlight_path:
        node_colors = [colors["path_node"] if node in highlight_path else colors["node"] for node in G.nodes()]
        
        path_edges_set = set()
        if G.is_directed():
            for i in range(len(highlight_path) - 1):
                path_edges_set.add((highlight_path[i], highlight_path[i+1]))
        else: # Para grafos não direcionados
            for i in range(len(highlight_path) - 1):
                path_edges_set.add(tuple(sorted((highlight_path[i], highlight_path[i+1]))))

        edge_colors = [colors["path_edge"] if (G.is_directed() and (u, v) in path_edges_set) or \
                                              (not G.is_directed() and tuple(sorted((u,v))) in path_edges_set) else colors["edge"]
                       for u, v in G.edges()]
    elif communities:
        node_colors = [communities[node] for node in G.nodes()]
        edge_colors = [colors["edge"] for _ in G.edges()]
    else:
        node_colors = [colors["node"] for _ in G.nodes()]
        edge_colors = [colors["edge"] for _ in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_size=800, 
            node_color=node_colors if not communities else node_colors,
            cmap=plt.cm.tab20 if communities else None, # Usa um colormap para comunidades
            font_size=10, font_weight='bold', 
            edge_color=edge_colors, width=2, edgecolors='white', linewidths=1)
    
    if current_edge:
        u, v = current_edge
        start_x, start_y = pos[u]
        end_x, end_y = pos[v]
        
        vec = np.array([end_x - start_x, end_y - start_y])
        norm_vec = vec / np.linalg.norm(vec)
        
        arrow_start_pos = pos[u] + norm_vec * (0.1 * np.sqrt(800) / (2 * np.pi)) 
        arrow_end_pos = pos[v] - norm_vec * (0.1 * np.sqrt(800) / (2 * np.pi))
        
        plt.arrow(arrow_start_pos[0], arrow_start_pos[1],
                  arrow_end_pos[0] - arrow_start_pos[0],
                  arrow_end_pos[1] - arrow_start_pos[1],
                  color=colors["arrow"],
                  head_width=0.05, head_length=0.1, width=0.01, length_includes_head=True,
                  shape='full', lw=0)
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    plt.title(title, fontweight='bold')
    plt.axis('off')
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    st.pyplot(plt)

def animate_algorithm(G, pos, steps, graph_type="Random", algorithm_name="Algoritmo"):
    colors = GRAPH_COLORS.get(graph_type, GRAPH_COLORS["Random"])
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        node_status, edge_status_anim, current_nodes, current_edge = steps[frame]
        
        node_colors = []
        for node in G.nodes():
            if node in current_nodes:
                node_colors.append(colors["exploring"])
            elif node_status.get(node, False):
                node_colors.append(colors["visited"])
            else:
                node_colors.append(colors["node"])
        
        edge_colors = []
        for u, v in G.edges():
            is_highlighted = False
            if G.is_directed():
                if (u, v) in edge_status_anim:
                    is_highlighted = True
            else:
                if (u, v) in edge_status_anim or (v, u) in edge_status_anim:
                    is_highlighted = True
            
            if is_highlighted:
                edge_colors.append(colors["path_edge"])
            else:
                edge_colors.append(colors["edge"])
        
        nx.draw(G, pos, ax=ax, with_labels=True, node_size=800,
                node_color=node_colors, font_size=10, font_weight='bold',
                edge_color=edge_colors, width=2, edgecolors='white', linewidths=1)
        
        if current_edge:
            u, v = current_edge
            start_x, start_y = pos[u]
            end_x, end_y = pos[v]
            
            vec = np.array([end_x - start_x, end_y - start_y])
            norm_vec = vec / np.linalg.norm(vec)
            
            arrow_start_pos = pos[u] + norm_vec * (0.1 * np.sqrt(800) / (2 * np.pi))
            arrow_end_pos = pos[v] - norm_vec * (0.1 * np.sqrt(800) / (2 * np.pi))
            
            ax.arrow(arrow_start_pos[0], arrow_start_pos[1],
                     arrow_end_pos[0] - arrow_start_pos[0],
                     arrow_end_pos[1] - arrow_start_pos[1],
                     color=colors["arrow"],
                     head_width=0.05, head_length=0.1, width=0.01, length_includes_head=True,
                     shape='full', lw=0)
        
        ax.set_title(f"{algorithm_name} - Passo {frame + 1}/{len(steps)}", fontweight='bold')
        plt.axis('off')
        return ax
    
    anim = FuncAnimation(fig, update, frames=len(steps), interval=1000)
    plt.close()
    
    try:
        temp_file = "temp_animation.gif"
        writer = PillowWriter(fps=1)
        anim.save(temp_file, writer=writer, dpi=100)
        
        with open(temp_file, "rb") as f:
            gif_bytes = io.BytesIO(f.read())
        
        os.remove(temp_file)
        return gif_bytes
    except Exception as e:
        st.error(f"Erro ao criar animação: {str(e)}")
        update(len(steps)-1)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf

# --- Implementações dos Algoritmos de Caminho Mínimo ---

# Dijkstra (Apenas pesos positivos)
def dijkstra_with_animation(G, start, end):
    # Verificar se há pesos negativos
    for u, v, data in G.edges(data=True):
        if data.get('weight', 1) < 0:
            st.error("Dijkstra não funciona corretamente com pesos negativos. Por favor, use Bellman-Ford ou gere um grafo com pesos positivos.")
            return [], float('inf'), None, []

    pos = nx.spring_layout(G, seed=42)
    steps = []
    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous = {}
    queue = [(0, start)]
    visited = set()
    node_status = {node: False for node in G.nodes()}
    
    steps.append((node_status.copy(), set(), [start], None)) # Estado inicial

    while queue:
        current_dist, current_node = heapq.heappop(queue)
        
        if current_node in visited:
            continue
        
        visited.add(current_node)
        node_status[current_node] = True
        
        if current_node == end:
            break
            
        for neighbor in G.neighbors(current_node):
            edge_weight = G[current_node][neighbor].get('weight', 1)
            new_dist = current_dist + edge_weight
            
            # Animação: Aresta sendo considerada
            temp_edge_status = set() # Reinicia para mostrar apenas a aresta atual
            if G.is_directed(): temp_edge_status.add((current_node, neighbor))
            else: temp_edge_status.add(tuple(sorted((current_node, neighbor))))
            
            steps.append((node_status.copy(), temp_edge_status, [current_node, neighbor], (current_node, neighbor)))

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                previous[neighbor] = current_node
                heapq.heappush(queue, (new_dist, neighbor))
    
    path = []
    if end in previous:
        current = end
        while current != start:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        final_node_status = {node: True for node in path}
        final_edge_status = set()
        if G.is_directed():
            for i in range(len(path) - 1): final_edge_status.add((path[i], path[i+1]))
        else:
            for i in range(len(path) - 1): final_edge_status.add(tuple(sorted((path[i], path[i+1]))))

        steps.append((final_node_status, final_edge_status, [], None))
    else:
        steps.append((node_status.copy(), set(), [], None)) # Sem caminho
            
    return path, distances.get(end, float('inf')), pos, steps

# Bellman-Ford (Aceita pesos negativos, detecta ciclos negativos)
def bellman_ford_with_animation(G, start, end):
    pos = nx.spring_layout(G, seed=42)
    steps = []
    
    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous = {node: None for node in G.nodes()}
    node_status = {node: False for node in G.nodes()}
    
    steps.append((node_status.copy(), set(), [start], None)) # Estado inicial
    
    num_nodes = len(G.nodes())
    
    # Relaxamento das arestas V-1 vezes
    for i in range(num_nodes - 1):
        updated_in_iteration = False
        
        edges_to_process = list(G.edges(data=True))
        random.shuffle(edges_to_process) # Para variar a ordem da animação
        
        for u, v, data in edges_to_process:
            weight = data.get('weight', 1)
            
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                previous[v] = u
                updated_in_iteration = True
                
                temp_node_status = node_status.copy()
                temp_node_status[v] = True
                
                temp_edge_status = set()
                if G.is_directed(): temp_edge_status.add((u, v))
                else: temp_edge_status.add(tuple(sorted((u,v))))

                steps.append((temp_node_status, temp_edge_status, [u, v], (u, v)))
        
        if not updated_in_iteration:
            break
            
    # Verificação de ciclos negativos na V-ésima iteração
    has_negative_cycle = False
    cycle_nodes = None
    for u, v, data in G.edges(data=True):
        weight = data.get('weight', 1)
        if distances[u] != float('inf') and distances[u] + weight < distances[v]:
            has_negative_cycle = True
            
            # Tentar reconstruir o ciclo negativo para visualização
            cycle_path_trace = []
            current = v
            # Percorre um máximo de N nós para encontrar o ciclo
            for _ in range(num_nodes + 1):
                if current is None: break
                if current in cycle_path_trace:
                    start_index = cycle_path_trace.index(current)
                    cycle_nodes = cycle_path_trace[start_index:]
                    cycle_nodes.append(current) # Fecha o ciclo
                    break
                cycle_path_trace.append(current)
                current = previous.get(current)
            if cycle_nodes:
                cycle_nodes.reverse()
            break
            
    path = []
    if not has_negative_cycle and distances[end] != float('inf'):
        current = end
        while current != start:
            path.append(current)
            current = previous[current]
        path.append(start)
        path.reverse()
        
        final_node_status = {node: True for node in path}
        final_edge_status = set()
        if G.is_directed():
            for i in range(len(path) - 1): final_edge_status.add((path[i], path[i+1]))
        else:
            for i in range(len(path) - 1): final_edge_status.add(tuple(sorted((path[i], path[i+1]))))
        steps.append((final_node_status, final_edge_status, [], None))
    elif has_negative_cycle and cycle_nodes:
        # Se houver ciclo negativo, mostrar o ciclo se puder ser reconstruído
        final_node_status = {node: True for node in cycle_nodes}
        final_edge_status = set()
        if G.is_directed():
            for i in range(len(cycle_nodes) - 1): final_edge_status.add((cycle_nodes[i], cycle_nodes[i+1]))
        else: # Assumindo que o Bellman-Ford detectou em um grafo que pode ter arestas duplas para um "ciclo" efetivo
            for i in range(len(cycle_nodes) - 1): final_edge_status.add(tuple(sorted((cycle_nodes[i], cycle_nodes[i+1]))))
        
        steps.append((final_node_status, final_edge_status, [], None))
    else:
        steps.append((node_status.copy(), set(), [], None))
            
    return path, distances.get(end, float('inf')), pos, steps, has_negative_cycle

# A* (Usa heurística)
def astar_with_animation(G, start, end, heuristic_type):
    # A* não exige pesos positivos, mas a heurística deve ser admissível e consistente
    # Se houver pesos negativos, A* ainda pode funcionar, mas a heurística euclidiana
    # pode não ser consistente, levando a resultados subótimos ou a não encontrar o caminho.
    # Para simplicidade e foco na heurística, vamos manter a expectativa de pesos positivos aqui.
    # Se quiser, podemos adicionar um aviso sobre pesos negativos e A*.
    for u, v, data in G.edges(data=True):
        if data.get('weight', 1) < 0:
            st.warning("A* pode não garantir o caminho mais curto com pesos negativos, a menos que a heurística seja cuidadosamente escolhida para ser consistente. Dijkstra ou Bellman-Ford são mais robustos para esses casos.")
            break # Apenas um aviso, não impede a execução

    pos = nx.spring_layout(G, seed=42) # Usar seed para layout reproduzível
    steps = []
    
    def default_heuristic(u, v):
        if heuristic_type == "Euclidiana" and u in pos and v in pos:
            # Calcula a distância euclidiana entre as posições dos nós
            return np.sqrt((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)
        else:
            return 0 # Heurística zero, equivalente ao Dijkstra
    
    open_set = []
    heapq.heappush(open_set, (0, start)) # (f_score, node)
    
    came_from = {}
    g_score = {node: float('inf') for node in G.nodes()} # Custo real do start até o nó atual
    g_score[start] = 0
    
    f_score = {node: float('inf') for node in G.nodes()} # Custo estimado do start até o end, passando pelo nó atual
    f_score[start] = default_heuristic(start, end)
    
    open_set_hash = {start} # Para checagem rápida se um nó está no open_set
    
    node_status = {node: False for node in G.nodes()}
    
    steps.append((node_status.copy(), set(), [start], None)) # Estado inicial
    
    while open_set:
        current_f_score, current = heapq.heappop(open_set)
        
        if current in open_set_hash:
            open_set_hash.remove(current)
        
        node_status[current] = True # Marca o nó como visitado (processado)
        
        if current == end:
            break
            
        for neighbor in G.neighbors(current):
            edge_weight = G[current][neighbor].get('weight', 1)
            tentative_g_score = g_score[current] + edge_weight
            
            # Animação: Aresta sendo considerada
            temp_edge_status = set()
            if G.is_directed(): temp_edge_status.add((current, neighbor))
            else: temp_edge_status.add(tuple(sorted((current, neighbor))))

            steps.append((node_status.copy(), temp_edge_status, [current, neighbor], (current, neighbor)))

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + default_heuristic(neighbor, end)
                
                if neighbor not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    open_set_hash.add(neighbor)

    path = []
    if end in came_from:
        current = end
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        
        final_node_status = {node: True for node in path}
        final_edge_status = set()
        if G.is_directed():
            for i in range(len(path) - 1): final_edge_status.add((path[i], path[i+1]))
        else:
            for i in range(len(path) - 1): final_edge_status.add(tuple(sorted((path[i], path[i+1]))))
        steps.append((final_node_status, final_edge_status, [], None))
    else:
        steps.append((node_status.copy(), set(), [], None)) # Sem caminho
            
    return path, g_score.get(end, float('inf')), pos, steps

# --- Implementações dos Algoritmos de Busca ---

def bfs_with_animation(G, start_node):
    pos = nx.spring_layout(G, seed=42)
    steps = []
    
    visited = {node: False for node in G.nodes()}
    queue = deque([start_node])
    visited[start_node] = True
    
    node_status = {node: False for node in G.nodes()}
    visited_edges = set() # Arestas que foram usadas para visitar um nó
    
    node_status[start_node] = True
    steps.append((node_status.copy(), visited_edges.copy(), [start_node], None))
    
    while queue:
        current_node = queue.popleft()
        
        for neighbor in G.neighbors(current_node):
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
                
                # Animação: Aresta sendo considerada
                temp_edge_status = visited_edges.copy()
                if G.is_directed(): temp_edge_status.add((current_node, neighbor))
                else: temp_edge_status.add(tuple(sorted((current_node, neighbor))))

                temp_node_status = node_status.copy()
                temp_node_status[neighbor] = True
                
                steps.append((temp_node_status, temp_edge_status, [current_node, neighbor], (current_node, neighbor)))
                
                # Adiciona a aresta ao conjunto permanente de arestas visitadas
                if G.is_directed(): visited_edges.add((current_node, neighbor))
                else: visited_edges.add(tuple(sorted((current_node, neighbor))))
                
    steps.append((node_status.copy(), visited_edges.copy(), [], None))
    return pos, steps

def dfs_with_animation(G, start_node):
    pos = nx.spring_layout(G, seed=42)
    steps = []
    
    visited = {node: False for node in G.nodes()}
    stack = [start_node]
    
    node_status = {node: False for node in G.nodes()}
    visited_edges = set() # Arestas no caminho DFS
    
    node_status[start_node] = True
    steps.append((node_status.copy(), visited_edges.copy(), [start_node], None))
    
    while stack:
        current_node = stack[-1] # Peek
        
        if not visited[current_node]:
            visited[current_node] = True
            node_status[current_node] = True
            steps.append((node_status.copy(), visited_edges.copy(), [current_node], None)) # Nó visitado
            
            neighbors_to_explore = list(G.neighbors(current_node))
            random.shuffle(neighbors_to_explore) 

            found_unvisited_neighbor = False
            for neighbor in neighbors_to_explore:
                if not visited[neighbor]:
                    stack.append(neighbor)
                    
                    # Animação: Aresta sendo considerada
                    temp_edge_status = visited_edges.copy()
                    if G.is_directed(): temp_edge_status.add((current_node, neighbor))
                    else: temp_edge_status.add(tuple(sorted((current_node, neighbor))))
                    
                    steps.append((node_status.copy(), temp_edge_status, [current_node, neighbor], (current_node, neighbor)))
                    
                    if G.is_directed(): visited_edges.add((current_node, neighbor))
                    else: visited_edges.add(tuple(sorted((current_node, neighbor))))
                    
                    found_unvisited_neighbor = True
                    break
            
            if not found_unvisited_neighbor:
                stack.pop() # Backtrack
        else:
            stack.pop() # Nó já visitado, remove da pilha
            
    steps.append((node_status.copy(), visited_edges.copy(), [], None))
    return pos, steps

# --- Funções para a aba de Finanças (usando Bellman-Ford) ---

def create_finance_network():
    G = nx.DiGraph()
    currencies = ["USD", "EUR", "GBP", "JPY", "BRL", "CAD", "AUD", "CHF"]
    G.add_nodes_from(currencies)
    
    conversions = [
        ("USD", "EUR", 0.90), 
        ("EUR", "GBP", 0.85),
        ("GBP", "USD", 1.35), # Taxa de arbitragem: 0.90 * 0.85 * 1.35 = 1.03375
        
        ("USD", "JPY", 150.0), ("JPY", "BRL", 0.035), ("BRL", "USD", 0.20),
        ("EUR", "CHF", 0.98), ("CHF", "USD", 1.10), ("USD", "CAD", 1.30),
        ("CAD", "AUD", 1.05), ("AUD", "JPY", 95.0), ("JPY", "EUR", 0.006),
        ("GBP", "CHF", 1.15), ("CHF", "EUR", 1.02), ("EUR", "BRL", 5.50)
    ]
    
    for u, v, rate in conversions:
        G.add_edge(u, v, weight=-np.log(rate)) # Usamos -log(rate) para detecção de ciclo negativo
        
    return G

def detect_arbitrage(G, start):
    distances = {node: float('inf') for node in G.nodes()}
    distances[start] = 0
    previous = {node: None for node in G.nodes()}
    
    num_nodes = len(G.nodes())
    for _ in range(num_nodes - 1):
        for u, v, data in G.edges(data=True):
            if distances[u] != float('inf') and distances[u] + data['weight'] < distances[v]:
                distances[v] = distances[u] + data['weight']
                previous[v] = u
    
    arbitrage_cycle = None
    for u, v, data in G.edges(data=True):
        if distances[u] != float('inf') and distances[u] + data['weight'] < distances[v]:
            cycle_path_trace = []
            current = v
            for _ in range(num_nodes + 1):
                if current is None: break
                if current in cycle_path_trace:
                    start_index = cycle_path_trace.index(current)
                    arbitrage_cycle = cycle_path_trace[start_index:]
                    arbitrage_cycle.append(current)
                    break
                cycle_path_trace.append(current)
                current = previous.get(current)
            if arbitrage_cycle:
                arbitrage_cycle.reverse()
                return arbitrage_cycle
            
    return None

# --- Streamlit UI ---

st.sidebar.title("Aplicações de Grafos")
opcao = st.sidebar.radio(
    "Selecione a aplicação:",
    ["Navegação GPS", "Redes de Comunicação", "Redes Sociais", "Finanças (Bellman-Ford)"],
    index=0
)

# --- Aba: Navegação GPS (Dijkstra e A*) ---
if opcao == "Navegação GPS":
    st.header("🚗 Navegação GPS com Grafos", divider='rainbow')
    st.markdown("""
    ### Encontrando o Caminho Mais Curto (Dijkstra e A*)
    Nesta seção, exploramos como os grafos são usados em sistemas de navegação GPS.
    As cidades ou pontos de interesse são representados como **nós** e as estradas ou rotas como **arestas**.
    O **peso** de cada aresta pode ser a distância, o tempo de viagem ou o custo.
    
    - **Algoritmo de Dijkstra:** Encontra o caminho mais curto entre dois nós em grafos com **pesos de aresta não negativos**.
    - **Algoritmo A\*:** Uma extensão do Dijkstra que usa uma **heurística** para guiar a busca, tornando-o mais eficiente para encontrar caminhos em grandes redes.
    """)
    
    st.markdown("---")
    st.subheader("Configuração do Grafo Aleatório")
    col_nodes, col_edges = st.columns(2)
    with col_nodes:
        num_nodes_gps = st.slider("Número de Cidades (Nós)", min_value=5, max_value=20, value=10, key="num_nodes_gps")
    with col_edges:
        max_possible_edges = num_nodes_gps * (num_nodes_gps - 1) // 2
        num_edges_gps = st.slider("Número de Estradas (Arestas)", min_value=num_nodes_gps - 1, max_value=max_possible_edges, value=int(max_possible_edges * 0.4), key="num_edges_gps")
        if num_edges_gps < num_nodes_gps - 1:
            st.warning("O número de arestas deve ser no mínimo (Nós - 1) para um grafo conectado.")
            num_edges_gps = num_nodes_gps - 1

    st.info("Para Dijkstra e A*, os pesos das arestas serão gerados **positivos**.")
    min_weight_gps = st.slider("Peso Mínimo da Aresta", min_value=1, max_value=10, value=1, key="min_weight_gps")
    max_weight_gps = st.slider("Peso Máximo da Aresta", min_value=1, max_value=20, value=10, key="max_weight_gps")

    generate_graph_button_gps = st.button("Gerar Novo Mapa Aleatório", key="generate_gps_graph")
    
    # Armazenar o grafo na sessão para que ele persista
    if 'G_gps' not in st.session_state or generate_graph_button_gps:
        st.session_state.G_gps = generate_random_graph(num_nodes_gps, num_edges_gps, is_directed=False, min_weight=min_weight_gps, max_weight=max_weight_gps)
        st.session_state.pos_gps = nx.spring_layout(st.session_state.G_gps, seed=42) # Recalcula posições
    
    G_gps = st.session_state.G_gps
    pos_gps = st.session_state.pos_gps
    edge_labels_gps = nx.get_edge_attributes(G_gps, 'weight')
    
    st.subheader("Mapa da Rede Gerado (Grafo GPS)")
    plot_graph(G_gps, pos_gps, edge_labels_gps, graph_type="GPS", title="Mapa Aleatório de Cidades")
    
    st.markdown("---")
    st.subheader("Encontrar Rota")
    
    nodes_in_graph_gps = list(G_gps.nodes())
    if len(nodes_in_graph_gps) < 2:
        st.warning("Gere um grafo com pelo menos 2 nós para calcular uma rota.")
    else:
        col_start, col_end = st.columns(2)
        with col_start:
            start_node_gps = st.selectbox("Ponto de Partida", nodes_in_graph_gps, key="gps_start")
        with col_end:
            # Garante que o nó de destino seja diferente do de partida, se possível
            default_end_index = (nodes_in_graph_gps.index(start_node_gps) + 1) % len(nodes_in_graph_gps)
            end_node_gps = st.selectbox("Ponto de Chegada", nodes_in_graph_gps, index=default_end_index, key="gps_end")
            
        algorithm_choice_gps = st.radio("Escolha o algoritmo:", ["Dijkstra", "A*"], horizontal=True)
        
        heuristic_choice_gps = "Zero (Igual ao Dijkstra)"
        if algorithm_choice_gps == "A*":
            heuristic_choice_gps = st.radio("Escolha a heurística para A*:", ["Zero (Igual ao Dijkstra)", "Euclidiana (baseado na posição)"], horizontal=True)
            
        if st.button("Calcular Rota e Animar", key="gps_calculate"):
            if start_node_gps == end_node_gps:
                st.warning("Ponto de partida e chegada são os mesmos. Não há rota a calcular.")
            elif start_node_gps not in G_gps.nodes() or end_node_gps not in G_gps.nodes():
                st.error("Nó de partida ou chegada não encontrado no grafo.")
            else:
                path_gps = []
                distance_gps = float('inf')
                steps_gps = []
                
                if algorithm_choice_gps == "Dijkstra":
                    path_gps, distance_gps, pos_gps, steps_gps = dijkstra_with_animation(G_gps, start_node_gps, end_node_gps)
                elif algorithm_choice_gps == "A*":
                    path_gps, distance_gps, pos_gps, steps_gps = astar_with_animation(G_gps, start_node_gps, end_node_gps, heuristic_choice_gps)
                
                if path_gps:
                    st.success(f"Caminho mais curto encontrado: {' → '.join(path_gps)}")
                    st.info(f"Distância total: {distance_gps:.2f}")
                    
                    st.subheader("Animação da Busca da Rota")
                    gif_bytes_gps = animate_algorithm(G_gps, pos_gps, steps_gps, graph_type="GPS", algorithm_name=algorithm_choice_gps)
                    st.image(gif_bytes_gps, use_column_width=True)
                    
                    st.subheader("Rota Final Destacada")
                    plot_graph(G_gps, pos_gps, edge_labels_gps, graph_type="GPS", highlight_path=path_gps, title="Rota Mais Curta")
                else:
                    st.error("Não foi possível encontrar um caminho entre os pontos selecionados.")


# --- Aba: Redes de Comunicação (BFS e DFS) ---
elif opcao == "Redes de Comunicação":
    st.header("📡 Redes de Comunicação com Grafos", divider='rainbow')
    st.markdown("""
    ### Roteamento e Conectividade (BFS e DFS)
    Grafos são fundamentais para modelar redes de comunicação, como a internet.
    Aqui, os **nós** podem ser servidores, roteadores ou dispositivos, e as **arestas**
    representam as conexões físicas ou lógicas entre eles.
    
    - **Busca em Largura (BFS):** Ideal para encontrar o caminho mais curto em termos de número de "saltos" (conexões), ou para verificar a conectividade de todos os nós a partir de um ponto.
    - **Busca em Profundidade (DFS):** Útil para explorar um caminho o mais longe possível antes de retroceder, bom para rastrear caminhos específicos ou verificar ciclos.
    """)
    
    st.markdown("---")
    st.subheader("Configuração do Grafo Aleatório")
    col_nodes_comm, col_edges_comm, col_directed_comm = st.columns(3)
    with col_nodes_comm:
        num_nodes_comm = st.slider("Número de Dispositivos (Nós)", min_value=5, max_value=20, value=10, key="num_nodes_comm")
    with col_edges_comm:
        max_possible_edges_comm = num_nodes_comm * (num_nodes_comm - 1)
        num_edges_comm = st.slider("Número de Conexões (Arestas)", min_value=num_nodes_comm - 1, max_value=max_possible_edges_comm, value=int(max_possible_edges_comm * 0.2), key="num_edges_comm")
        if num_edges_comm < num_nodes_comm - 1:
            st.warning("O número de arestas deve ser no mínimo (Nós - 1) para um grafo conectado.")
            num_edges_comm = num_nodes_comm - 1
    with col_directed_comm:
        is_directed_comm = st.checkbox("Grafo Direcionado?", value=False, key="is_directed_comm")

    st.info("Os pesos das arestas (latência/custo) serão gerados positivos.")
    min_weight_comm = st.slider("Peso Mínimo da Conexão", min_value=1, max_value=10, value=1, key="min_weight_comm")
    max_weight_comm = st.slider("Peso Máximo da Conexão", min_value=1, max_value=20, value=10, key="max_weight_comm")

    generate_graph_button_comm = st.button("Gerar Nova Rede Aleatória", key="generate_comm_graph")

    if 'G_comm' not in st.session_state or generate_graph_button_comm:
        st.session_state.G_comm = generate_random_graph(num_nodes_comm, num_edges_comm, is_directed=is_directed_comm, min_weight=min_weight_comm, max_weight=max_weight_comm)
        st.session_state.pos_comm = nx.spring_layout(st.session_state.G_comm, seed=42)
    
    G_comm = st.session_state.G_comm
    pos_comm = st.session_state.pos_comm
    edge_labels_comm = nx.get_edge_attributes(G_comm, 'weight')
    
    st.subheader("Estrutura da Rede de Comunicação")
    plot_graph(G_comm, pos_comm, edge_labels_comm, graph_type="Communication", title="Rede de Comunicação Aleatória")
    
    st.markdown("---")
    st.subheader("Simulação de Busca na Rede")
    
    nodes_in_graph_comm = list(G_comm.nodes())
    if len(nodes_in_graph_comm) == 0:
        st.warning("Gere um grafo com pelo menos 1 nó para iniciar a busca.")
    else:
        start_node_comm = st.selectbox("Nó Inicial da Busca", nodes_in_graph_comm, key="comm_start")
        algorithm_choice_comm = st.radio("Escolha o algoritmo de busca:", ["BFS (Busca em Largura)", "DFS (Busca em Profundidade)"], horizontal=True)
        
        if st.button("Executar Busca e Animar", key="comm_execute"):
            pos_result = None
            steps_result = []
            
            if start_node_comm not in G_comm.nodes():
                st.error("Nó inicial não encontrado no grafo.")
            else:
                if algorithm_choice_comm == "BFS (Busca em Largura)":
                    pos_result, steps_result = bfs_with_animation(G_comm, start_node_comm)
                    st.success(f"Busca em Largura (BFS) a partir de '{start_node_comm}' concluída.")
                elif algorithm_choice_comm == "DFS (Busca em Profundidade)":
                    pos_result, steps_result = dfs_with_animation(G_comm, start_node_comm)
                    st.success(f"Busca em Profundidade (DFS) a partir de '{start_node_comm}' concluída.")
                
                if steps_result:
                    st.subheader("Animação da Busca na Rede")
                    gif_bytes_comm = animate_algorithm(G_comm, pos_result, steps_result, graph_type="Communication", algorithm_name=algorithm_choice_comm.split(' ')[0])
                    st.image(gif_bytes_comm, use_column_width=True)
                    
                    st.subheader("Nós Visitados (Estado Final)")
                    final_node_status = steps_result[-1][0]
                    final_edge_status_for_plot = steps_result[-1][1]
                    
                    highlighted_nodes = [node for node, visited in final_node_status.items() if visited]
                    
                    plt.figure(figsize=(8, 6))
                    node_colors_final = [GRAPH_COLORS["Communication"]["visited"] if node in highlighted_nodes else GRAPH_COLORS["Communication"]["node"] for node in G_comm.nodes()]
                    
                    edge_colors_final = []
                    for u, v in G_comm.edges():
                        edge_key = (u,v) if G_comm.is_directed() else tuple(sorted((u,v)))
                        if edge_key in final_edge_status_for_plot:
                            edge_colors_final.append(GRAPH_COLORS["Communication"]["path_edge"])
                        else:
                            edge_colors_final.append(GRAPH_COLORS["Communication"]["edge"])

                    nx.draw(G_comm, pos_result, with_labels=True, node_size=800,
                            node_color=node_colors_final,
                            edge_color=edge_colors_final,
                            width=2, edgecolors='white', linewidths=1, font_size=10, font_weight='bold')
                    
                    nx.draw_networkx_edge_labels(G_comm, pos_result, edge_labels=edge_labels_comm, font_size=8)
                    plt.title(f"Busca {algorithm_choice_comm.split(' ')[0]} a partir de {start_node_comm} (Final)", fontweight='bold')
                    plt.axis('off')
                    fig = plt.gcf()
                    fig.patch.set_facecolor('white')
                    st.pyplot(plt)
                else:
                    st.warning("Nenhum passo de animação gerado. O nó inicial pode estar isolado ou não há nós no grafo.")

# --- Aba: Redes Sociais ---
elif opcao == "Redes Sociais":
    st.header("👥 Análise de Redes Sociais", divider='rainbow')
    st.markdown("""
    ### Conectividade, Influência e Comunidades
    Grafos são a estrutura perfeita para representar redes sociais, onde **nós** são pessoas
    (ou perfis) e **arestas** são amizades, conexões ou interações.
    
    - **Centralidade:** Métricas como **Centralidade de Grau** (número de conexões),
      **Centralidade de Intermediação** (estar em muitos caminhos curtos) e
      **Centralidade de Proximidade** (quão perto dos outros nós) ajudam a identificar
      indivíduos influentes ou importantes na rede.
    - **Detecção de Comunidades:** Algoritmos como o **Louvain** permitem encontrar grupos
      de nós que estão mais densamente conectados entre si do que com o resto da rede,
      revelando "panelinhas" ou subgrupos.
    """)
    
    st.markdown("---")
    st.subheader("Configuração da Rede Social Aleatória")
    col_nodes_social, col_edges_social = st.columns(2)
    with col_nodes_social:
        num_nodes_social = st.slider("Número de Pessoas (Nós)", min_value=5, max_value=30, value=15, key="num_nodes_social")
    with col_edges_social:
        max_possible_edges_social = num_nodes_social * (num_nodes_social - 1) // 2
        num_edges_social = st.slider("Número de Conexões (Arestas)", min_value=num_nodes_social - 1, max_value=max_possible_edges_social, value=int(max_possible_edges_social * 0.3), key="num_edges_social")
        if num_edges_social < num_nodes_social - 1:
            st.warning("O número de arestas deve ser no mínimo (Nós - 1) para um grafo conectado.")
            num_edges_social = num_nodes_social - 1

    generate_graph_button_social = st.button("Gerar Nova Rede Social Aleatória", key="generate_social_graph")

    if 'G_social' not in st.session_state or generate_graph_button_social:
        # Redes sociais são geralmente não direcionadas
        st.session_state.G_social = generate_random_graph(num_nodes_social, num_edges_social, is_directed=False, min_weight=1, max_weight=1) # Pesos não importam para centralidade/comunidade
        st.session_state.pos_social = nx.spring_layout(st.session_state.G_social, seed=42)
    
    G_social = st.session_state.G_social
    pos_social = st.session_state.pos_social
    
    st.subheader("Exemplo de Rede Social Gerada")
    plot_graph(G_social, pos_social, graph_type="Social", title="Rede Social Aleatória")
    
    st.markdown("---")
    st.subheader("Análise da Rede")
    
    analysis_type = st.selectbox(
        "Escolha o tipo de análise:",
        ["Métricas de Centralidade", "Detecção de Comunidades"],
        key="social_analysis_type"
    )
    
    if G_social.number_of_nodes() == 0:
        st.warning("Gere um grafo com pelo menos 1 nó para realizar a análise.")
    elif analysis_type == "Métricas de Centralidade":
        st.write("Calculando métricas de centralidade:")
        
        degree_centrality = nx.degree_centrality(G_social)
        betweenness_centrality = nx.betweenness_centrality(G_social)
        closeness_centrality = nx.closeness_centrality(G_social)
        
        st.subheader("Centralidade de Grau")
        df_degree = pd.DataFrame.from_dict(degree_centrality, orient='index', columns=['Grau'])
        st.dataframe(df_degree.sort_values(by='Grau', ascending=False))
        st.info("Mede o número de conexões diretas de um nó. Pessoas com alto grau têm muitas amizades.")
        
        st.subheader("Centralidade de Intermediação")
        df_betweenness = pd.DataFrame.from_dict(betweenness_centrality, orient='index', columns=['Intermediação'])
        st.dataframe(df_betweenness.sort_values(by='Intermediação', ascending=False))
        st.info("Mede a frequência com que um nó aparece nos caminhos mais curtos entre outros pares de nós. Pessoas com alta intermediação são 'pontes'.")
        
        st.subheader("Centralidade de Proximidade")
        df_closeness = pd.DataFrame.from_dict(closeness_centrality, orient='index', columns=['Proximidade'])
        st.dataframe(df_closeness.sort_values(by='Proximidade', ascending=False))
        st.info("Mede o quão perto um nó está de todos os outros. Pessoas com alta proximidade podem espalhar informações rapidamente.")
        
        # Visualização de Centralidade de Grau
        if G_social.number_of_nodes() > 0:
            node_sizes_degree = [v * 3000 for v in degree_centrality.values()] # Ajuste o multiplicador
            node_colors_degree = [degree_centrality[node] for node in G_social.nodes()]
            
            plt.figure(figsize=(8, 6))
            nx.draw(G_social, pos_social, with_labels=True, node_size=node_sizes_degree,
                    node_color=node_colors_degree, cmap=plt.cm.viridis,
                    edge_color=GRAPH_COLORS["Social"]["edge"], width=2, edgecolors='white', linewidths=1,
                    font_size=10, font_weight='bold')
            plt.title("Visualização por Centralidade de Grau", fontweight='bold')
            # Garante que o colorbar lide com valores min/max corretamente
            if node_colors_degree: # Evita erro se a lista estiver vazia
                sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(node_colors_degree), vmax=max(node_colors_degree)), cmap=plt.cm.viridis)
                sm.set_array([]) # Necessário para o colorbar
                plt.colorbar(sm, ax=plt.gca(), label="Centralidade de Grau")
            plt.axis('off')
            fig = plt.gcf()
            fig.patch.set_facecolor('white')
            st.pyplot(plt)
        else:
            st.warning("O grafo está vazio, não é possível visualizar a centralidade.")
        
    elif analysis_type == "Detecção de Comunidades":
        st.write("Detectando comunidades usando o algoritmo de Louvain:")
        
        if G_social.number_of_nodes() > 0:
            partition = community_louvain.best_partition(G_social)
            num_communities = len(set(partition.values()))
            
            st.success(f"Encontradas {num_communities} comunidades na rede.")
            
            plot_graph(G_social, pos_social, graph_type="Social", communities=partition, title="Comunidades na Rede Social")
            
            st.info("Nós da mesma cor pertencem à mesma comunidade. Essas comunidades são grupos de pessoas mais conectadas entre si.")
        else:
            st.warning("O grafo social está vazio. Não é possível detectar comunidades.")

# --- Aba: Finanças (Bellman-Ford) ---
elif opcao == "Finanças (Bellman-Ford)":
    st.header("💰 Aplicações Financeiras com Bellman-Ford", divider='rainbow')
    
    st.markdown("""
    ### Detecção de Arbitragem em Câmbio de Moedas
    **Por que Bellman-Ford?**
    - **Único algoritmo** que detecta **ciclos negativos** em grafos com pesos negativos.
    - Ideal para identificar oportunidades de **arbitragem** em câmbio de moedas.
    - Permite encontrar ciclos onde a multiplicação das taxas de câmbio resulta em **ganho**.
    
    **Como funciona?**
    1. Modelamos as moedas como **vértices** e as conversões como **arestas**.
    2. Usamos **-log(taxa)** como peso para transformar o problema de otimização de produto em busca de soma mínima.
    3. Bellman-Ford identifica se existe tal ciclo (oportunidade de arbitragem), onde a soma dos pesos negativos indica um ganho no câmbio.
    """)
    
    # Criar grafo financeiro (fixo para arbitragem)
    G_finance_fixed = create_finance_network()
    pos_finance_fixed = nx.circular_layout(G_finance_fixed)
    edge_labels_finance_fixed = { (u, v): f"{np.exp(-d['weight']):.4f}" for u, v, d in G_finance_fixed.edges(data=True) }
    
    st.subheader("Rede de Conversão de Moedas (Exemplo Fixo para Arbitragem)")
    plot_graph(G_finance_fixed, pos_finance_fixed, edge_labels_finance_fixed, graph_type="Finance", title="Exemplo de Rede de Câmbio")
    
    st.markdown("""
    **Explicação das Arestas:**
    - Uma aresta `USD` → `EUR` com label `0.90` significa que **1 USD compra 0.90 EUR**.
    - O peso interno usado pelo algoritmo é `-ln(0.90)` (aproximadamente `0.105`).
    - Uma sequência de trocas (um ciclo) que resulte em um **produto das taxas > 1** (e.g., $1 \to 1.03) indica arbitragem. Isso corresponde a uma **soma de pesos negativos** no grafo transformado.
    """)
    
    if st.button("Detectar Oportunidades de Arbitragem (Exemplo Fixo)", key="finance_arbitrage_btn_fixed"):
        arbitrage_found = False
        
        for currency in G_finance_fixed.nodes():
            cycle = detect_arbitrage(G_finance_fixed, currency)
            if cycle:
                st.error(f"🚨 **Arbitragem encontrada!** Ciclo: {' → '.join(cycle)}")
                
                gain = 1.0
                st.write("**Passos para o Lucro:**")
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    rate = np.exp(-G_finance_fixed[u][v]['weight'])
                    gain *= rate
                    st.write(f"- Trocar **{u}** por **{v}** (taxa: **{rate:.4f}**)")
                
                u_final, v_final = cycle[-1], cycle[0]
                rate_final = np.exp(-G_finance_fixed[u_final][v_final]['weight'])
                gain *= rate_final
                st.write(f"- Trocar **{u_final}** por **{v_final}** (taxa: **{rate_final:.4f}**)")
                
                st.success(f"**Ganho Total:** {gain:.4f} (Lucro: **{(gain-1)*100:.2f}%** em cada unidade inicial)")
                
                st.subheader("Ciclo de Arbitragem Destacado")
                plot_graph(G_finance_fixed, pos_finance_fixed, edge_labels_finance_fixed, graph_type="Finance", highlight_path=cycle, title="Ciclo de Arbitragem")
                
                arbitrage_found = True
                break
        
        if not arbitrage_found:
            st.success("✅ Nenhuma oportunidade de arbitragem encontrada neste momento no grafo fixo.")
    
    st.markdown("---")
    st.subheader("Simulação do Algoritmo Bellman-Ford (Caminho Mais Curto em Grafo Aleatório)")
    st.info("Esta simulação mostra como o Bellman-Ford encontra o caminho de menor 'custo' (ou seja, maior ganho) entre duas moedas, respeitando pesos negativos. Se houver ciclos negativos, ele irá detectá-los.")

    col_nodes_bf, col_edges_bf, col_directed_bf = st.columns(3)
    with col_nodes_bf:
        num_nodes_bf = st.slider("Número de Nós", min_value=5, max_value=20, value=7, key="num_nodes_bf")
    with col_edges_bf:
        max_possible_edges_bf = num_nodes_bf * (num_nodes_bf - 1)
        num_edges_bf = st.slider("Número de Arestas", min_value=num_nodes_bf - 1, max_value=max_possible_edges_bf, value=int(max_possible_edges_bf * 0.3), key="num_edges_bf")
        if num_edges_bf < num_nodes_bf - 1:
            st.warning("O número de arestas deve ser no mínimo (Nós - 1) para um grafo conectado.")
            num_edges_bf = num_nodes_bf - 1
    with col_directed_bf:
        is_directed_bf = st.checkbox("Grafo Direcionado?", value=True, key="is_directed_bf") # Bellman-Ford é mais comum em digrafos

    st.info("Para Bellman-Ford, os pesos das arestas podem ser **positivos ou negativos**.")
    min_weight_bf = st.slider("Peso Mínimo da Aresta", min_value=-10, max_value=10, value=-5, key="min_weight_bf")
    max_weight_bf = st.slider("Peso Máximo da Aresta", min_value=1, max_value=10, value=5, key="max_weight_bf")

    generate_graph_button_bf = st.button("Gerar Novo Grafo Aleatório para Bellman-Ford", key="generate_bf_graph")

    if 'G_bf' not in st.session_state or generate_graph_button_bf:
        st.session_state.G_bf = generate_random_graph(num_nodes_bf, num_edges_bf, is_directed=is_directed_bf, min_weight=min_weight_bf, max_weight=max_weight_bf)
        st.session_state.pos_bf = nx.spring_layout(st.session_state.G_bf, seed=42)
    
    G_bf = st.session_state.G_bf
    pos_bf = st.session_state.pos_bf
    edge_labels_bf = nx.get_edge_attributes(G_bf, 'weight')

    st.subheader("Grafo Aleatório para Bellman-Ford")
    plot_graph(G_bf, pos_bf, edge_labels_bf, graph_type="Finance", title="Grafo Aleatório com Pesos Positivos/Negativos")
    
    nodes_in_graph_bf = list(G_bf.nodes())
    if len(nodes_in_graph_bf) < 2:
        st.warning("Gere um grafo com pelo menos 2 nós para calcular um caminho.")
    else:
        col_start_bf, col_end_bf = st.columns(2)
        with col_start_bf:
            start_node_bf = st.selectbox("Nó de Origem", nodes_in_graph_bf, key="bf_start")
        with col_end_bf:
            default_end_index_bf = (nodes_in_graph_bf.index(start_node_bf) + 1) % len(nodes_in_graph_bf)
            end_node_bf = st.selectbox("Nó de Destino", nodes_in_graph_bf, index=default_end_index_bf, key="bf_end")
        
        if st.button("Executar Bellman-Ford", key="finance_bellman_ford_sim_btn"):
            if start_node_bf == end_node_bf:
                st.warning("Nó de origem e destino são os mesmos. Não há caminho a calcular.")
            elif start_node_bf not in G_bf.nodes() or end_node_bf not in G_bf.nodes():
                st.error("Nó de origem ou destino não encontrado no grafo.")
            else:
                path, distance, pos, steps, has_negative_cycle = bellman_ford_with_animation(G_bf, start_node_bf, end_node_bf)
                
                if has_negative_cycle:
                    st.error("⚠️ **Alerta:** O grafo gerado contém **ciclos negativos**! O caminho mais curto pode não ser bem definido. A animação mostrará a detecção de um ciclo negativo (se reconstruído).")
                
                if path:
                    st.success(f"Caminho encontrado: {' → '.join(path)}")
                    st.info(f"Custo total do caminho: {distance:.2f}")
                    
                    st.subheader("Animação do Algoritmo Bellman-Ford")
                    gif_bytes = animate_algorithm(G_bf, pos, steps, graph_type="Finance", algorithm_name="Bellman-Ford")
                    st.image(gif_bytes, use_column_width=True)
                    
                    st.subheader("Caminho Final Destacado")
                    plot_graph(G_bf, pos, edge_labels_bf, graph_type="Finance", highlight_path=path, title="Caminho Bellman-Ford")
                elif has_negative_cycle:
                    # Se não há um path para o destino, mas há ciclo negativo, mostrar a detecção
                    st.warning("Um ciclo negativo foi detectado, impedindo a determinação de um caminho mais curto único. A animação mostrará o ciclo.")
                    if steps:
                         st.subheader("Animação do Algoritmo Bellman-Ford")
                         gif_bytes = animate_algorithm(G_bf, pos, steps, graph_type="Finance", algorithm_name="Bellman-Ford (Ciclo Negativo)")
                         st.image(gif_bytes, use_column_width=True)
                         
                         final_node_status = steps[-1][0]
                         final_edge_status = steps[-1][1]
                         highlighted_nodes = [node for node, visited in final_node_status.items() if visited]
                         
                         plt.figure(figsize=(8, 6))
                         node_colors_final = [GRAPH_COLORS["Finance"]["visited"] if node in highlighted_nodes else GRAPH_COLORS["Finance"]["node"] for node in G_bf.nodes()]
                         edge_colors_final = []
                         for u, v in G_bf.edges():
                             edge_key = (u,v) if G_bf.is_directed() else tuple(sorted((u,v)))
                             if edge_key in final_edge_status:
                                 edge_colors_final.append(GRAPH_COLORS["Finance"]["path_edge"])
                             else:
                                 edge_colors_final.append(GRAPH_COLORS["Finance"]["edge"])

                         nx.draw(G_bf, pos, with_labels=True, node_size=800,
                                 node_color=node_colors_final,
                                 edge_color=edge_colors_final,
                                 width=2, edgecolors='white', linewidths=1, font_size=10, font_weight='bold')
                         
                         nx.draw_networkx_edge_labels(G_bf, pos, edge_labels=edge_labels_bf, font_size=8)
                         plt.title(f"Ciclo Negativo Detectado por Bellman-Ford", fontweight='bold')
                         plt.axis('off')
                         fig = plt.gcf()
                         fig.patch.set_facecolor('white')
                         st.pyplot(plt)
                else:
                    st.error("Não foi encontrado um caminho entre os nós selecionados, e nenhum ciclo negativo foi detectado que inclua o destino.")

# Rodapé
st.markdown("---")
st.markdown("""
**🔍 Observação:** Esta aplicação demonstra algoritmos de forma simplificada para fins educacionais.
Os grafos aleatórios são gerados com diferentes densidades e pesos para explorar a robustez dos algoritmos.
""")