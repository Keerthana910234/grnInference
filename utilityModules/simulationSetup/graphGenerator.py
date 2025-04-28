import networkx as nx
import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd

def relabel_nodes(G):
    mapping = {i: f"g{i+1}" for i in range(G.number_of_nodes())}
    relabeled_G = nx.relabel_nodes(G, mapping)
    return relabeled_G

def generate_canonical_form(G):
    return nx.weisfeiler_lehman_graph_hash(G)

def generate_random_adjacency_matrix(n, num_edges, seed=None, max_attempts = 10):
    random.seed(seed)
    G = nx.DiGraph()
    for i in range(n):
        G.add_node(i)

        # Start with a randomly connected structure
    nodes = list(G.nodes())
    random.shuffle(nodes)
    connected_nodes = [nodes.pop()]  # Start with one node and grow the connected component

    while nodes:
        new_node = nodes.pop()
        if random.random() < 0.5:
                # Connect from new node to connected component
            G.add_edge(new_node, random.choice(connected_nodes))
        else:
                # Connect from connected component to new node
            G.add_edge(random.choice(connected_nodes), new_node)
        connected_nodes.append(new_node)

    # Additional edges while ensuring acyclicity
    possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j and not G.has_edge(i, j)]
    random.shuffle(possible_edges)

    edges_added = len(G.edges)
    while edges_added < num_edges and possible_edges:
        edge = possible_edges.pop(0)
        G.add_edge(*edge)
        edges_added += 1
        #Removed the condition to check for diacyclic networks
        if edges_added == num_edges:
            relabelledG = relabel_nodes(G)
            return relabelledG
    raise ValueError(f"Unable to generate a DAG with {n} nodes and {num_edges} edges after {max_attempts} attempts.")

def generateLinearAdjacencyMatrix(n, seed=None):
    """
    Generate a linear directed acyclic graph (DAG) with n nodes.

    Args:
        n (int): Number of nodes in the graph.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        nx.DiGraph: A linear directed acyclic graph with n nodes.
    """
    random.seed(seed)
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i)
    
    # Add edges in a linear fashion (i -> i+1)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    
    # Relabel nodes if desired
    relabelledG = relabel_nodes(G)
    return relabelledG

def generateGraph(n, num_edges = 5, seed= 101010, max_attempts=100):
    # G = generate_random_adjacency_matrix(n, num_edges, seed, max_attempts)
    G = generateLinearAdjacencyMatrix(n)
    canonical_form = generate_canonical_form(G)
    return G, canonical_form

def generate_random_weakly_connected_dag(n, num_edges, seed=None, max_attempts=10):
    random.seed(seed)
    for attempt in range(max_attempts):
        G = nx.DiGraph()
        for i in range(n):
            G.add_node(i)

        # Start with a randomly connected structure
        nodes = list(G.nodes())
        random.shuffle(nodes)
        connected_nodes = [nodes.pop()]  # Start with one node and grow the connected component

        while nodes:
            new_node = nodes.pop()
            if random.random() < 0.5:
                # Connect from new node to connected component
                G.add_edge(new_node, random.choice(connected_nodes))
            else:
                # Connect from connected component to new node
                G.add_edge(random.choice(connected_nodes), new_node)
            connected_nodes.append(new_node)

        # Additional edges while ensuring acyclicity
        possible_edges = [(i, j) for i in range(n) for j in range(n) if i != j and not G.has_edge(i, j)]
        random.shuffle(possible_edges)

        edges_added = len(G.edges)
        while edges_added < num_edges and possible_edges:
            edge = possible_edges.pop(0)
            G.add_edge(*edge)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(*edge)
            else:
                edges_added += 1

        if edges_added == num_edges:
            relabelledG = relabel_nodes(G)
            return relabelledG

    raise ValueError(f"Unable to generate a DAG with {n} nodes and {num_edges} edges after {max_attempts} attempts.")

def generateGraphDag(n, num_edges, seed, max_attempts):
    G = generate_random_weakly_connected_dag(n, num_edges, seed, max_attempts)
    canonical_form = generate_canonical_form(G)
    return G, canonical_form

def visualizeGraphs(graphs, per_row=3):
    """
    Visualize a list of directed graph objects with arrowheads in a circular layout,
    where arrows connect properly to nodes.
    """
    total = len(graphs)
    rows = (total + per_row - 1) // per_row
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 20, rows * 20))

    if rows == 1:
        axes = [axes]
    elif rows > 1:
        axes = axes.tolist()

    # Increase node size
    node_size = 10000  # Increased from 3000

    radius = np.sqrt(node_size) / 2

    for i, G in enumerate(graphs):
        row_index = i // per_row
        col_index = i % per_row
        ax = axes[row_index][col_index]
        
        pos = nx.circular_layout(G)

        # Draw nodes with increased size
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', 
                             node_size=node_size, alpha=0.6)

        # Draw edges with smaller arrows
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='black',
            width=5,  # Reduced from 2
            arrowsize=50,  # Reduced from 20
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            min_source_margin=40,  # Slightly increased margin for larger nodes
            min_target_margin=40   
        )

        # Draw labels with larger font size
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=30, font_weight='bold')  # Increased from 14

        ax.set_title(f'Graph {i+1}', fontsize=30)
        ax.set_aspect('equal')
        ax.axis("off")

    for i in range(total, rows * per_row):
        row_index = i // per_row
        col_index = i % per_row
        axes[row_index][col_index].axis('off')

    plt.tight_layout()
    plt.show()

# def visualizeGraphs(graphs, per_row=3):
#     """
#     Visualize a list of directed graph objects with arrowheads in a circular layout,
#     where arrows connect properly to nodes.
#     """
#     total = len(graphs)
#     rows = (total + per_row - 1) // per_row
#     fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 20, rows * 20))

#     if rows == 1:
#         axes = [axes]
#     elif rows > 1:
#         axes = axes.tolist()

#     # Calculate node radius based on node size
#     node_size = 3000
#     radius = np.sqrt(node_size) / 2

#     for i, G in enumerate(graphs):
#         row_index = i // per_row
#         col_index = i % per_row
#         ax = axes[row_index][col_index]
        
#         # Use circular layout
#         pos = nx.circular_layout(G)

#         # Draw nodes
#         nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', 
#                              node_size=node_size, alpha=0.6)

#         # Draw edges with arrows
#         nx.draw_networkx_edges(
#             G, pos, ax=ax,
#             edge_color='black',
#             width=2,
#             arrowsize=20,
#             arrowstyle='->',
#             connectionstyle='arc3,rad=0.1',
#             min_source_margin=25,  # Distance from source node
#             min_target_margin=25   # Distance from target node
#         )

#         # Draw labels
#         nx.draw_networkx_labels(G, pos, ax=ax, font_size=14, font_weight='bold')

#         # Set title
#         ax.set_title(f'Graph {i+1}', fontsize=30)
        
#         # Set equal aspect ratio and remove axes
#         ax.set_aspect('equal')
#         ax.axis("off")

#     # Hide any unused subplot axes
#     for i in range(total, rows * per_row):
#         row_index = i // per_row
#         col_index = i % per_row
#         axes[row_index][col_index].axis('off')

#     plt.tight_layout()
#     plt.show()

# def visualizeGraphs(graphs, per_row=3):
#     """
#     Visualize a list of graph objects in a grid layout using matplotlib and networkx.

#     This function displays the graphs specified in the list, arranging them into rows 
#     and columns. Each graph is drawn using a Kamada-Kawai layout to spread nodes evenly.
#     The function handles both directional and bi-directional edges distinctly.

#     Parameters
#     ----------
#     graphs : list of networkx.Graph
#         A list of networkx graph objects to be visualized. Each graph can be either directed or undirected.
#     per_row : int, optional
#         The number of graphs to display per row in the visualization. Defaults to 3.

#     Returns
#     -------
#     None
#         The function directly visualizes the graphs and does not return any value.
#     """
#     total = len(graphs)
#     rows = (total + per_row - 1) // per_row  # Calculate the required number of rows
#     fig, axes = plt.subplots(rows, per_row, figsize=(per_row * 6, rows * 6))  # Adjusted figure size
    
#     # Ensure axes is always a 2D array for consistency
#     if rows == 1:
#         axes = [axes]
#     elif rows > 1:
#         axes = axes.tolist()

#     for i, G in enumerate(graphs):
#         row_index = i // per_row
#         col_index = i % per_row
#         ax = axes[row_index][col_index]
        
#         pos = nx.kamada_kawai_layout(G)  # Layout that might spread nodes more evenly
#         # Manually scale positions to increase edge lengths
#         pos = {node: (x * 2, y * 2) for node, (x, y) in pos.items()}

#         nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=500, alpha=0.6)
#         nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

#         for u, v in G.edges():
#             if G.has_edge(v, u):
#                 # Draw bi-directional edges with a curve
#                 nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], arrows=True,
#                                        connectionstyle='arc3,rad=0.2', arrowstyle='-|>', style='solid')
#             else:
#                 # Draw normal directed edges
#                 nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], arrows=True,
#                                        connectionstyle='arc3,rad=0.0', arrowstyle='-|>', style='solid')

#         ax.set_title(f'Graph {i+1}')

#     # Hide axes for unused subplots
#     for i in range(total, rows * per_row):
#         row_index = i // per_row
#         col_index = i % per_row
#         ax = axes[row_index][col_index]
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()



#########################################################################################################################
#OLD CODE#
# def generate_random_adjacency_matrix(n, seed=None):
#     np.random.seed(seed)
#     while True:
#         matrix = np.random.randint(0, 2, size=(n, n))
#         np.fill_diagonal(matrix, 0)  # No self-loops
#         G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
#         G = relabel_nodes(G)
#         if nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G):  # Check if the graph is weakly connected
#             return matrix

# def generate_canonical_form(matrix):
#     G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
#     return nx.weisfeiler_lehman_graph_hash(G)

# def relabel_nodes(G):
#     # Define a mapping from old labels (integers) to new labels (strings)
#     mapping = {i: f"g{i+1}" for i in range(G.number_of_nodes())}
#     # Relabel the nodes according to the mapping
#     G = nx.relabel_nodes(G, mapping)
#     return G

# def generate_graph(n, seed):
#     new_matrix = generate_random_adjacency_matrix(n, seed)
#     new_canonical_form = generate_canonical_form(new_matrix)
#     return new_matrix, new_canonical_form

# def generate_random_adjacency_matrix(n, seed=None):
#     np.random.seed(seed)
#     while True:
#         matrix = np.random.randint(0, 2, size=(n, n))
#         np.fill_diagonal(matrix, 0)  # No self-loops
#         G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
#         relabeled_G = relabel_nodes(G)
#         if nx.is_weakly_connected(relabeled_G):
#             return relabeled_G  # Return the graph after relabeling and checks
#         # if nx.is_weakly_connected(relabeled_G) and nx.is_directed_acyclic_graph(relabeled_G):
#         #     return relabeled_G  # Return the graph after relabeling and checks
########################################################################################################################
