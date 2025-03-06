"""
Dijkstra's Algorithm and Weighted Graphs - Chapter 7 Implementation
==================================================================

This module demonstrates the concepts from Chapter 7 of "Grokking Algorithms":
1. Dijkstra's Algorithm for finding shortest paths in weighted graphs
2. Weighted graph implementation
3. Edge cases and limitations of Dijkstra's algorithm
4. Detecting negative weight cycles
5. Solving the exercise problems from Chapter 7

"""

import logging
import time
import heapq
from collections import deque
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import math

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Weighted Graph Implementation --------------------

class WeightedGraph:
    """
    A weighted graph implementation using adjacency lists.
    """
    def __init__(self, directed: bool = False):
        """
        Initialize an empty weighted graph.
        
        Args:
            directed: Whether the graph is directed (True) or undirected (False)
        """
        self.adjacency_list = {}
        self.directed = directed
    
    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph if it doesn't exist."""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, vertex1: Any, vertex2: Any, weight: float) -> None:
        """
        Add a weighted edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
            weight: Edge weight
        """
        # Add vertices if they don't exist
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        
        # Add edge from vertex1 to vertex2
        self.adjacency_list[vertex1].append((vertex2, weight))
        
        # If graph is undirected, add edge from vertex2 to vertex1 as well
        if not self.directed:
            self.adjacency_list[vertex2].append((vertex1, weight))
    
    def get_neighbors(self, vertex: Any) -> List[Tuple[Any, float]]:
        """
        Get all neighbors of a vertex with edge weights.
        
        Returns:
            List of tuples (neighbor, weight)
        """
        return self.adjacency_list.get(vertex, [])
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self.adjacency_list.keys())
    
    def __str__(self) -> str:
        """String representation of the graph."""
        result = "Weighted Graph:\n"
        for vertex, neighbors in self.adjacency_list.items():
            result += f"{vertex} -> {neighbors}\n"
        return result

# -------------------- Dijkstra's Algorithm Implementation --------------------

def dijkstra(graph: WeightedGraph, start: Any, end: Any = None) -> Tuple[Dict[Any, float], Dict[Any, Any]]:
    """
    Implement Dijkstra's algorithm to find shortest paths from start vertex.
    
    Args:
        graph: The weighted graph to search
        start: The starting vertex
        end: Optional target vertex to find
        
    Returns:
        distances: Dictionary mapping each vertex to its shortest distance from start
        previous: Dictionary mapping each vertex to its previous vertex in the shortest path
    """
    if start not in graph.adjacency_list:
        logger.warning(f"Start vertex {start} not in graph")
        return {}, {}
    
    # Initialize distances with infinity for all vertices except start
    distances = {vertex: float('infinity') for vertex in graph.get_vertices()}
    distances[start] = 0
    
    # Initialize previous vertex for path reconstruction
    previous = {vertex: None for vertex in graph.get_vertices()}
    
    # Priority queue for vertices to process
    # Each entry is (distance, vertex)
    priority_queue = [(0, start)]
    
    # Set of processed vertices
    processed = set()
    
    while priority_queue:
        # Get vertex with smallest distance
        current_distance, current = heapq.heappop(priority_queue)
        
        # Skip if already processed or if we've reached the target
        if current in processed or (end and current == end):
            continue
        
        # Mark as processed
        processed.add(current)
        
        # If we're only interested in distance to end and we've reached it
        if end and current == end:
            break
        
        # Update distances to neighbors
        for neighbor, weight in graph.get_neighbors(current):
            # Skip processed neighbors
            if neighbor in processed:
                continue
            
            # Calculate new distance
            distance = current_distance + weight
            
            # Update if we found a shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(priority_queue, (distance, neighbor))
    
    return distances, previous

def reconstruct_path(previous: Dict[Any, Any], start: Any, end: Any) -> List[Any]:
    """
    Reconstruct path from start to end using previous dictionary.
    
    Args:
        previous: Dictionary mapping each vertex to its previous vertex
        start: Start vertex
        end: End vertex
        
    Returns:
        List of vertices forming the path from start to end
    """
    path = []
    current = end
    
    # Check if end is reachable
    if previous[end] is None and end != start:
        return []
    
    # Work backwards from end to start
    while current is not None:
        path.append(current)
        current = previous[current]
    
    # Reverse to get path from start to end
    path.reverse()
    return path

# -------------------- Negative Weight Cycle Detection --------------------

def bellman_ford(graph: WeightedGraph, start: Any) -> Tuple[Dict[Any, float], Dict[Any, Any], bool]:
    """
    Implement Bellman-Ford algorithm to detect negative weight cycles.
    
    Args:
        graph: The weighted graph to search
        start: The starting vertex
        
    Returns:
        distances: Dictionary mapping each vertex to its shortest distance from start
        previous: Dictionary mapping each vertex to its previous vertex in the shortest path
        has_negative_cycle: Boolean indicating whether a negative cycle was detected
    """
    vertices = graph.get_vertices()
    
    # Initialize distances with infinity for all vertices except start
    distances = {vertex: float('infinity') for vertex in vertices}
    distances[start] = 0
    
    # Initialize previous vertex for path reconstruction
    previous = {vertex: None for vertex in vertices}
    
    # Relax edges |V| - 1 times
    for _ in range(len(vertices) - 1):
        for u in vertices:
            for v, weight in graph.get_neighbors(u):
                if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u
    
    # Check for negative weight cycles
    # If we can relax further, then we have a negative cycle
    has_negative_cycle = False
    for u in vertices:
        for v, weight in graph.get_neighbors(u):
            if distances[u] != float('infinity') and distances[u] + weight < distances[v]:
                has_negative_cycle = True
                break
        if has_negative_cycle:
            break
    
    return distances, previous, has_negative_cycle

# -------------------- Visualizing Weighted Graphs --------------------

def visualize_weighted_graph(graph: WeightedGraph, title: str = "Weighted Graph Visualization", 
                            highlight_path: List[Any] = None, layout: str = "spring") -> None:
    """
    Visualize a weighted graph using NetworkX and matplotlib.
    
    Args:
        graph: The weighted graph to visualize
        title: Title for the visualization
        highlight_path: Optional list of vertices to highlight as a path
        layout: The layout algorithm to use ('spring', 'circular', etc.)
    """
    # Create NetworkX graph
    if graph.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add all vertices and weighted edges
    for vertex in graph.get_vertices():
        G.add_node(vertex)
    
    for vertex, neighbors in graph.adjacency_list.items():
        for neighbor, weight in neighbors:
            G.add_edge(vertex, neighbor, weight=weight)
    
    # Create figure
    plt.figure(figsize=(12, 9))
    
    # Choose layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "random":
        pos = nx.random_layout(G)
    else:
        pos = nx.spring_layout(G)  # Default to spring layout
    
    # Draw regular nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="lightblue")
    
    # Create edge labels with weights
    edge_labels = {(u, v): data['weight'] for u, v, data in G.edges(data=True)}
    
    # Draw edges and edge labels
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    # Draw highlighted path if provided
    if highlight_path and len(highlight_path) > 1:
        # Create edges from the path
        path_edges = [(highlight_path[i], highlight_path[i+1]) 
                     for i in range(len(highlight_path)-1)]
        
        # Highlight the nodes and edges in the path
        nx.draw_networkx_nodes(G, pos, nodelist=highlight_path, 
                               node_size=700, node_color="orange")
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                               width=3, edge_color="orange")
    
    # Draw vertex labels
    nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Set title and display
    plt.title(title)
    plt.axis("off")
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Graph visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercise 7.1 - Shortest Path Weight --------------------

def exercise_7_1_a():
    """
    Exercise 7.1.A: Find the weight of the shortest path from start to finish.
    """
    logger.info("\nExercise 7.1.A: Weight of the shortest path from start to finish")
    logger.info("-" * 60)
    
    # Create the weighted graph for exercise 7.1.A
    graph = WeightedGraph()
    
    # Add edges based on the exercise diagram
    graph.add_edge("start", "A", 2)
    graph.add_edge("start", "B", 5)
    graph.add_edge("A", "C", 4)
    graph.add_edge("A", "D", 2)
    graph.add_edge("B", "D", 5)
    graph.add_edge("C", "finish", 3)
    graph.add_edge("D", "finish", 1)
    
    logger.info(f"Graph structure:\n{graph}")
    
    # Find shortest path using Dijkstra's algorithm
    distances, previous = dijkstra(graph, "start")
    path = reconstruct_path(previous, "start", "finish")
    
    # Calculate total weight of the path
    total_weight = distances["finish"]
    
    logger.info(f"Shortest path from start to finish: {path}")
    logger.info(f"Weight of shortest path: {total_weight}")
    
    # Visualize the graph with the shortest path highlighted
    visualize_weighted_graph(graph, "Exercise 7.1.A - Shortest Path", 
                            highlight_path=path)
    
    logger.info(f"The answer to Exercise 7.1.A is: The weight of the shortest path is {total_weight}")
    
    return total_weight

def exercise_7_1_b():
    """
    Exercise 7.1.B: Find the weight of the shortest path from start to finish.
    """
    logger.info("\nExercise 7.1.B: Weight of the shortest path from start to finish")
    logger.info("-" * 60)
    
    # Create the weighted graph for exercise 7.1.B
    graph = WeightedGraph()
    
    # Add edges based on the exercise diagram
    graph.add_edge("start", "A", 10)
    graph.add_edge("A", "B", 20)
    graph.add_edge("A", "C", 1)
    graph.add_edge("B", "finish", 30)
    graph.add_edge("C", "finish", 30)
    
    logger.info(f"Graph structure:\n{graph}")
    
    # Find shortest path using Dijkstra's algorithm
    distances, previous = dijkstra(graph, "start")
    path = reconstruct_path(previous, "start", "finish")
    
    # Calculate total weight of the path
    total_weight = distances["finish"]
    
    logger.info(f"Shortest path from start to finish: {path}")
    logger.info(f"Weight of shortest path: {total_weight}")
    
    # Visualize the graph with the shortest path highlighted
    visualize_weighted_graph(graph, "Exercise 7.1.B - Shortest Path", 
                            highlight_path=path)
    
    logger.info(f"The answer to Exercise 7.1.B is: The weight of the shortest path is {total_weight}")
    
    return total_weight

def exercise_7_1_c():
    """
    Exercise 7.1.C: Find the weight of the shortest path from start to finish.
    This one has a negative weight cycle.
    """
    logger.info("\nExercise 7.1.C: Weight of the shortest path with negative cycle")
    logger.info("-" * 60)
    
    # Create the weighted graph for exercise 7.1.C
    graph = WeightedGraph(directed=True)
    
    # Add edges based on the exercise diagram with negative weights
    graph.add_edge("start", "A", 2)
    graph.add_edge("A", "B", 2)
    graph.add_edge("B", "C", 2)
    graph.add_edge("C", "A", -10)  # Negative weight creates a cycle
    graph.add_edge("A", "finish", 2)
    
    logger.info(f"Graph structure:\n{graph}")
    
    # Check for negative cycles using Bellman-Ford
    distances, previous, has_negative_cycle = bellman_ford(graph, "start")
    
    if has_negative_cycle:
        logger.info("A negative weight cycle was detected!")
        logger.info("This means there is no shortest path because we can achieve arbitrarily low weights.")
        
        # Visualize the graph
        visualize_weighted_graph(graph, "Exercise 7.1.C - Graph with Negative Cycle")
        
        logger.info("The answer to Exercise 7.1.C is: Trick question. No shortest path exists due to a negative-weight cycle.")
        
        return None
    else:
        # This should not execute for this graph
        path = reconstruct_path(previous, "start", "finish")
        total_weight = distances["finish"]
        logger.info(f"Shortest path from start to finish: {path}")
        logger.info(f"Weight of shortest path: {total_weight}")
        return total_weight

# -------------------- Main Function --------------------

def main():
    """Main function to run Dijkstra's algorithm implementations and exercises."""
    logger.info("=" * 80)
    logger.info("Dijkstra's Algorithm and Weighted Graphs - Chapter 7 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run Exercise 7.1.A
        weight_a = exercise_7_1_a()
        
        # Run Exercise 7.1.B
        weight_b = exercise_7_1_b()
        
        # Run Exercise 7.1.C
        weight_c = exercise_7_1_c()
        
        # Summarize the results
        logger.info("\nSummary of Exercise 7.1 results:")
        logger.info(f"A: The weight of the shortest path is {weight_a}")
        logger.info(f"B: The weight of the shortest path is {weight_b}")
        logger.info(f"C: No shortest path exists (negative-weight cycle)")
        
        logger.info("\nAll exercises completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()