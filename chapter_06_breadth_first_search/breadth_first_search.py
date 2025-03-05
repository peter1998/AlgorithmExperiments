"""
Breadth-First Search and Graphs - Chapter 6 Implementation
==========================================================

This module demonstrates the concepts from Chapter 6 of "Grokking Algorithms":
1. Breadth-First Search (BFS) algorithm
2. Graphs implementation and traversal
3. Finding shortest paths with BFS
4. Working with different types of graphs
5. Distinguishing between graphs and trees


"""

import logging
import time
from collections import deque
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Graph Implementation --------------------

class Graph:
    """
    A simple graph implementation using adjacency lists.
    """
    def __init__(self, directed: bool = False):
        """
        Initialize an empty graph.
        
        Args:
            directed: Whether the graph is directed (True) or undirected (False)
        """
        self.adjacency_list = {}
        self.directed = directed
    
    def add_vertex(self, vertex: Any) -> None:
        """Add a vertex to the graph if it doesn't exist."""
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []
    
    def add_edge(self, vertex1: Any, vertex2: Any) -> None:
        """
        Add an edge between two vertices.
        
        Args:
            vertex1: First vertex
            vertex2: Second vertex
        """
        # Add vertices if they don't exist
        self.add_vertex(vertex1)
        self.add_vertex(vertex2)
        
        # Add edge from vertex1 to vertex2
        if vertex2 not in self.adjacency_list[vertex1]:
            self.adjacency_list[vertex1].append(vertex2)
        
        # If graph is undirected, add edge from vertex2 to vertex1 as well
        if not self.directed and vertex1 not in self.adjacency_list[vertex2]:
            self.adjacency_list[vertex2].append(vertex1)
    
    def get_neighbors(self, vertex: Any) -> List[Any]:
        """Get all neighbors of a vertex."""
        return self.adjacency_list.get(vertex, [])
    
    def get_vertices(self) -> List[Any]:
        """Get all vertices in the graph."""
        return list(self.adjacency_list.keys())
    
    def __str__(self) -> str:
        """String representation of the graph."""
        result = "Graph:\n"
        for vertex, neighbors in self.adjacency_list.items():
            result += f"{vertex} -> {neighbors}\n"
        return result

# -------------------- Breadth-First Search Implementation --------------------

def breadth_first_search(graph: Graph, start: Any, target: Any = None) -> Optional[List[Any]]:
    """
    Perform breadth-first search from start vertex.
    
    Args:
        graph: The graph to search
        start: The starting vertex
        target: Optional target vertex to find
        
    Returns:
        If target is specified, returns the shortest path to target or None if not found.
        If target is None, returns all vertices in BFS order.
    """
    if start not in graph.adjacency_list:
        logger.warning(f"Start vertex {start} not in graph")
        return None
    
    # Queue for BFS
    queue = deque([start])
    
    # Track visited vertices
    visited = {start}
    
    # Track parent of each vertex for reconstructing path
    parent = {start: None}
    
    # Result list if no specific target
    result = [start]
    
    # BFS
    while queue:
        current = queue.popleft()
        
        # Return path if target is found
        if target is not None and current == target:
            return reconstruct_path(parent, start, target)
        
        # Process all neighbors
        for neighbor in graph.get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current
                result.append(neighbor)
    
    # If we're looking for a specific target and didn't find it
    if target is not None:
        return None
    
    # Otherwise return all vertices in BFS order
    return result

def reconstruct_path(parent: Dict[Any, Any], start: Any, end: Any) -> List[Any]:
    """
    Reconstruct path from start to end using parent dictionary.
    
    Args:
        parent: Dictionary mapping each vertex to its parent
        start: Start vertex
        end: End vertex
        
    Returns:
        List of vertices forming the path from start to end
    """
    path = []
    current = end
    
    # Work backwards from end to start
    while current is not None:
        path.append(current)
        current = parent[current]
    
    # Reverse to get path from start to end
    path.reverse()
    return path

def shortest_path_length(graph: Graph, start: Any, end: Any) -> Optional[int]:
    """
    Find the length of the shortest path between start and end.
    
    Args:
        graph: The graph to search
        start: Start vertex
        end: End vertex
        
    Returns:
        The length (number of edges) in the shortest path or None if no path exists
    """
    path = breadth_first_search(graph, start, end)
    if path is None:
        return None
    
    # Path length is number of edges, which is number of vertices - 1
    return len(path) - 1

# -------------------- Visualizing Graphs --------------------

def visualize_graph(graph: Graph, title: str = "Graph Visualization", 
                   highlight_path: List[Any] = None, layout: str = "spring",
                   node_labels: bool = True) -> None:
    """
    Visualize a graph using NetworkX and matplotlib.
    
    Args:
        graph: The graph to visualize
        title: Title for the visualization
        highlight_path: Optional list of vertices to highlight as a path
        layout: The layout algorithm to use ('spring', 'circular', etc.)
        node_labels: Whether to show node labels
    """
    # Create NetworkX graph
    if graph.directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    # Add all vertices and edges
    for vertex in graph.get_vertices():
        G.add_node(vertex)
    
    for vertex, neighbors in graph.adjacency_list.items():
        for neighbor in neighbors:
            G.add_edge(vertex, neighbor)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
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
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    
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
    
    # Draw labels if requested
    if node_labels:
        nx.draw_networkx_labels(G, pos, font_size=12)
    
    # Set title and display
    plt.title(title)
    plt.axis("off")
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    logger.info(f"Graph visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercises 6.1 - Simple BFS Example --------------------

def exercise_6_1():
    """
    Exercise 6.1: Find the length of the shortest path from start to finish.
    """
    logger.info("\nExercise 6.1: Find the shortest path from start to finish")
    logger.info("-" * 60)
    
    # Create the graph from the exercise
    graph = Graph()
    
    # Add edges based on the exercise diagram
    graph.add_edge("start", "a")
    graph.add_edge("start", "b")
    graph.add_edge("a", "c")
    graph.add_edge("a", "d")
    graph.add_edge("b", "e")
    graph.add_edge("c", "f")
    graph.add_edge("d", "g")
    graph.add_edge("e", "finish")
    graph.add_edge("f", "finish")
    graph.add_edge("g", "finish")
    
    logger.info(f"Graph structure:\n{graph}")
    
    # Find shortest path
    path = breadth_first_search(graph, "start", "finish")
    length = shortest_path_length(graph, "start", "finish")
    
    logger.info(f"Shortest path from start to finish: {path}")
    logger.info(f"Length of shortest path: {length}")
    
    # Visualize the graph with the shortest path highlighted
    visualize_graph(graph, "Exercise 6.1 - Shortest Path from Start to Finish", 
                   highlight_path=path)
    
    logger.info(f"The answer to Exercise 6.1 is: The shortest path has a length of {length}")

# -------------------- Exercise 6.2 - Word Transformation --------------------

def build_word_transformation_graph(words: List[str]) -> Graph:
    """
    Build a graph where words are connected if they differ by one letter.
    
    Args:
        words: List of words to include in the graph
        
    Returns:
        A graph with edges between words that differ by exactly one letter
    """
    graph = Graph()
    
    # Add all words as vertices
    for word in words:
        graph.add_vertex(word)
    
    # Connect words that differ by exactly one letter
    for i, word1 in enumerate(words):
        for word2 in words[i+1:]:
            if one_letter_difference(word1, word2):
                graph.add_edge(word1, word2)
    
    return graph

def one_letter_difference(word1: str, word2: str) -> bool:
    """Check if two words differ by exactly one letter."""
    if len(word1) != len(word2):
        return False
    
    differences = 0
    for c1, c2 in zip(word1, word2):
        if c1 != c2:
            differences += 1
            if differences > 1:
                return False
    
    return differences == 1

def exercise_6_2():
    """
    Exercise 6.2: Find the length of the shortest path from "cab" to "bat".
    """
    logger.info("\nExercise 6.2: Find the shortest path from 'cab' to 'bat'")
    logger.info("-" * 60)
    
    # Create a graph with the words
    words = ["cab", "cat", "car", "bar", "bat", "mat", "hat", "cot", "dog"]
    graph = build_word_transformation_graph(words)
    
    logger.info(f"Word transformation graph:\n{graph}")
    
    # Find shortest path
    path = breadth_first_search(graph, "cab", "bat")
    length = shortest_path_length(graph, "cab", "bat")
    
    logger.info(f"Shortest path from 'cab' to 'bat': {path}")
    logger.info(f"Length of shortest path: {length}")
    
    # Visualize the graph with the shortest path highlighted
    visualize_graph(graph, "Exercise 6.2 - Word Transformation", 
                   highlight_path=path)
    
    logger.info(f"The answer to Exercise 6.2 is: The shortest path has a length of {length}")

# -------------------- Exercise 6.3 - Topological Sort Validation --------------------

def is_valid_topological_order(graph: Graph, order: List[Any]) -> bool:
    """
    Check if a given order is a valid topological ordering for a graph.
    
    Args:
        graph: A directed graph
        order: A list of vertices
        
    Returns:
        True if the order is a valid topological ordering, False otherwise
    """
    # Check if all vertices are in the order
    if set(graph.get_vertices()) != set(order):
        return False
    
    # Create a mapping from vertex to its position in the order
    position = {vertex: i for i, vertex in enumerate(order)}
    
    # Check that for each edge (u, v), u comes before v in the order
    for vertex, neighbors in graph.adjacency_list.items():
        vertex_pos = position[vertex]
        for neighbor in neighbors:
            if vertex_pos >= position[neighbor]:
                # Found an edge that violates the topological ordering
                return False
    
    return True

def exercise_6_3():
    """
    Exercise 6.3: Check if each list is a valid topological sort.
    """
    logger.info("\nExercise 6.3: Validate topological sort orderings")
    logger.info("-" * 60)
    
    # Create a directed graph for morning routine
    graph = Graph(directed=True)
    
    # Add edges based on the exercise
    graph.add_edge("wake up", "exercise")
    graph.add_edge("wake up", "brush teeth")
    graph.add_edge("exercise", "shower")
    graph.add_edge("brush teeth", "eat breakfast")
    graph.add_edge("shower", "get dressed")
    graph.add_edge("get dressed", "eat breakfast")
    
    logger.info(f"Morning routine graph:\n{graph}")
    
    # The three orders to check
    order_a = ["wake up", "brush teeth", "exercise", "eat breakfast", "shower", "get dressed"]
    order_b = ["wake up", "exercise", "shower", "get dressed", "brush teeth", "eat breakfast"]
    order_c = ["wake up", "brush teeth", "exercise", "shower", "eat breakfast", "get dressed"]
    
    # Check each order
    valid_a = is_valid_topological_order(graph, order_a)
    valid_b = is_valid_topological_order(graph, order_b)
    valid_c = is_valid_topological_order(graph, order_c)
    
    logger.info(f"Order A: {order_a}")
    logger.info(f"Valid? {valid_a}")
    
    logger.info(f"Order B: {order_b}")
    logger.info(f"Valid? {valid_b}")
    
    logger.info(f"Order C: {order_c}")
    logger.info(f"Valid? {valid_c}")
    
    # Visualize the graph
    visualize_graph(graph, "Exercise 6.3 - Morning Routine Graph", layout="circular")
    
    logger.info("The answers to Exercise 6.3 are:")
    logger.info(f"A: {'Valid' if valid_a else 'Invalid'}")
    logger.info(f"B: {'Valid' if valid_b else 'Invalid'}")
    logger.info(f"C: {'Valid' if valid_c else 'Invalid'}")

# -------------------- Exercise 6.4 - Create Valid Topological Order --------------------

def topological_sort(graph: Graph) -> List[Any]:
    """
    Perform a topological sort on a directed acyclic graph.
    
    Args:
        graph: A directed acyclic graph
        
    Returns:
        A valid topological ordering of the vertices
    """
    # Count incoming edges for each vertex
    in_degree = {vertex: 0 for vertex in graph.get_vertices()}
    for vertex in graph.get_vertices():
        for neighbor in graph.get_neighbors(vertex):
            in_degree[neighbor] = in_degree.get(neighbor, 0) + 1
    
    # Queue of vertices with no incoming edges
    queue = deque([vertex for vertex, degree in in_degree.items() if degree == 0])
    
    # Result list
    result = []
    
    # Process vertices
    while queue:
        current = queue.popleft()
        result.append(current)
        
        # Update in-degrees of neighbors
        for neighbor in graph.get_neighbors(current):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    # Check if all vertices are included (no cycles)
    if len(result) != len(graph.get_vertices()):
        raise ValueError("Graph contains a cycle, no valid topological ordering exists")
    
    return result

def exercise_6_4():
    """
    Exercise 6.4: Create a valid topological ordering for a larger graph.
    """
    logger.info("\nExercise 6.4: Create a valid topological ordering")
    logger.info("-" * 60)
    
    # Create a directed graph for the morning routine
    graph = Graph(directed=True)
    
    # Add edges based on the exercise
    graph.add_edge("Wake up", "Exercise")
    graph.add_edge("Wake up", "Shower")
    graph.add_edge("Exercise", "Shower")
    graph.add_edge("Shower", "Brush teeth")
    graph.add_edge("Shower", "Get dressed")
    graph.add_edge("Brush teeth", "Eat breakfast")
    graph.add_edge("Get dressed", "Eat breakfast")
    graph.add_edge("Pack lunch", "Eat breakfast")
    
    logger.info(f"Larger morning routine graph:\n{graph}")
    
    # Perform topological sort
    try:
        order = topological_sort(graph)
        logger.info(f"Valid topological ordering: {order}")
        
        # Verify the ordering
        is_valid = is_valid_topological_order(graph, order)
        logger.info(f"Is the ordering valid? {is_valid}")
        
        # Visualize the graph
        visualize_graph(graph, "Exercise 6.4 - Larger Morning Routine Graph", layout="circular")
        
        logger.info("The answer to Exercise 6.4 is:")
        logger.info(f"A valid ordering: {order}")
    except ValueError as e:
        logger.error(f"Error performing topological sort: {e}")

# -------------------- Exercise 6.5 - Identifying Trees --------------------

def is_tree(graph: Graph) -> bool:
    """
    Check if a given graph is a tree.
    
    A graph is a tree if it is connected and has no cycles.
    
    Args:
        graph: The graph to check
        
    Returns:
        True if the graph is a tree, False otherwise
    """
    if not graph.get_vertices():
        return True  # Empty graph is a tree by definition
    
    # Start BFS from any vertex
    start = next(iter(graph.get_vertices()))
    visited = set()
    queue = deque([(start, None)])  # (vertex, parent)
    
    while queue:
        current, parent = queue.popleft()
        
        # Mark current vertex as visited
        visited.add(current)
        
        # Check all neighbors
        for neighbor in graph.get_neighbors(current):
            # If we find a neighbor that is already visited and is not the parent,
            # then we have found a cycle
            if neighbor in visited and neighbor != parent:
                return False
            
            # Add unvisited neighbors to the queue
            if neighbor not in visited:
                queue.append((neighbor, current))
    
    # If the graph is connected, all vertices should be visited
    return len(visited) == len(graph.get_vertices())

def exercise_6_5():
    """
    Exercise 6.5: Determine which graphs are also trees.
    """
    logger.info("\nExercise 6.5: Identifying trees")
    logger.info("-" * 60)
    
    # Create Graph A
    graph_a = Graph()
    graph_a.add_edge(1, 2)
    graph_a.add_edge(1, 3)
    graph_a.add_edge(2, 4)
    graph_a.add_edge(2, 5)
    graph_a.add_edge(3, 6)
    graph_a.add_edge(3, 7)
    
    # Create Graph B
    graph_b = Graph()
    graph_b.add_edge(1, 2)
    graph_b.add_edge(1, 3)
    graph_b.add_edge(2, 4)
    graph_b.add_edge(3, 4)  # This edge creates a cycle
    
    # Create Graph C
    graph_c = Graph()
    graph_c.add_edge("A", "B")
    graph_c.add_edge("A", "C")
    graph_c.add_edge("B", "D")
    graph_c.add_edge("B", "E")
    
    # Check if each graph is a tree
    is_a_tree = is_tree(graph_a)
    is_b_tree = is_tree(graph_b)
    is_c_tree = is_tree(graph_c)
    
    logger.info(f"Graph A is a tree: {is_a_tree}")
    logger.info(f"Graph B is a tree: {is_b_tree}")
    logger.info(f"Graph C is a tree: {is_c_tree}")
    
    # Visualize the graphs
    visualize_graph(graph_a, "Exercise 6.5 - Graph A", layout="circular")
    visualize_graph(graph_b, "Exercise 6.5 - Graph B", layout="circular")
    visualize_graph(graph_c, "Exercise 6.5 - Graph C", layout="circular")
    
    logger.info("The answers to Exercise 6.5 are:")
    logger.info(f"A: {'Tree' if is_a_tree else 'Not a tree'}")
    logger.info(f"B: {'Tree' if is_b_tree else 'Not a tree'}")
    logger.info(f"C: {'Tree' if is_c_tree else 'Not a tree'}")

# -------------------- Main Function --------------------

def main():
    """Main function to run the BFS implementations and exercises."""
    logger.info("=" * 80)
    logger.info("Breadth-First Search and Graphs - Chapter 6 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run Exercise 6.1
        exercise_6_1()
        
        # Run Exercise 6.2
        exercise_6_2()
        
        # Run Exercise 6.3
        exercise_6_3()
        
        # Run Exercise 6.4
        exercise_6_4()
        
        # Run Exercise 6.5
        exercise_6_5()
        
        logger.info("\nAll exercises completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()