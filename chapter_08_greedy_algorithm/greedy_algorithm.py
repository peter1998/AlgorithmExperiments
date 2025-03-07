"""
Greedy Algorithms and NP-Complete Problems - Chapter 8 Implementation
====================================================================

This module demonstrates the concepts from Chapter 8 of "Grokking Algorithms":
1. Greedy algorithms and their limitations
2. NP-complete problems
3. Approximation algorithms for hard problems
4. Examples of greedy vs. optimal solutions
5. Identifying NP-complete problems

"""

import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import defaultdict, deque

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Greedy Algorithms Examples --------------------

class Box:
    """Represents a box with width, height, depth dimensions."""
    
    def __init__(self, width: float, height: float, depth: float, name: str = None):
        """Initialize a box with given dimensions."""
        self.width = width
        self.height = height
        self.depth = depth
        self.volume = width * height * depth
        self.name = name or f"Box({width}x{height}x{depth})"
    
    def fits_in(self, remaining_width: float, remaining_height: float, 
                remaining_depth: float) -> bool:
        """Check if the box fits in the given remaining space."""
        # Try all possible orientations
        orientations = [
            (self.width, self.height, self.depth),
            (self.width, self.depth, self.height),
            (self.height, self.width, self.depth),
            (self.height, self.depth, self.width),
            (self.depth, self.width, self.height),
            (self.depth, self.height, self.width)
        ]
        
        for w, h, d in orientations:
            if (w <= remaining_width and h <= remaining_height and d <= remaining_depth):
                return True
        
        return False
    
    def __str__(self) -> str:
        """String representation of the box."""
        return f"{self.name}: {self.width}x{self.height}x{self.depth} (vol: {self.volume})"

def pack_truck_greedy(boxes: List[Box], truck_width: float, truck_height: float, 
                     truck_depth: float) -> List[Box]:
    """
    Greedy algorithm for the box packing problem (Exercise 8.1).
    
    Args:
        boxes: List of Box objects to pack
        truck_width: Width of the truck
        truck_height: Height of the truck
        truck_depth: Depth of the truck
        
    Returns:
        List of boxes that were packed into the truck
    """
    # Sort boxes by volume in descending order
    sorted_boxes = sorted(boxes, key=lambda box: box.volume, reverse=True)
    
    # Remaining space in the truck (simplified model)
    remaining_width = truck_width
    remaining_height = truck_height
    remaining_depth = truck_depth
    remaining_volume = truck_width * truck_height * truck_depth
    
    packed_boxes = []
    
    # Simple greedy algorithm: take boxes in order of decreasing volume
    for box in sorted_boxes:
        if box.volume <= remaining_volume and box.fits_in(remaining_width, remaining_height, remaining_depth):
            packed_boxes.append(box)
            # Update remaining volume (this is very simplified)
            remaining_volume -= box.volume
    
    return packed_boxes

def visualize_truck_packing(packed_boxes: List[Box], unpacked_boxes: List[Box], 
                           truck_dimensions: Tuple[float, float, float], 
                           title: str = "Truck Packing Visualization"):
    """
    Visualize the truck packing problem results.
    
    Args:
        packed_boxes: List of boxes that were packed
        unpacked_boxes: List of boxes that were not packed
        truck_dimensions: (width, height, depth) of the truck
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Calculate total possible volume and used volume
    truck_volume = truck_dimensions[0] * truck_dimensions[1] * truck_dimensions[2]
    used_volume = sum(box.volume for box in packed_boxes)
    efficiency = (used_volume / truck_volume) * 100
    
    # Create bar chart
    plt.subplot(1, 2, 1)
    plt.bar(['Truck Capacity', 'Used Space'], [truck_volume, used_volume], color=['gray', 'green'])
    plt.ylabel('Volume')
    plt.title(f'Space Utilization: {efficiency:.2f}%')
    
    # Create pie chart of packed vs unpacked boxes
    plt.subplot(1, 2, 2)
    packed_count = len(packed_boxes)
    unpacked_count = len(unpacked_boxes)
    plt.pie([packed_count, unpacked_count], 
            labels=[f'Packed: {packed_count}', f'Unpacked: {unpacked_count}'],
            autopct='%1.1f%%',
            colors=['green', 'red'])
    plt.title('Boxes Packed vs. Unpacked')
    
    # Add overall title
    plt.suptitle(title)
    plt.tight_layout()
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Truck packing visualization saved as '{filename}'")
    plt.close()

class Activity:
    """Represents an activity with a point value and duration."""
    
    def __init__(self, name: str, points: int, duration: float):
        """Initialize an activity with name, points, and duration in hours."""
        self.name = name
        self.points = points
        self.duration = duration
        # Value density: points per hour
        self.value_density = points / duration if duration > 0 else 0
    
    def __str__(self) -> str:
        """String representation of the activity."""
        return f"{self.name}: {self.points} points, {self.duration} hours"

def plan_trip_greedy(activities: List[Activity], available_time: float) -> List[Activity]:
    """
    Greedy algorithm for the trip planning problem (Exercise 8.2).
    
    Args:
        activities: List of Activity objects to choose from
        available_time: Total available time in hours
        
    Returns:
        List of activities selected for the trip
    """
    # Sort activities by point value in descending order
    sorted_activities = sorted(activities, key=lambda activity: activity.points, reverse=True)
    
    selected_activities = []
    remaining_time = available_time
    
    # Greedy algorithm: take activities with highest point value first
    for activity in sorted_activities:
        if activity.duration <= remaining_time:
            selected_activities.append(activity)
            remaining_time -= activity.duration
    
    return selected_activities

def plan_trip_optimal(activities: List[Activity], available_time: float) -> List[Activity]:
    """
    Dynamic programming solution for the trip planning problem (optimal solution).
    This is essentially the knapsack problem.
    
    Args:
        activities: List of Activity objects to choose from
        available_time: Total available time in hours
        
    Returns:
        List of activities selected for the trip
    """
    # Discretize time for dynamic programming
    # Scale to integers to avoid floating point issues
    scale_factor = 10  # Assuming we want 0.1 hour precision
    max_time = int(available_time * scale_factor)
    n = len(activities)
    
    # Create DP table
    # dp[i][j] = maximum points with first i activities and j time
    dp = [[0 for _ in range(max_time + 1)] for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for j in range(max_time + 1):
            activity = activities[i-1]
            activity_time = int(activity.duration * scale_factor)
            
            # If we can't take this activity, copy the value from above
            if activity_time > j:
                dp[i][j] = dp[i-1][j]
            else:
                # Max of taking or not taking the activity
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-activity_time] + activity.points)
    
    # Reconstruct the solution
    selected_activities = []
    remaining_time = max_time
    
    for i in range(n, 0, -1):
        # If value changed, we took this activity
        if dp[i][remaining_time] != dp[i-1][remaining_time]:
            activity = activities[i-1]
            selected_activities.append(activity)
            remaining_time -= int(activity.duration * scale_factor)
    
    return selected_activities

def visualize_trip_planning(greedy_activities: List[Activity], optimal_activities: List[Activity],
                           all_activities: List[Activity], available_time: float,
                           title: str = "Trip Planning Comparison"):
    """
    Visualize the trip planning problem results.
    
    Args:
        greedy_activities: Activities selected by the greedy algorithm
        optimal_activities: Activities selected by the optimal algorithm
        all_activities: All available activities
        available_time: Total available time in hours
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Calculate statistics
    greedy_points = sum(activity.points for activity in greedy_activities)
    greedy_time = sum(activity.duration for activity in greedy_activities)
    
    optimal_points = sum(activity.points for activity in optimal_activities)
    optimal_time = sum(activity.duration for activity in optimal_activities)
    
    # Plot activities by points vs. duration
    plt.subplot(2, 2, 1)
    plt.scatter([a.duration for a in all_activities], 
                [a.points for a in all_activities], 
                color='gray', alpha=0.5, label='All Activities')
    
    plt.scatter([a.duration for a in greedy_activities], 
                [a.points for a in greedy_activities], 
                color='blue', label='Greedy Selection')
    
    plt.scatter([a.duration for a in optimal_activities], 
                [a.points for a in optimal_activities], 
                color='green', label='Optimal Selection')
    
    plt.xlabel('Duration (hours)')
    plt.ylabel('Points')
    plt.title('Activities by Points vs. Duration')
    plt.legend()
    
    # Bar chart comparing total points
    plt.subplot(2, 2, 2)
    plt.bar(['Greedy', 'Optimal'], [greedy_points, optimal_points])
    plt.ylabel('Total Points')
    plt.title('Total Points Comparison')
    
    # Bar chart comparing time usage
    plt.subplot(2, 2, 3)
    plt.bar(['Available', 'Greedy', 'Optimal'], 
            [available_time, greedy_time, optimal_time])
    plt.ylabel('Time (hours)')
    plt.title('Time Usage Comparison')
    
    # Table of selected activities
    plt.subplot(2, 2, 4)
    plt.axis('off')
    table_data = [
        ['Algorithm', 'Points', 'Time Used', 'Efficiency'],
        ['Greedy', f"{greedy_points}", f"{greedy_time:.1f}h", f"{greedy_points/greedy_time:.1f} pts/h"],
        ['Optimal', f"{optimal_points}", f"{optimal_time:.1f}h", f"{optimal_points/optimal_time:.1f} pts/h"]
    ]
    plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.25, 0.25, 0.25])
    plt.title('Performance Comparison')
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Trip planning visualization saved as '{filename}'")
    plt.close()

# -------------------- NP-Complete Problems Examples --------------------

def generate_tsp_graph(num_cities: int, seed: int = 42) -> nx.Graph:
    """
    Generate a random graph for the Traveling Salesman Problem.
    
    Args:
        num_cities: Number of cities
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph representing cities and distances
    """
    random.seed(seed)
    
    # Create a complete graph with random edge weights
    G = nx.complete_graph(num_cities)
    
    # Assign random weights to edges
    for (u, v) in G.edges():
        G[u][v]['weight'] = random.randint(1, 100)
    
    return G

def tsp_nearest_neighbor(graph: nx.Graph, start_city: int = 0) -> Tuple[List[int], float]:
    """
    Nearest neighbor heuristic for the Traveling Salesman Problem.
    
    Args:
        graph: NetworkX graph representing cities and distances
        start_city: Starting city index
        
    Returns:
        Tuple of (tour, total_distance)
    """
    tour = [start_city]
    current_city = start_city
    unvisited = set(graph.nodes())
    unvisited.remove(start_city)
    total_distance = 0
    
    while unvisited:
        # Find the nearest unvisited city
        next_city = min(unvisited, 
                        key=lambda city: graph[current_city][city]['weight'])
        
        # Add to tour
        tour.append(next_city)
        total_distance += graph[current_city][next_city]['weight']
        
        # Update current city
        current_city = next_city
        unvisited.remove(next_city)
    
    # Return to start city
    tour.append(start_city)
    total_distance += graph[current_city][start_city]['weight']
    
    return tour, total_distance

def visualize_tsp(graph: nx.Graph, tour: List[int], title: str = "TSP Solution"):
    """
    Visualize a solution to the Traveling Salesman Problem.
    
    Args:
        graph: NetworkX graph representing cities and distances
        tour: List of cities in the order they are visited
        title: Title for the visualization
    """
    # Create a new graph for visualization
    tour_graph = nx.Graph()
    
    # Add all nodes from the original graph
    for node in graph.nodes():
        tour_graph.add_node(node)
    
    # Add edges from the tour
    for i in range(len(tour) - 1):
        city1, city2 = tour[i], tour[i + 1]
        weight = graph[city1][city2]['weight']
        tour_graph.add_edge(city1, city2, weight=weight)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create positions for the cities (in a circle for simplicity)
    pos = nx.circular_layout(graph)
    
    # Draw the original graph (light edges)
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_edges(graph, pos, width=0.5, alpha=0.2)
    
    # Draw the tour (heavy edges)
    nx.draw_networkx_edges(tour_graph, pos, width=2, edge_color='blue')
    
    # Draw edge labels for the tour
    edge_labels = {(tour[i], tour[i+1]): graph[tour[i]][tour[i+1]]['weight'] 
                  for i in range(len(tour) - 1)}
    nx.draw_networkx_edge_labels(tour_graph, pos, edge_labels=edge_labels)
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_size=12)
    
    # Set title and display
    plt.title(title)
    plt.axis('off')
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"TSP visualization saved as '{filename}'")
    plt.close()

def generate_graph_coloring_problem(num_regions: int, edge_probability: float = 0.3, 
                                   seed: int = 42) -> nx.Graph:
    """
    Generate a random graph for the Graph Coloring Problem.
    
    Args:
        num_regions: Number of regions
        edge_probability: Probability of an edge between any two regions
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph representing regions and adjacencies
    """
    random.seed(seed)
    
    # Create a random graph
    G = nx.gnp_random_graph(num_regions, edge_probability, seed=seed)
    
    # Make sure the graph is connected
    if not nx.is_connected(G):
        for component in list(nx.connected_components(G))[1:]:
            # Connect to a node in the main component
            node_in_component = list(component)[0]
            node_in_main = list(nx.connected_components(G))[0][0]
            G.add_edge(node_in_component, node_in_main)
    
    return G

def greedy_graph_coloring(graph: nx.Graph) -> Dict[int, int]:
    """
    Greedy algorithm for the Graph Coloring Problem.
    
    Args:
        graph: NetworkX graph representing regions
        
    Returns:
        Dictionary mapping node to color (integer)
    """
    # Sort nodes by degree (highest first) for better results
    nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
    
    colors = {}  # node -> color
    
    for node in nodes:
        # Get colors of adjacent nodes
        adjacent_colors = {colors.get(neighbor) for neighbor in graph.neighbors(node) 
                           if neighbor in colors}
        
        # Find the smallest available color
        color = 0
        while color in adjacent_colors:
            color += 1
        
        colors[node] = color
    
    return colors

def visualize_graph_coloring(graph: nx.Graph, coloring: Dict[int, int], 
                            title: str = "Graph Coloring Solution"):
    """
    Visualize a solution to the Graph Coloring Problem.
    
    Args:
        graph: NetworkX graph representing regions
        coloring: Dictionary mapping node to color
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Generate positions for the nodes
    pos = nx.spring_layout(graph, seed=42)
    
    # Get a list of all colors used
    colors_used = set(coloring.values())
    num_colors = len(colors_used)
    
    # Create a colormap
    color_map = plt.cm.get_cmap('tab10', num_colors)
    
    # Draw the graph
    nx.draw(graph, pos, 
            node_color=[color_map(coloring[node]) for node in graph.nodes()],
            with_labels=True,
            font_weight='bold')
    
    # Set title and add color information
    plt.title(f"{title}\nNumber of colors used: {num_colors}")
    plt.axis('off')
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Graph coloring visualization saved as '{filename}'")
    plt.close()

def generate_clique_problem(num_nodes: int, clique_size: int, edge_probability: float = 0.2,
                           seed: int = 42) -> nx.Graph:
    """
    Generate a random graph with a clique for the Maximum Clique Problem.
    
    Args:
        num_nodes: Total number of nodes
        clique_size: Size of the clique to embed
        edge_probability: Probability of random edges outside the clique
        seed: Random seed for reproducibility
        
    Returns:
        NetworkX graph with an embedded clique
    """
    random.seed(seed)
    
    # Create a graph with the specified number of nodes
    G = nx.Graph()
    for i in range(num_nodes):
        G.add_node(i)
    
    # Create a clique (fully connected subgraph)
    clique_nodes = list(range(clique_size))
    for i in range(clique_size):
        for j in range(i + 1, clique_size):
            G.add_edge(i, j)
    
    # Add random edges outside the clique
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if not G.has_edge(i, j) and random.random() < edge_probability:
                G.add_edge(i, j)
    
    return G

def greedy_max_clique(graph: nx.Graph) -> List[int]:
    """
    Greedy algorithm for the Maximum Clique Problem.
    
    Args:
        graph: NetworkX graph
        
    Returns:
        List of nodes forming a clique
    """
    # Sort nodes by degree (highest first)
    nodes = sorted(graph.nodes(), key=lambda node: graph.degree(node), reverse=True)
    
    clique = []
    
    for node in nodes:
        # Check if node forms a clique with all nodes already in the clique
        can_add = True
        for clique_node in clique:
            if not graph.has_edge(node, clique_node):
                can_add = False
                break
        
        if can_add:
            clique.append(node)
    
    return clique

def visualize_clique(graph: nx.Graph, clique: List[int], 
                    title: str = "Maximum Clique Solution"):
    """
    Visualize a solution to the Maximum Clique Problem.
    
    Args:
        graph: NetworkX graph
        clique: List of nodes forming a clique
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Generate positions for the nodes
    pos = nx.spring_layout(graph, seed=42)
    
    # Create a set of clique nodes for faster lookup
    clique_set = set(clique)
    
    # Draw the graph with clique nodes highlighted
    nx.draw_networkx_nodes(graph, pos, 
                          nodelist=[n for n in graph.nodes() if n not in clique_set],
                          node_color='lightblue',
                          node_size=300)
    
    nx.draw_networkx_nodes(graph, pos, 
                          nodelist=clique,
                          node_color='red',
                          node_size=500)
    
    # Draw edges within the clique with a different color
    clique_edges = [(u, v) for u in clique for v in clique if u < v]
    non_clique_edges = [(u, v) for (u, v) in graph.edges() if (u, v) not in clique_edges]
    
    nx.draw_networkx_edges(graph, pos, 
                          edgelist=non_clique_edges,
                          width=1.0,
                          alpha=0.5)
    
    nx.draw_networkx_edges(graph, pos, 
                          edgelist=clique_edges,
                          width=2.0,
                          edge_color='red')
    
    # Draw node labels
    nx.draw_networkx_labels(graph, pos, font_weight='bold')
    
    # Set title and add clique information
    plt.title(f"{title}\nClique size: {len(clique)}")
    plt.axis('off')
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300)
    logger.info(f"Maximum clique visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercise 8.1 - Box Packing --------------------

def exercise_8_1():
    """
    Exercise 8.1: Box packing problem using a greedy algorithm.
    """
    logger.info("\nExercise 8.1: Box packing problem")
    logger.info("-" * 60)
    
    # Create a set of boxes with different dimensions
    boxes = [
        Box(2, 3, 2, "Small box"),
        Box(3, 3, 3, "Medium box"),
        Box(5, 4, 3, "Large box"),
        Box(1, 1, 2, "Tiny box"),
        Box(4, 2, 2, "Flat box"),
        Box(2, 2, 2, "Cube box"),
        Box(5, 5, 5, "XL box"),
        Box(1, 5, 3, "Long box"),
        Box(3, 2, 2, "Small medium box"),
        Box(4, 4, 4, "Large cube")
    ]
    
    # Define truck dimensions
    truck_width = 10
    truck_height = 8
    truck_depth = 15
    
    # Pack the truck using greedy algorithm
    packed_boxes = pack_truck_greedy(boxes, truck_width, truck_height, truck_depth)
    
    # Calculate statistics
    total_volume = truck_width * truck_height * truck_depth
    packed_volume = sum(box.volume for box in packed_boxes)
    unpacked_boxes = [box for box in boxes if box not in packed_boxes]
    
    logger.info(f"Truck dimensions: {truck_width}x{truck_height}x{truck_depth}")
    logger.info(f"Total truck volume: {total_volume}")
    logger.info(f"Number of boxes packed: {len(packed_boxes)} out of {len(boxes)}")
    logger.info(f"Volume utilized: {packed_volume} ({packed_volume/total_volume*100:.2f}%)")
    
    logger.info("\nBoxes packed:")
    for box in packed_boxes:
        logger.info(f"  {box}")
    
    logger.info("\nBoxes not packed:")
    for box in unpacked_boxes:
        logger.info(f"  {box}")
    
    # Visualize the results
    visualize_truck_packing(packed_boxes, unpacked_boxes, 
                           (truck_width, truck_height, truck_depth),
                           "Exercise 8.1 - Box Packing (Greedy)")
    
    logger.info("\nConclusion for Exercise 8.1:")
    logger.info("The greedy approach of selecting the largest boxes first does not guarantee")
    logger.info("an optimal solution. It might leave gaps that could be filled with smaller boxes,")
    logger.info("resulting in suboptimal space utilization.")

# -------------------- Exercise 8.2 - Trip Planning --------------------

def exercise_8_2():
    """
    Exercise 8.2: Trip planning problem using greedy and optimal algorithms.
    """
    logger.info("\nExercise 8.2: Trip planning problem")
    logger.info("-" * 60)
    
    # Create a set of activities for a 7-day Europe trip
    activities = [
        Activity("Eiffel Tower", 8, 3),
        Activity("Louvre Museum", 10, 4),
        Activity("Colosseum", 9, 3),
        Activity("Vatican Museums", 10, 4),
        Activity("Sagrada Familia", 8, 2),
        Activity("British Museum", 7, 3),
        Activity("Acropolis", 9, 3),
        Activity("Rijksmuseum", 6, 2),
        Activity("Neuschwanstein Castle", 7, 5),
        Activity("Prague Castle", 6, 3),
        Activity("Anne Frank House", 7, 1.5),
        Activity("Alhambra", 8, 3),
        Activity("Prado Museum", 6, 2),
        Activity("Versailles Palace", 7, 4),
        Activity("Park Güell", 5, 1.5),
        Activity("Stonehenge", 6, 2),
        Activity("Sistine Chapel", 9, 2),
        Activity("Buckingham Palace", 5, 1),
        Activity("Swiss Alps Hiking", 8, 6),
        Activity("Amsterdam Canal Cruise", 4, 1)
    ]
    
    # Total available time (7 days, 10 hours per day)
    available_time = 7 * 10
    
    # Plan the trip using the greedy algorithm
    greedy_plan = plan_trip_greedy(activities, available_time)
    
    # Plan the trip using the optimal algorithm (dynamic programming)
    optimal_plan = plan_trip_optimal(activities, available_time)
    
    # Calculate statistics
    greedy_points = sum(activity.points for activity in greedy_plan)
    greedy_time = sum(activity.duration for activity in greedy_plan)
    
    optimal_points = sum(activity.points for activity in optimal_plan)
    optimal_time = sum(activity.duration for activity in optimal_plan)
    
    logger.info(f"Available time: {available_time} hours")
    
    logger.info("\nGreedy plan (selecting activities with highest points):")
    logger.info(f"Total points: {greedy_points}")
    logger.info(f"Total time used: {greedy_time} hours")
    for activity in greedy_plan:
        logger.info(f"  {activity}")
    
    logger.info("\nOptimal plan (dynamic programming):")
    logger.info(f"Total points: {optimal_points}")
    logger.info(f"Total time used: {optimal_time} hours")
    for activity in optimal_plan:
        logger.info(f"  {activity}")
    
    # Visualize the results
    visualize_trip_planning(greedy_plan, optimal_plan, activities, available_time,
                          "Exercise 8.2 - Europe Trip Planning")
    
    logger.info("\nConclusion for Exercise 8.2:")
    logger.info("The greedy approach of selecting activities with the highest points first")
    logger.info("does not guarantee an optimal solution. The optimal solution using dynamic")
    logger.info("programming can achieve higher total points by selecting a better combination")
    logger.info("of activities that fit within the time constraint.")

# -------------------- Exercise 8.3-8.5 - Identifying Greedy Algorithms --------------------

def exercise_8_3_to_8_5():
    """
    Exercises 8.3-8.5: Identify which algorithms are greedy.
    """
    logger.info("\nExercises 8.3-8.5: Identifying greedy algorithms")
    logger.info("-" * 60)
    
    # Exercise 8.3: Is Quicksort a greedy algorithm?
    logger.info("Exercise 8.3: Is Quicksort a greedy algorithm?")
    logger.info("Answer: No. Quicksort is a divide-and-conquer algorithm, not a greedy algorithm.")
    logger.info("Explanation: Quicksort does not make locally optimal choices at each step.")
    logger.info("Instead, it recursively divides the problem and solves subproblems independently.")
    
    # Exercise 8.4: Is Breadth-first search a greedy algorithm?
    logger.info("\nExercise 8.4: Is Breadth-first search a greedy algorithm?")
    logger.info("Answer: Yes. BFS can be considered a greedy algorithm.")
    logger.info("Explanation: BFS always explores the closest vertices first (in terms of number of edges).")
    logger.info("This is a locally optimal choice, making it greedy in nature.")
    
    # Exercise 8.5: Is Dijkstra's algorithm a greedy algorithm?
    logger.info("\nExercise 8.5: Is Dijkstra's algorithm a greedy algorithm?")
    logger.info("Answer: Yes. Dijkstra's algorithm is a greedy algorithm.")
    logger.info("Explanation: Dijkstra's algorithm always selects the vertex with the smallest distance")
    logger.info("from the source, which is a locally optimal choice at each step.")
    
    logger.info("\nSummary:")
    logger.info("8.3 Quicksort - Not a greedy algorithm")
    logger.info("8.4 Breadth-first search - Greedy algorithm")
    logger.info("8.5 Dijkstra's algorithm - Greedy algorithm")
    
# -------------------- Exercises 8.6-8.8 - Identifying NP-Complete Problems --------------------

def exercise_8_6():
    """
    Exercise 8.6: Is the Traveling Salesman Problem NP-complete?
    """
    logger.info("\nExercise 8.6: Traveling Salesman Problem")
    logger.info("-" * 60)
    
    logger.info("Question: A postman needs to deliver to 20 homes. He needs to find the shortest")
    logger.info("route that goes to all 20 homes. Is this an NP-complete problem?")
    
    logger.info("\nAnswer: Yes, this is the Traveling Salesman Problem (TSP), which is NP-complete.")
    
    logger.info("\nExplanation:")
    logger.info("The Traveling Salesman Problem involves finding the shortest possible route")
    logger.info("that visits each city exactly once and returns to the origin city.")
    logger.info("Key characteristics that make it NP-complete:")
    logger.info("1. The number of possible routes grows factorially with the number of cities")
    logger.info("2. There is no known polynomial-time algorithm to solve it optimally")
    logger.info("3. The problem can be verified in polynomial time")
    
    # Demonstrate TSP with a small example
    num_cities = 10
    logger.info(f"\nDemonstration with {num_cities} cities:")
    
    # Generate a random graph for TSP
    tsp_graph = generate_tsp_graph(num_cities)
    
    # Solve with the nearest neighbor heuristic
    start_time = time.time()
    tour, total_distance = tsp_nearest_neighbor(tsp_graph)
    end_time = time.time()
    
    # Report results
    logger.info(f"Nearest neighbor heuristic solution:")
    logger.info(f"Tour: {tour}")
    logger.info(f"Total distance: {total_distance}")
    logger.info(f"Computation time: {(end_time - start_time)*1000:.2f} ms")
    
    # For comparison, calculate the number of possible tours
    num_possible_tours = factorial(num_cities - 1) // 2  # (n-1)! / 2
    logger.info(f"Number of possible tours: {num_possible_tours:,}")
    
    # Estimated time for brute force
    estimated_time_per_tour = 1e-6  # 1 microsecond per tour (very optimistic)
    estimated_total_time = num_possible_tours * estimated_time_per_tour
    
    if estimated_total_time < 60:
        logger.info(f"Estimated time for brute force: {estimated_total_time:.2f} seconds")
    elif estimated_total_time < 3600:
        logger.info(f"Estimated time for brute force: {estimated_total_time/60:.2f} minutes")
    elif estimated_total_time < 86400:
        logger.info(f"Estimated time for brute force: {estimated_total_time/3600:.2f} hours")
    elif estimated_total_time < 31536000:
        logger.info(f"Estimated time for brute force: {estimated_total_time/86400:.2f} days")
    else:
        logger.info(f"Estimated time for brute force: {estimated_total_time/31536000:.2f} years")
    
    # Visualize the TSP solution
    visualize_tsp(tsp_graph, tour, "Exercise 8.6 - TSP Solution")

def exercise_8_7():
    """
    Exercise 8.7: Is finding the largest clique NP-complete?
    """
    logger.info("\nExercise 8.7: Maximum Clique Problem")
    logger.info("-" * 60)
    
    logger.info("Question: Finding the largest clique in a set of people (a clique is a set")
    logger.info("of people who all know each other). Is this an NP-complete problem?")
    
    logger.info("\nAnswer: Yes, finding the maximum clique is an NP-complete problem.")
    
    logger.info("\nExplanation:")
    logger.info("The Maximum Clique Problem involves finding the largest complete subgraph")
    logger.info("(a group where everyone is connected to everyone else).")
    logger.info("Key characteristics that make it NP-complete:")
    logger.info("1. The number of possible subsets grows exponentially")
    logger.info("2. There is no known polynomial-time algorithm")
    logger.info("3. The problem can be verified in polynomial time")
    
    # Demonstrate with a small example
    num_nodes = 15
    known_clique_size = 5
    
    logger.info(f"\nDemonstration with a graph of {num_nodes} people:")
    
    # Generate a random graph with an embedded clique
    clique_graph = generate_clique_problem(num_nodes, known_clique_size)
    
    # Use a greedy algorithm to find a clique
    start_time = time.time()
    found_clique = greedy_max_clique(clique_graph)
    end_time = time.time()
    
    # Report results
    logger.info(f"Greedy algorithm found a clique of size: {len(found_clique)}")
    logger.info(f"Clique: {found_clique}")
    logger.info(f"Computation time: {(end_time - start_time)*1000:.2f} ms")
    
    # For comparison, calculate the number of possible subsets
    num_possible_subsets = 2**num_nodes  # 2^n
    logger.info(f"Number of possible subsets to check: {num_possible_subsets:,}")
    
    # Visualize the clique solution
    visualize_clique(clique_graph, found_clique, "Exercise 8.7 - Maximum Clique")

def exercise_8_8():
    """
    Exercise 8.8: Is the Graph Coloring Problem NP-complete?
    """
    logger.info("\nExercise 8.8: Graph Coloring Problem")
    logger.info("-" * 60)
    
    logger.info("Question: You're making a map of the USA, and you need to color adjacent")
    logger.info("states with different colors. You have to find the minimum number of colors")
    logger.info("you need so that no two adjacent states are the same color. Is this an")
    logger.info("NP-complete problem?")
    
    logger.info("\nAnswer: Yes, the Graph Coloring Problem is NP-complete.")
    
    logger.info("\nExplanation:")
    logger.info("The Graph Coloring Problem involves assigning colors to vertices of a graph")
    logger.info("such that no adjacent vertices have the same color, using the minimum")
    logger.info("number of colors possible.")
    logger.info("Key characteristics that make it NP-complete:")
    logger.info("1. Finding the minimum number of colors (chromatic number) is hard")
    logger.info("2. There is no known polynomial-time algorithm")
    logger.info("3. The problem can be verified in polynomial time")
    
    # Demonstrate with the USA map
    logger.info("\nDemonstration with a simplified USA map:")
    
    # Create a graph representing the USA (simplified)
    usa_graph = nx.Graph()
    
    # Add some states (not all 50 for simplicity)
    states = [
        "Washington", "Oregon", "California", "Nevada", "Idaho", "Montana", 
        "Wyoming", "Utah", "Colorado", "Arizona", "New Mexico", "Texas",
        "Oklahoma", "Kansas", "Nebraska", "South Dakota", "North Dakota"
    ]
    
    for state in states:
        usa_graph.add_node(state)
    
    # Add borders (adjacencies)
    borders = [
        ("Washington", "Oregon"), ("Washington", "Idaho"),
        ("Oregon", "Idaho"), ("Oregon", "California"), ("Oregon", "Nevada"),
        ("California", "Nevada"), ("California", "Arizona"),
        ("Nevada", "Idaho"), ("Nevada", "Utah"), ("Nevada", "Arizona"),
        ("Idaho", "Montana"), ("Idaho", "Wyoming"), ("Idaho", "Utah"),
        ("Montana", "Wyoming"), ("Montana", "North Dakota"), ("Montana", "South Dakota"),
        ("Wyoming", "South Dakota"), ("Wyoming", "Nebraska"), ("Wyoming", "Colorado"), ("Wyoming", "Utah"),
        ("Utah", "Colorado"), ("Utah", "Arizona"), ("Utah", "New Mexico"),
        ("Colorado", "Kansas"), ("Colorado", "Nebraska"), ("Colorado", "Oklahoma"), ("Colorado", "New Mexico"),
        ("Arizona", "New Mexico"),
        ("New Mexico", "Texas"), ("New Mexico", "Oklahoma"),
        ("Texas", "Oklahoma"),
        ("Oklahoma", "Kansas"),
        ("Kansas", "Nebraska"),
        ("Nebraska", "South Dakota"),
        ("South Dakota", "North Dakota")
    ]
    
    for (state1, state2) in borders:
        usa_graph.add_edge(state1, state2)
    
    # Apply greedy coloring
    start_time = time.time()
    coloring = greedy_graph_coloring(usa_graph)
    end_time = time.time()
    
    # Count the number of colors used
    num_colors = len(set(coloring.values()))
    
    # Report results
    logger.info(f"Greedy algorithm used {num_colors} colors")
    
    # Group states by color
    colors_to_states = defaultdict(list)
    for state, color in coloring.items():
        colors_to_states[color].append(state)
    
    for color, states_list in colors_to_states.items():
        logger.info(f"Color {color}: {', '.join(states_list)}")
    
    logger.info(f"Computation time: {(end_time - start_time)*1000:.2f} ms")
    
    # Visualize the coloring
    visualize_graph_coloring(usa_graph, coloring, "Exercise 8.8 - USA Map Coloring")

def factorial(n):
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n-1)

# -------------------- Main Function --------------------

def main():
    """Main function to run the greedy algorithms and NP-completeness demonstrations."""
    logger.info("=" * 80)
    logger.info("Greedy Algorithms and NP-Complete Problems - Chapter 8 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run Exercise 8.1 - Box Packing
        exercise_8_1()
        
        # Run Exercise 8.2 - Trip Planning
        exercise_8_2()
        
        # Run Exercises 8.3-8.5 - Identifying Greedy Algorithms
        exercise_8_3_to_8_5()
        
        # Run Exercise 8.6 - Traveling Salesman Problem
        exercise_8_6()
        
        # Run Exercise 8.7 - Maximum Clique Problem
        exercise_8_7()
        
        # Run Exercise 8.8 - Graph Coloring Problem
        exercise_8_8()
        
        logger.info("\nSummary of Chapter 8 exercises:")
        logger.info("8.1 Box Packing - Greedy solution is not optimal")
        logger.info("8.2 Trip Planning - Greedy solution is not optimal")
        logger.info("8.3 Quicksort - Not a greedy algorithm")
        logger.info("8.4 Breadth-first search - Greedy algorithm")
        logger.info("8.5 Dijkstra's algorithm - Greedy algorithm")
        logger.info("8.6 TSP - NP-complete problem")
        logger.info("8.7 Maximum Clique - NP-complete problem")
        logger.info("8.8 Graph Coloring - NP-complete problem")
        
        logger.info("\nAll exercises completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()