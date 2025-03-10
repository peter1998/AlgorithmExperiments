"""
Dynamic Programming and Knapsack Problem - Chapter 9 Implementation
==================================================================

This module demonstrates the concepts from Chapter 9 of "Grokking Algorithms":
1. Dynamic Programming approach
2. The 0/1 Knapsack Problem
3. Longest Common Subsequence/Substring
4. Solutions to the chapter's exercises

"""

import logging
import time
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Knapsack Problem Implementation --------------------

class Item:
    """Represents an item with weight and value."""
    
    def __init__(self, name: str, weight: float, value: float):
        """
        Initialize an item with name, weight, and value.
        
        Args:
            name: Name of the item
            weight: Weight of the item
            value: Value of the item
        """
        self.name = name
        self.weight = weight
        self.value = value
        
        # Value per weight unit (for greedy approach comparison)
        self.value_per_weight = value / weight if weight > 0 else 0
    
    def __str__(self) -> str:
        """String representation of the item."""
        return f"{self.name}: weight={self.weight}, value={self.value}"

def knapsack_recursive(items: List[Item], capacity: float) -> Tuple[float, List[Item]]:
    """
    Recursive solution to the knapsack problem (not efficient).
    
    Args:
        items: List of items available
        capacity: Maximum weight capacity of the knapsack
        
    Returns:
        Tuple of (total value, list of selected items)
    """
    # Base case: no items or no capacity
    if not items or capacity <= 0:
        return 0, []
    
    # If current item is too heavy, skip it
    if items[0].weight > capacity:
        return knapsack_recursive(items[1:], capacity)
    
    # Calculate value if we include the current item
    current_item = items[0]
    include_value, include_items = knapsack_recursive(items[1:], capacity - current_item.weight)
    include_value += current_item.value
    include_items = [current_item] + include_items
    
    # Calculate value if we exclude the current item
    exclude_value, exclude_items = knapsack_recursive(items[1:], capacity)
    
    # Return the better option
    if include_value > exclude_value:
        return include_value, include_items
    else:
        return exclude_value, exclude_items

def knapsack_dynamic(items: List[Item], capacity: float) -> Tuple[float, List[Item]]:
    """
    Dynamic programming solution to the knapsack problem.
    
    Args:
        items: List of items available
        capacity: Maximum weight capacity of the knapsack
        
    Returns:
        Tuple of (total value, list of selected items)
    """
    n = len(items)
    
    # Handle edge cases
    if n == 0 or capacity <= 0:
        return 0, []
    
    # Discretize weights to use as indices
    # Scale to integers to avoid floating point issues
    scale_factor = 10  # Assuming we want 0.1 weight precision
    scaled_capacity = int(capacity * scale_factor)
    scaled_weights = [int(item.weight * scale_factor) for item in items]
    
    # Create DP table
    # dp[i][j] = maximum value with first i items and j capacity
    dp = [[0 for _ in range(scaled_capacity + 1)] for _ in range(n + 1)]
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(scaled_capacity + 1):
            item_index = i - 1
            item_weight = scaled_weights[item_index]
            
            # If current item can't fit, copy value from above
            if item_weight > w:
                dp[i][w] = dp[i-1][w]
            else:
                # Max of including or excluding the item
                dp[i][w] = max(
                    dp[i-1][w],  # exclude
                    dp[i-1][w-item_weight] + items[item_index].value  # include
                )
    
    # Reconstruct the solution
    selected_items = []
    w = scaled_capacity
    
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            # We included this item
            item_index = i - 1
            selected_items.append(items[item_index])
            w -= scaled_weights[item_index]
    
    # Return the total value and selected items
    return dp[n][scaled_capacity], selected_items

def knapsack_greedy(items: List[Item], capacity: float) -> Tuple[float, List[Item]]:
    """
    Greedy approach to the knapsack problem (not optimal).
    
    Args:
        items: List of items available
        capacity: Maximum weight capacity of the knapsack
        
    Returns:
        Tuple of (total value, list of selected items)
    """
    # Sort items by value per weight in descending order
    sorted_items = sorted(items, key=lambda item: item.value_per_weight, reverse=True)
    
    selected_items = []
    total_value = 0
    remaining_capacity = capacity
    
    for item in sorted_items:
        if item.weight <= remaining_capacity:
            selected_items.append(item)
            total_value += item.value
            remaining_capacity -= item.weight
    
    return total_value, selected_items

def visualize_knapsack(capacity: float, all_items: List[Item], 
                      selected_items: List[Item], title: str = "Knapsack Solution"):
    """
    Visualize the knapsack problem solution.
    
    Args:
        capacity: Knapsack capacity
        all_items: All available items
        selected_items: Items selected for the knapsack
        title: Title for the visualization
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Calculate total values and weights
    selected_value = sum(item.value for item in selected_items)
    selected_weight = sum(item.weight for item in selected_items)
    
    # Plot items as scatter points
    plt.subplot(2, 2, 1)
    plt.scatter([item.weight for item in all_items if item not in selected_items], 
                [item.value for item in all_items if item not in selected_items], 
                color='gray', alpha=0.6, s=100, label='Not Selected')
    
    plt.scatter([item.weight for item in selected_items], 
                [item.value for item in selected_items], 
                color='green', s=100, label='Selected')
    
    # Add item names as labels
    for item in all_items:
        plt.annotate(item.name, (item.weight, item.value), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Weight')
    plt.ylabel('Value')
    plt.title('Items by Weight and Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot value per weight
    plt.subplot(2, 2, 2)
    sorted_by_ratio = sorted(all_items, key=lambda item: item.value_per_weight, reverse=True)
    
    bars = plt.bar([item.name for item in sorted_by_ratio], 
                  [item.value_per_weight for item in sorted_by_ratio],
                  color=['green' if item in selected_items else 'gray' for item in sorted_by_ratio])
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Item')
    plt.ylabel('Value per Weight')
    plt.title('Items by Value per Weight Ratio')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot capacity usage
    plt.subplot(2, 2, 3)
    plt.bar(['Capacity', 'Used'], [capacity, selected_weight], 
            color=['blue', 'green'], alpha=0.7)
    plt.xlabel('Knapsack')
    plt.ylabel('Weight')
    plt.title(f'Capacity Usage: {selected_weight}/{capacity} ({selected_weight/capacity*100:.1f}%)')
    
    # Create a table of selected items
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    table_data = [['Item', 'Weight', 'Value', 'Value/Weight']]
    for item in selected_items:
        table_data.append([item.name, f"{item.weight}", f"{item.value}", f"{item.value_per_weight:.2f}"])
    
    table_data.append(['Total', f"{selected_weight}", f"{selected_value}", f"{selected_value/selected_weight:.2f}"])
    
    table = plt.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.3, 0.2, 0.2, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Selected Items')
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Knapsack visualization saved as '{filename}'")
    plt.close()

# -------------------- Longest Common Subsequence/Substring --------------------

def longest_common_subsequence(str1: str, str2: str) -> Tuple[int, str]:
    """
    Find the longest common subsequence between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Tuple of (length of LCS, the LCS itself)
    """
    m, n = len(str1), len(str2)
    
    # Create DP table
    # dp[i][j] = length of LCS of str1[0...i-1] and str2[0...j-1]
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # Reconstruct the LCS
    lcs = []
    i, j = m, n
    
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    # Reverse the LCS
    lcs.reverse()
    
    return dp[m][n], ''.join(lcs)

def longest_common_substring(str1: str, str2: str) -> Tuple[int, str]:
    """
    Find the longest common substring between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Tuple of (length of LCS, the LCS itself)
    """
    m, n = len(str1), len(str2)
    
    # Create DP table
    # dp[i][j] = length of LCS ending at str1[i-1] and str2[j-1]
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Variables to keep track of maximum length and ending position
    max_length = 0
    end_pos = 0
    
    # Fill dp table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos = i
            else:
                dp[i][j] = 0
    
    # Extract the substring
    start_pos = end_pos - max_length
    lcs = str1[start_pos:end_pos]
    
    return max_length, lcs

def visualize_lcs_matrix(str1: str, str2: str, dp_matrix: List[List[int]], 
                       is_subsequence: bool, title: str = "LCS Matrix"):
    """
    Visualize the dynamic programming matrix for LCS.
    
    Args:
        str1: First string
        str2: Second string
        dp_matrix: The DP matrix
        is_subsequence: True if LCS, False if LCSubstring
        title: Title for the visualization
    """
    # Create a DataFrame for better visualization
    df = pd.DataFrame(dp_matrix)
    
    # Add row and column headers
    df.columns = [''] + list(str2)
    df.index = [''] + list(str1)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a table
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    
    # Convert DataFrame to numpy array for displaying
    cell_text = df.values.astype(str)
    
    # Create the table
    table = plt.table(cellText=cell_text, 
                     rowLabels=df.index, 
                     colLabels=df.columns,
                     loc='center', 
                     cellLoc='center')
    
    # Highlight cells based on value
    # For substring, only highlight if value > 0
    # For subsequence, highlight based on value magnitude
    for i in range(len(df)):
        for j in range(len(df.columns)):
            cell = table.get_celld()[(i, j)]
            value = int(cell_text[i][j]) if cell_text[i][j].isdigit() else 0
            
            if is_subsequence:
                # Gradient based on value for subsequence
                if value > 0:
                    intensity = min(1.0, value / max([max(row) for row in dp_matrix]))
                    cell.set_facecolor((0.8 - 0.8 * intensity, 0.9, 0.8 - 0.8 * intensity))
            else:
                # For substring, highlight only positive values
                if value > 0:
                    intensity = min(1.0, value / max([max(row) for row in dp_matrix]))
                    cell.set_facecolor((0.8, 0.9 - 0.9 * intensity, 0.8))
    
    # Adjust table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    # Set title
    plt.title(title, fontsize=16)
    
    # Save the figure
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"LCS matrix visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercise 9.1 - MP3 Player Theft --------------------

def exercise_9_1():
    """
    Exercise 9.1: Extending the knapsack problem with an MP3 player.
    """
    logger.info("\nExercise 9.1: MP3 Player Theft")
    logger.info("-" * 60)
    
    # Create items from the textbook example plus the MP3 player
    items = [
        Item("Stereo", 4, 3000),
        Item("Laptop", 3, 2000),
        Item("Guitar", 1, 1500),
        Item("iPhone", 1, 2000),
        Item("MP3 Player", 1, 1000)  # New item from Exercise 9.1
    ]
    
    # Knapsack capacity
    capacity = 4
    
    logger.info("You're a thief with a knapsack that can hold 4 pounds of goods.")
    logger.info("Available items:")
    for item in items:
        logger.info(f"  {item}")
    
    # Solve using dynamic programming
    total_value, selected_items = knapsack_dynamic(items, capacity)
    
    logger.info("\nOptimal solution:")
    logger.info(f"Total value: ${total_value}")
    logger.info("Selected items:")
    for item in selected_items:
        logger.info(f"  {item}")
    
    # Check if MP3 player is in the solution
    mp3_player_selected = any(item.name == "MP3 Player" for item in selected_items)
    
    logger.info(f"\nShould you steal the MP3 player? {'Yes' if mp3_player_selected else 'No'}")
    
    if mp3_player_selected:
        logger.info("The MP3 player is part of the optimal solution!")
    else:
        logger.info("The MP3 player is not part of the optimal solution.")
    
    # Visualize the solution
    visualize_knapsack(capacity, items, selected_items, "Exercise 9.1 - MP3 Player Theft")
    
    logger.info("\nThe answer to Exercise 9.1 is: Yes, you should steal the MP3 player.")
    logger.info("You could steal the MP3 player, the iPhone, and the guitar, worth a total of $4,500.")

# -------------------- Exercise 9.2 - Camping Trip --------------------

def exercise_9_2():
    """
    Exercise 9.2: Knapsack problem for a camping trip.
    """
    logger.info("\nExercise 9.2: Camping Trip")
    logger.info("-" * 60)
    
    # Create items for the camping trip
    items = [
        Item("Water", 3, 10),
        Item("Book", 1, 3),
        Item("Food", 2, 9),
        Item("Jacket", 2, 5),
        Item("Camera", 1, 6)
    ]
    
    # Knapsack capacity
    capacity = 6
    
    logger.info("You're going camping with a knapsack that can hold 6 pounds of items.")
    logger.info("Available items:")
    for item in items:
        logger.info(f"  {item}")
    
    # Solve using dynamic programming
    total_value, selected_items = knapsack_dynamic(items, capacity)
    
    logger.info("\nOptimal solution:")
    logger.info(f"Total value: {total_value}")
    logger.info("Selected items:")
    for item in selected_items:
        logger.info(f"  {item}")
    
    # Compare with the expected answer
    expected_items = {"Water", "Food", "Camera"}
    actual_items = {item.name for item in selected_items}
    
    logger.info(f"\nExpected items: {', '.join(expected_items)}")
    logger.info(f"Selected items: {', '.join(actual_items)}")
    
    if expected_items == actual_items:
        logger.info("The solution matches the expected answer!")
    else:
        logger.info("The solution does not match the expected answer.")
        
        # If there's a discrepancy, verify by checking all possible combinations
        logger.info("Verifying by checking all possible combinations...")
        
        total_value_brute, selected_items_brute = knapsack_recursive(items, capacity)
        brute_items = {item.name for item in selected_items_brute}
        
        logger.info(f"Brute force solution value: {total_value_brute}")
        logger.info(f"Brute force selected items: {', '.join(brute_items)}")
        
        if total_value == total_value_brute:
            logger.info("The dynamic programming solution is correct!")
        else:
            logger.info("There may be an issue with the dynamic programming solution.")
    
    # Visualize the solution
    visualize_knapsack(capacity, items, selected_items, "Exercise 9.2 - Camping Trip")
    
    logger.info("\nThe answer to Exercise 9.2 is: Take Water, Food, and Camera.")

# -------------------- Exercise 9.3 - Longest Common Substring --------------------

def exercise_9_3():
    """
    Exercise 9.3: Find the longest common substring between "blue" and "clues".
    """
    logger.info("\nExercise 9.3: Longest Common Substring")
    logger.info("-" * 60)
    
    # The two strings
    str1 = "blue"
    str2 = "clues"
    
    logger.info(f"Finding the longest common substring between '{str1}' and '{str2}'")
    
    # Calculate the LCS
    length, substring = longest_common_substring(str1, str2)
    
    logger.info(f"Longest common substring: '{substring}'")
    logger.info(f"Length: {length}")
    
    # Create the DP matrix
    m, n = len(str1), len(str2)
    dp_matrix = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Fill the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp_matrix[i][j] = dp_matrix[i-1][j-1] + 1
    
    # Visualize the matrix
    visualize_lcs_matrix(str1, str2, dp_matrix, False, "Exercise 9.3 - LCS Matrix (blue vs clues)")
    
    # Print the matrix
    logger.info("\nDP Matrix for LCS:")
    logger.info("    " + " ".join(list(" " + str2)))
    for i in range(m + 1):
        row = str1[i-1] if i > 0 else " "
        for j in range(n + 1):
            row += f" {dp_matrix[i][j]}"
        logger.info(row)
    
    logger.info("\nThe answer to Exercise 9.3 is: 'lue' with length 3.")

# -------------------- Main Function --------------------

def main():
    """Main function to run the dynamic programming demonstrations."""
    logger.info("=" * 80)
    logger.info("Dynamic Programming and Knapsack Problem - Chapter 9 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run Exercise 9.1 - MP3 Player Theft
        exercise_9_1()
        
        # Run Exercise 9.2 - Camping Trip
        exercise_9_2()
        
        # Run Exercise 9.3 - Longest Common Substring
        exercise_9_3()
        
        logger.info("\nAll exercises completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()