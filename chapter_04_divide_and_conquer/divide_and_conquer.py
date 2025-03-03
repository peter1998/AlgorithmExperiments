"""
Divide and Conquer - Chapter 4 Implementation
============================================

This module demonstrates the concepts from Chapter 4 of "Grokking Algorithms":
1. Understanding divide and conquer strategy
2. Implementing recursive divide and conquer algorithms
3. Analyzing time complexity of algorithms
4. Solving classic problems using divide and conquer approach

Author: [Your Name]
GitHub: [Your GitHub Username]
"""

import sys
import logging
import time
import random
from typing import List, Any, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get recursion limit for reference
RECURSION_LIMIT = sys.getrecursionlimit()
logger.info(f"Current Python recursion limit: {RECURSION_LIMIT}")

# -------------------- Visualization Utilities --------------------

def visualize_algorithm_comparison(algorithms: List[Tuple[str, List[float]]], 
                                   sizes: List[int],
                                   title: str = "Algorithm Time Complexity Comparison",
                                   log_scale: bool = False) -> None:
    """Visualize performance comparison between different algorithms."""
    plt.figure(figsize=(12, 6))
    
    for name, times in algorithms:
        plt.plot(sizes, times, 'o-', label=name)
    
    if log_scale:
        plt.yscale('log')
    
    plt.xlabel('Input Size (n)')
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the figure
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Comparison visualization saved as '{filename}'")
    plt.close()

def visualize_array_operations(array: List[int], operation_name: str) -> None:
    """Visualize array operations for educational purposes."""
    plt.figure(figsize=(10, 4))
    
    # Original array
    plt.subplot(1, 2, 1)
    plt.bar(range(len(array)), array, color='skyblue')
    plt.title('Original Array')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.xticks(range(len(array)))
    
    # Array after operation
    plt.subplot(1, 2, 2)
    
    if operation_name == "doubling_all":
        # Visualize doubling all elements
        result = [x * 2 for x in array]
        plt.bar(range(len(result)), result, color='lightgreen')
        plt.title('After Doubling All Elements')
    
    elif operation_name == "doubling_first":
        # Visualize doubling just the first element
        result = array.copy()
        if result:
            result[0] *= 2
        plt.bar(range(len(result)), result, color='lightcoral')
        plt.title('After Doubling First Element')
    
    elif operation_name == "multiplication_table":
        # For multiplication table, show a heatmap instead
        plt.subplot(1, 2, 2)
        multiplication_table = [[x * y for y in array] for x in array]
        plt.imshow(multiplication_table, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title('Multiplication Table')
        plt.xlabel('Index')
        plt.ylabel('Index')
        plt.xticks(range(len(array)))
        plt.yticks(range(len(array)), array)
        for i in range(len(array)):
            for j in range(len(array)):
                plt.text(j, i, multiplication_table[i][j], 
                         ha="center", va="center", color="white" if multiplication_table[i][j] > np.mean(multiplication_table) else "black")
    
    else:
        # Default to showing the original array again
        plt.bar(range(len(array)), array, color='skyblue')
        plt.title('Array (No Operation)')
    
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.xticks(range(len(array)))
    
    # Save the figure
    filename = f"array_{operation_name}_visualization.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Array operation visualization saved as '{filename}'")
    plt.close()

# -------------------- Exercise 4.1: Recursive Sum --------------------

def sum_recursive(arr: List[int]) -> int:
    """
    Recursively sum all elements in an array.
    
    This is the implementation for Exercise 4.1.
    
    Args:
        arr: List of integers to sum
        
    Returns:
        Sum of all integers in the array
    """
    # Base case: empty array sums to 0
    if not arr:
        return 0
    
    # Recursive case: first element + sum of rest of array
    return arr[0] + sum_recursive(arr[1:])

def demonstrate_sum_recursive() -> None:
    """Demonstrate the recursive sum function."""
    logger.info("\nDemonstration: Recursive Sum (Exercise 4.1)")
    logger.info("-" * 60)
    
    test_cases = [
        [],
        [5],
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        list(range(10))
    ]
    
    for arr in test_cases:
        result = sum_recursive(arr)
        logger.info(f"sum_recursive({arr}) = {result}")

# -------------------- Exercise 4.2: Count Items Recursively --------------------

def count_recursive(arr: List[Any]) -> int:
    """
    Recursively count the number of items in a list.
    
    This is the implementation for Exercise 4.2.
    
    Args:
        arr: List of items to count
        
    Returns:
        Number of items in the list
    """
    # Base case: empty array has 0 items
    if not arr:
        return 0
    
    # Recursive case: 1 (for current item) + count of rest of array
    return 1 + count_recursive(arr[1:])

def demonstrate_count_recursive() -> None:
    """Demonstrate the recursive count function."""
    logger.info("\nDemonstration: Recursive Count (Exercise 4.2)")
    logger.info("-" * 60)
    
    test_cases = [
        [],
        [5],
        [1, 2, 3, 4, 5],
        ["apple", "banana", "cherry", "date"],
        list(range(10))
    ]
    
    for arr in test_cases:
        result = count_recursive(arr)
        logger.info(f"count_recursive({arr}) = {result}")
        logger.info(f"Built-in len function: len({arr}) = {len(arr)}")

# -------------------- Exercise 4.3: Find Maximum Recursively --------------------

def max_recursive(arr: List[int]) -> Optional[int]:
    """
    Recursively find the maximum value in a list.
    
    This is the implementation for Exercise 4.3.
    
    Args:
        arr: List of integers to find maximum in
        
    Returns:
        Maximum value in the list, or None if the list is empty
    """
    # Edge case: empty array has no maximum
    if not arr:
        return None
    
    # Base case: single-element array's maximum is that element
    if len(arr) == 1:
        return arr[0]
    
    # Base case: two-element array's maximum is the larger element
    if len(arr) == 2:
        return arr[0] if arr[0] > arr[1] else arr[1]
    
    # Recursive case: compare first element with maximum of rest of array
    sub_max = max_recursive(arr[1:])
    return arr[0] if arr[0] > sub_max else sub_max

def demonstrate_max_recursive() -> None:
    """Demonstrate the recursive maximum function."""
    logger.info("\nDemonstration: Recursive Maximum (Exercise 4.3)")
    logger.info("-" * 60)
    
    test_cases = [
        [5],
        [1, 2, 3, 4, 5],
        [5, 2, 9, 1, 7],
        [42, 17, 8, 94, 23, 61],
        list(range(10)),
        [-5, -10, -3, -1, -7]
    ]
    
    for arr in test_cases:
        result = max_recursive(arr)
        logger.info(f"max_recursive({arr}) = {result}")
        logger.info(f"Built-in max function: max({arr}) = {max(arr)}")

# -------------------- Exercise 4.4: Binary Search --------------------

def binary_search_recursive(arr: List[int], target: int, low: int = None, high: int = None) -> Optional[int]:
    """
    Recursive implementation of binary search.
    
    This is the implementation for Exercise 4.4.
    
    Args:
        arr: Sorted list of integers to search in
        target: Value to find
        low: Lower bound of the search range (inclusive)
        high: Upper bound of the search range (inclusive)
        
    Returns:
        Index of the target element if found, None otherwise
    """
    # Initialize low and high for the first call
    if low is None:
        low = 0
    if high is None:
        high = len(arr) - 1
    
    # Base case: empty array or invalid bounds
    if low > high:
        return None
    
    # Calculate middle index
    mid = (low + high) // 2
    
    # Base case: found the target
    if arr[mid] == target:
        return mid
    
    # Recursive cases
    if arr[mid] > target:
        # Search in the left half
        return binary_search_recursive(arr, target, low, mid - 1)
    else:
        # Search in the right half
        return binary_search_recursive(arr, target, mid + 1, high)

def demonstrate_binary_search() -> None:
    """Demonstrate the recursive binary search function."""
    logger.info("\nDemonstration: Recursive Binary Search (Exercise 4.4)")
    logger.info("-" * 60)
    
    # Test with sorted arrays
    test_cases = [
        ([1, 2, 3, 4, 5], 3),
        ([1, 2, 3, 4, 5], 6),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90], 70),
        ([10, 20, 30, 40, 50, 60, 70, 80, 90], 45),
        (list(range(0, 100, 2)), 48),
        (list(range(0, 100, 2)), 49)
    ]
    
    for arr, target in test_cases:
        result = binary_search_recursive(arr, target)
        if result is not None:
            logger.info(f"binary_search_recursive({arr}, {target}) = {result} (Found at index {result})")
        else:
            logger.info(f"binary_search_recursive({arr}, {target}) = {result} (Not found)")

# -------------------- Exercises 4.5-4.8: Big O Analysis --------------------

def print_array(arr: List[int]) -> None:
    """
    Print each element in an array.
    
    Exercise 4.5: Time complexity O(n)
    
    Args:
        arr: List of integers to print
    """
    logger.info(f"Printing array elements (O(n) time complexity):")
    for element in arr:
        logger.info(f"  {element}")

def double_all_elements(arr: List[int]) -> List[int]:
    """
    Double the value of each element in an array.
    
    Exercise 4.6: Time complexity O(n)
    
    Args:
        arr: List of integers to double
        
    Returns:
        New list with all elements doubled
    """
    return [element * 2 for element in arr]

def double_first_element(arr: List[int]) -> List[int]:
    """
    Double the value of just the first element in an array.
    
    Exercise 4.7: Time complexity O(1)
    
    Args:
        arr: List of integers
        
    Returns:
        New list with first element doubled
    """
    if not arr:
        return []
    
    result = arr.copy()
    result[0] *= 2
    return result

def create_multiplication_table(arr: List[int]) -> List[List[int]]:
    """
    Create a multiplication table with all elements in the array.
    
    Exercise 4.8: Time complexity O(n²)
    
    Args:
        arr: List of integers
        
    Returns:
        2D list representing the multiplication table
    """
    return [[x * y for y in arr] for x in arr]

def demonstrate_big_o_examples() -> None:
    """Demonstrate the Big O examples from Exercises 4.5-4.8."""
    logger.info("\nDemonstration: Big O Examples (Exercises 4.5-4.8)")
    logger.info("-" * 60)
    
    # Example array
    arr = [2, 3, 7, 8, 10]
    logger.info(f"Example array: {arr}")
    
    # Exercise 4.5: O(n) - Print array
    logger.info("\nExercise 4.5: Printing array elements (O(n))")
    print_array(arr)
    
    # Exercise 4.6: O(n) - Double all elements
    logger.info("\nExercise 4.6: Doubling all elements (O(n))")
    doubled_all = double_all_elements(arr)
    logger.info(f"Original array: {arr}")
    logger.info(f"After doubling all elements: {doubled_all}")
    visualize_array_operations(arr, "doubling_all")
    
    # Exercise 4.7: O(1) - Double first element
    logger.info("\nExercise 4.7: Doubling first element (O(1))")
    doubled_first = double_first_element(arr)
    logger.info(f"Original array: {arr}")
    logger.info(f"After doubling first element: {doubled_first}")
    visualize_array_operations(arr, "doubling_first")
    
    # Exercise 4.8: O(n²) - Multiplication table
    logger.info("\nExercise 4.8: Creating multiplication table (O(n²))")
    mult_table = create_multiplication_table(arr)
    logger.info(f"Original array: {arr}")
    logger.info("Multiplication table:")
    for row in mult_table:
        logger.info(f"  {row}")
    visualize_array_operations(arr, "multiplication_table")

# -------------------- Performance Testing --------------------

def measure_execution_time(func, *args, **kwargs) -> float:
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Execution time in seconds
    """
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

def compare_sum_implementations() -> None:
    """Compare recursive and iterative sum implementations."""
    logger.info("\nComparing Sum Implementations:")
    logger.info("-" * 60)
    
    def sum_iterative(arr):
        """Iterative implementation of sum."""
        total = 0
        for element in arr:
            total += element
        return total
    
    # Generate test data of various sizes
    sizes = [10, 100, 500, 1000]
    results = {
        "sizes": sizes,
        "recursive_times": [],
        "iterative_times": []
    }
    
    for size in sizes:
        # Generate random array
        arr = [random.randint(1, 100) for _ in range(size)]
        
        # Measure recursive sum time
        recursive_time = measure_execution_time(sum_recursive, arr)
        results["recursive_times"].append(recursive_time)
        
        # Measure iterative sum time
        iterative_time = measure_execution_time(sum_iterative, arr)
        results["iterative_times"].append(iterative_time)
        
        logger.info(f"Size: {size}, Recursive: {recursive_time:.6f}s, Iterative: {iterative_time:.6f}s")
    
    # Visualize comparison
    algorithms = [
        ("Recursive Sum", results["recursive_times"]),
        ("Iterative Sum", results["iterative_times"])
    ]
    visualize_algorithm_comparison(algorithms, sizes, "Sum Implementation Comparison", log_scale=True)

def demonstrate_time_complexity() -> None:
    """Demonstrate the time complexity of different operations."""
    logger.info("\nDemonstrating Time Complexity:")
    logger.info("-" * 60)
    
    # Generate test data of various sizes
    sizes = [10, 100, 500, 1000, 5000]
    results = {
        "sizes": sizes,
        "o_1_times": [],
        "o_n_times": [],
        "o_n2_times": []
    }
    
    for size in sizes:
        # Generate random array
        arr = [random.randint(1, 100) for _ in range(size)]
        
        # Measure O(1) operation time
        o_1_time = measure_execution_time(double_first_element, arr)
        results["o_1_times"].append(o_1_time)
        
        # Measure O(n) operation time
        o_n_time = measure_execution_time(double_all_elements, arr)
        results["o_n_times"].append(o_n_time)
        
        # Measure O(n²) operation time (limit size for large arrays)
        if size <= 1000:
            o_n2_time = measure_execution_time(create_multiplication_table, arr)
            results["o_n2_times"].append(o_n2_time)
        else:
            sample = arr[:1000]  # Use a smaller sample for large arrays
            o_n2_time = measure_execution_time(create_multiplication_table, sample)
            results["o_n2_times"].append(o_n2_time)
        
        logger.info(f"Size: {size}, O(1): {o_1_time:.6f}s, O(n): {o_n_time:.6f}s, O(n²): {o_n2_time:.6f}s")
    
    # Visualize comparison
    algorithms = [
        ("O(1) - Double First Element", results["o_1_times"]),
        ("O(n) - Double All Elements", results["o_n_times"]),
        ("O(n²) - Multiplication Table", results["o_n2_times"])
    ]
    visualize_algorithm_comparison(algorithms, sizes, "Time Complexity Comparison", log_scale=True)

# -------------------- Main Function --------------------

def main():
    """Main function to run the divide and conquer demonstrations."""
    logger.info("=" * 80)
    logger.info("Divide and Conquer - Chapter 4 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Exercise 4.1: Recursive Sum
        demonstrate_sum_recursive()
        
        # Exercise 4.2: Recursive Count
        demonstrate_count_recursive()
        
        # Exercise 4.3: Recursive Maximum
        demonstrate_max_recursive()
        
        # Exercise 4.4: Recursive Binary Search
        demonstrate_binary_search()
        
        # Exercises 4.5-4.8: Big O Examples
        demonstrate_big_o_examples()
        
        # Additional demonstrations
        compare_sum_implementations()
        demonstrate_time_complexity()
        
        logger.info("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()