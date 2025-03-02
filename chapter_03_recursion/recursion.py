"""
Recursion and the Call Stack - Chapter 3 Implementation
=======================================================

This module demonstrates the concepts from Chapter 3 of "Grokking Algorithms":
1. Understanding how recursion works
2. Visualizing the call stack
3. Implementing recursive algorithms
4. Solving classic problems using recursion
5. Base cases and recursive cases

Author: [Your Name]
GitHub: [Your GitHub Username]
"""

import sys
import traceback
import logging
import time
from typing import Any, List, Dict, Optional, Callable
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get recursion limit
RECURSION_LIMIT = sys.getrecursionlimit()
logger.info(f"Current Python recursion limit: {RECURSION_LIMIT}")

# -------------------- Call Stack Visualization --------------------

def visualize_stack(frames: List[Dict], title: str = "Call Stack Visualization") -> None:
    """Visualize a call stack with matplotlib."""
    fig, ax = plt.subplots(figsize=(10, len(frames) * 0.5 + 2))
    
    # Plot frames from bottom to top
    for i, frame in enumerate(frames):
        # Frame rectangle
        rect = plt.Rectangle((0.1, i * 0.5), 0.8, 0.4, fill=True, 
                             color='lightblue', alpha=0.8, 
                             transform=ax.transData)
        ax.add_patch(rect)
        
        # Function name and args
        args_str = ', '.join(f"{k}={v}" for k, v in frame['args'].items())
        func_text = f"{frame['function']}({args_str})"
        ax.text(0.5, i * 0.5 + 0.2, func_text, ha='center', va='center', 
                transform=ax.transData)
    
    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, len(frames) * 0.5 + 0.5)
    ax.set_title(title)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Stack Depth")
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add annotations
    if len(frames) > 1:
        ax.annotate("Top of Stack", xy=(0.9, (len(frames) - 1) * 0.5 + 0.2), 
                    xytext=(1.1, (len(frames) - 1) * 0.5 + 0.2), 
                    arrowprops=dict(facecolor='black', shrink=0.05))
        ax.annotate("Bottom of Stack", xy=(0.9, 0.2), 
                    xytext=(1.1, 0.2), 
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    logger.info(f"Call stack visualization saved as '{title.lower().replace(' ', '_')}.png'")
    plt.close()

# -------------------- Recursion Basics --------------------

def countdown(i: int, stack_frames: List[Dict] = None) -> None:
    """A simple recursive countdown function."""
    # Initialize stack frames if None
    if stack_frames is None:
        stack_frames = []
    
    # Record this function call in the stack
    stack_frames.append({
        'function': 'countdown', 
        'args': {'i': i}
    })
    
    # Visualize the current stack if it's the initial call
    if len(stack_frames) == 1:
        visualize_stack(stack_frames, f"Countdown Start: i={i}")
    
    # Base case
    if i <= 0:
        logger.info("Reached base case!")
        visualize_stack(stack_frames, f"Countdown Base Case: i={i}")
        stack_frames.pop()  # Remove this call from the stack
        return
    
    # Recursive case
    logger.info(f"Counting down: {i}")
    
    # Visualize before the recursive call
    visualize_stack(stack_frames, f"Countdown Before Recursive Call: i={i}")
    
    # Recursive call
    countdown(i - 1, stack_frames)
    
    # This code executes after the recursive calls complete (unwinding the stack)
    logger.info(f"Recursive call completed for i={i}")
    
    # Visualize after the recursive call completes
    visualize_stack(stack_frames, f"Countdown After Recursive Call: i={i}")
    stack_frames.pop()  # Remove this call from the stack

def factorial(n: int, stack_frames: List[Dict] = None) -> int:
    """Calculate factorial recursively and visualize the call stack."""
    # Initialize stack frames if None
    if stack_frames is None:
        stack_frames = []
    
    # Record this function call in the stack
    stack_frames.append({
        'function': 'factorial', 
        'args': {'n': n}
    })
    
    # Visualize the current stack if it's the initial call
    if len(stack_frames) == 1:
        visualize_stack(stack_frames, f"Factorial Start: n={n}")
    
    # Base case
    if n <= 1:
        result = 1
        logger.info(f"Base case: factorial({n}) = {result}")
        visualize_stack(stack_frames, f"Factorial Base Case: n={n}, result={result}")
        stack_frames.pop()  # Remove this call from the stack
        return result
    
    # Recursive case
    logger.info(f"Computing factorial({n})")
    
    # Visualize before the recursive call
    visualize_stack(stack_frames, f"Factorial Before Recursive Call: n={n}")
    
    # Recursive call
    result = n * factorial(n - 1, stack_frames)
    
    # This code executes after the recursive calls complete
    logger.info(f"Recursive call completed: factorial({n}) = {result}")
    
    # Visualize after the recursive call completes
    visualize_stack(stack_frames, f"Factorial After Recursive Call: n={n}, result={result}")
    stack_frames.pop()  # Remove this call from the stack
    
    return result

# -------------------- Demonstrating Call Stack --------------------

def greet(name: str) -> None:
    """A function that demonstrates the call stack by using another function."""
    logger.info(f"Hello, {name}!")
    greet2(name)
    logger.info(f"Getting ready to say goodbye to {name}...")
    bye()
    logger.info("Done with the greet function")

def greet2(name: str) -> None:
    """Helper function for the greet example."""
    logger.info(f"How are you, {name}?")

def bye() -> None:
    """Another helper function for the greet example."""
    logger.info("Ok, bye!")

def demonstrate_call_stack() -> None:
    """Demonstrate a call stack similar to the one in Exercise 3.1."""
    logger.info("\nDemonstration: Call Stack (Exercise 3.1)")
    logger.info("-" * 60)
    
    # Record the call stack in each step
    stack_frames = []
    
    # Initial call to greet
    stack_frames.append({'function': 'greet', 'args': {'name': 'Maggie'}})
    visualize_stack(stack_frames, "Call Stack: Initial Call to greet")
    
    # greet calls greet2
    stack_frames.append({'function': 'greet2', 'args': {'name': 'Maggie'}})
    visualize_stack(stack_frames, "Call Stack: greet calls greet2")
    
    # greet2 returns
    stack_frames.pop()
    visualize_stack(stack_frames, "Call Stack: greet2 returns")
    
    # greet calls bye
    stack_frames.append({'function': 'bye', 'args': {}})
    visualize_stack(stack_frames, "Call Stack: greet calls bye")
    
    # bye returns
    stack_frames.pop()
    visualize_stack(stack_frames, "Call Stack: bye returns")
    
    # greet returns
    stack_frames.pop()
    visualize_stack(stack_frames, "Call Stack: greet returns")
    
    logger.info("\nExercise 3.1 - Information from the call stack:")
    logger.info("- The greet function is called first, with name = Maggie")
    logger.info("- Then the greet function calls the greet2 function, with name = Maggie")
    logger.info("- At this point, the greet function is in an incomplete, suspended state")
    logger.info("- The current function call is the greet2 function")
    logger.info("- After this function call completes, the greet function will resume")

# -------------------- Demonstrating Stack Overflow --------------------

def demonstrate_stack_overflow() -> None:
    """
    Demonstrate a stack overflow (Exercise 3.2).
    This will deliberately cause a RecursionError, but in a controlled way.
    """
    logger.info("\nDemonstration: Stack Overflow (Exercise 3.2)")
    logger.info("-" * 60)
    
    def recursive_function(n: int) -> None:
        """A recursive function without a proper base case."""
        logger.info(f"Recursive call {n}")
        # This will cause a RecursionError eventually
        recursive_function(n + 1)
    
    max_safe_depth = 50  # Much lower than the actual limit to avoid crashing
    stack_frames = [{'function': 'recursive_function', 'args': {'n': i}} 
                   for i in range(max_safe_depth)]
    
    visualize_stack(stack_frames[:10], "Stack Beginning to Grow")
    visualize_stack(stack_frames[:25], "Stack Continues Growing")
    visualize_stack(stack_frames, "Stack About to Overflow")
    
    logger.info("\nIn practice, with a real infinite recursion:")
    logger.info("1. Each recursive call adds a new frame to the call stack")
    logger.info("2. The stack has a limited size (usually a few thousand frames)")
    logger.info(f"3. In Python, the default recursion limit is {RECURSION_LIMIT}")
    logger.info("4. When the stack reaches this limit, you get a RecursionError")
    logger.info("5. This is called a 'stack overflow'")
    
    logger.info("\nWe won't actually run the infinite recursion, but here's a controlled example:")
    
    try:
        # Set a very small recursion limit for demonstration
        original_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max_safe_depth + 10)
        
        # Try the recursive function and catch the error
        recursive_function(1)
    except RecursionError as e:
        logger.info(f"RecursionError caught: {e}")
        logger.info("The stack has overflowed!")
    finally:
        # Restore the original recursion limit
        sys.setrecursionlimit(original_limit)

# -------------------- Classic Recursive Problems --------------------

def sum_array(arr: List[int], stack_frames: List[Dict] = None) -> int:
    """
    Classic recursive problem: Sum an array of integers.
    
    Args:
        arr: List of integers to sum
        stack_frames: List to track the call stack (used for visualization)
        
    Returns:
        Sum of all integers in the array
    """
    # Initialize stack frames if None
    if stack_frames is None:
        stack_frames = []
    
    # Record this function call in the stack
    stack_frames.append({
        'function': 'sum_array', 
        'args': {'arr': arr}
    })
    
    # Visualize the current stack if it's the initial call
    if len(stack_frames) == 1:
        visualize_stack(stack_frames, f"Sum Array Start: arr={arr}")
    
    # Base case
    if len(arr) == 0:
        result = 0
        logger.info(f"Base case: sum_array([]) = {result}")
        visualize_stack(stack_frames, f"Sum Array Base Case: arr=[], result={result}")
        stack_frames.pop()  # Remove this call from the stack
        return result
    
    # Recursive case
    logger.info(f"Computing sum_array({arr})")
    
    # Visualize before the recursive call
    visualize_stack(stack_frames, f"Sum Array Before Recursive Call: arr={arr}")
    
    # Recursive call: first item + sum of the rest
    result = arr[0] + sum_array(arr[1:], stack_frames)
    
    # This code executes after the recursive call completes
    logger.info(f"Recursive call completed: sum_array({arr}) = {result}")
    
    # Visualize after the recursive call completes
    visualize_stack(stack_frames, f"Sum Array After Recursive Call: arr={arr}, result={result}")
    stack_frames.pop()  # Remove this call from the stack
    
    return result

def fibonacci(n: int, stack_frames: List[Dict] = None) -> int:
    """
    Classic recursive problem: Calculate the nth Fibonacci number.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        stack_frames: List to track the call stack (used for visualization)
        
    Returns:
        The nth Fibonacci number
    """
    # Initialize stack frames if None
    if stack_frames is None:
        stack_frames = []
    
    # Record this function call in the stack
    stack_frames.append({
        'function': 'fibonacci', 
        'args': {'n': n}
    })
    
    # Visualize the current stack if it's the initial call
    if len(stack_frames) == 1:
        visualize_stack(stack_frames, f"Fibonacci Start: n={n}")
    
    # Base cases
    if n <= 0:
        result = 0
        logger.info(f"Base case: fibonacci({n}) = {result}")
        visualize_stack(stack_frames, f"Fibonacci Base Case: n={n}, result={result}")
        stack_frames.pop()  # Remove this call from the stack
        return result
    elif n == 1:
        result = 1
        logger.info(f"Base case: fibonacci({n}) = {result}")
        visualize_stack(stack_frames, f"Fibonacci Base Case: n={n}, result={result}")
        stack_frames.pop()  # Remove this call from the stack
        return result
    
    # Recursive case
    logger.info(f"Computing fibonacci({n})")
    
    # Visualize before the recursive calls
    visualize_stack(stack_frames, f"Fibonacci Before Recursive Calls: n={n}")
    
    # First recursive call
    fib_n_1 = fibonacci(n - 1, stack_frames)
    
    # Visualize between recursive calls
    visualize_stack(stack_frames, f"Fibonacci Between Recursive Calls: n={n}")
    
    # Second recursive call
    fib_n_2 = fibonacci(n - 2, stack_frames)
    
    # Combine the results
    result = fib_n_1 + fib_n_2
    
    # This code executes after both recursive calls complete
    logger.info(f"Recursive calls completed: fibonacci({n}) = {result}")
    
    # Visualize after the recursive calls complete
    visualize_stack(stack_frames, f"Fibonacci After Recursive Calls: n={n}, result={result}")
    stack_frames.pop()  # Remove this call from the stack
    
    return result

# -------------------- Optimized Recursive Implementations --------------------

def fibonacci_optimized(n: int, memo: Dict[int, int] = None) -> int:
    """
    An optimized version of the Fibonacci function using memoization.
    
    Args:
        n: The position in the Fibonacci sequence (0-indexed)
        memo: Dictionary to store previously calculated values
        
    Returns:
        The nth Fibonacci number
    """
    # Initialize memoization dictionary if None
    if memo is None:
        memo = {}
    
    # Base cases
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    # Check if we've already calculated this value
    if n in memo:
        return memo[n]
    
    # Recursive case with memoization
    memo[n] = fibonacci_optimized(n - 1, memo) + fibonacci_optimized(n - 2, memo)
    return memo[n]

def compare_fibonacci_implementations() -> None:
    """Compare the performance of optimized and unoptimized Fibonacci implementations."""
    logger.info("\nComparing Fibonacci Implementations:")
    logger.info("-" * 60)
    
    test_values = [5, 10, 15, 20, 25, 30]
    
    results = {
        "n": test_values,
        "naive_time": [],
        "optimized_time": [],
        "naive_result": [],
        "optimized_result": []
    }
    
    for n in test_values:
        # Test naive implementation
        if n <= 20:  # Limit to avoid excessive time for larger values
            start_time = time.time()
            naive_result = fibonacci(n)
            naive_time = time.time() - start_time
            results["naive_result"].append(naive_result)
            results["naive_time"].append(naive_time)
            logger.info(f"Naive fibonacci({n}) = {naive_result}, Time: {naive_time:.6f} seconds")
        else:
            results["naive_result"].append(None)
            results["naive_time"].append(None)
            logger.info(f"Skipping naive fibonacci({n}) - would take too long")
        
        # Test optimized implementation
        start_time = time.time()
        optimized_result = fibonacci_optimized(n)
        optimized_time = time.time() - start_time
        results["optimized_result"].append(optimized_result)
        results["optimized_time"].append(optimized_time)
        logger.info(f"Optimized fibonacci({n}) = {optimized_result}, Time: {optimized_time:.6f} seconds")
    
    # Create a visualization of the performance difference
    plt.figure(figsize=(12, 6))
    
    # Plot only where we have data for both implementations
    valid_indices = [i for i, n in enumerate(test_values) if results["naive_time"][i] is not None]
    
    if valid_indices:
        valid_n = [test_values[i] for i in valid_indices]
        valid_naive_time = [results["naive_time"][i] for i in valid_indices]
        valid_optimized_time = [results["optimized_time"][i] for i in valid_indices]
        
        plt.plot(valid_n, valid_naive_time, 'o-', label='Naive Implementation')
        plt.plot(valid_n, valid_optimized_time, 's-', label='Optimized Implementation')
        
        plt.yscale('log')
        plt.xlabel('n')
        plt.ylabel('Time (seconds, log scale)')
        plt.title('Performance Comparison: Fibonacci Implementations')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig('fibonacci_performance_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("Performance comparison visualization saved as 'fibonacci_performance_comparison.png'")
    
    # Create a separate plot for all optimized times
    plt.figure(figsize=(12, 6))
    plt.plot(test_values, results["optimized_time"], 's-', label='Optimized Implementation')
    plt.xlabel('n')
    plt.ylabel('Time (seconds)')
    plt.title('Performance of Optimized Fibonacci Implementation')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig('fibonacci_optimized_performance.png', dpi=300, bbox_inches='tight')
    logger.info("Optimized performance visualization saved as 'fibonacci_optimized_performance.png'")

# -------------------- Main Function --------------------

def main():
    """Main function to run the recursion demonstrations."""
    logger.info("=" * 80)
    logger.info("Recursion and the Call Stack - Chapter 3 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Basic recursion demonstration with countdown
        logger.info("\nDemonstrating basic recursion with countdown:")
        countdown(3)
        
        # Factorial calculation with call stack visualization
        logger.info("\nCalculating factorial recursively:")
        result = factorial(5)
        logger.info(f"5! = {result}")
        
        # Demonstrate call stack for Exercise 3.1
        demonstrate_call_stack()
        
        # Demonstrate stack overflow for Exercise 3.2
        demonstrate_stack_overflow()
        
        # Classic recursive problems
        logger.info("\nSolving classic recursive problems:")
        
        # Sum array
        arr = [1, 2, 3, 4, 5]
        result = sum_array(arr)
        logger.info(f"Sum of {arr} = {result}")
        
        # Fibonacci
        n = 6
        result = fibonacci(n)
        logger.info(f"Fibonacci({n}) = {result}")
        
        # Compare optimized and unoptimized implementations
        compare_fibonacci_implementations()
        
        logger.info("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()