"""
Binary Search and Big O Notation - Advanced Algorithm Experiments
===============================================================

This module demonstrates concepts from Chapter 1 of "Grokking Algorithms":
1. Binary search implementation with step counting and complexity analysis
2. Big O notation visualization for different algorithms
3. Theoretical vs. practical performance measurement

Author: Petar Matov
GitHub: petar1998

"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # Progress bar for longer operations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Callable
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ------ Binary Search with Step Counting ------

def binary_search(sorted_list: List[Any], item: Any) -> Tuple[int, int]:
    """
    Performs binary search and returns both the index of the found item and the steps taken.
    
    Args:
        sorted_list: A sorted list to search in
        item: The item to search for
        
    Returns:
        Tuple containing (index, steps) where index is the position of the found item
        (-1 if not found) and steps is the number of operations performed
    """
    low = 0
    high = len(sorted_list) - 1
    steps = 0
    
    while low <= high:
        steps += 1
        mid = (low + high) // 2
        guess = sorted_list[mid]
        
        if guess == item:
            return mid, steps
        elif guess > item:
            high = mid - 1
        else:
            low = mid + 1
            
    return -1, steps

# ------ Binary Search Recursive Implementation ------

def binary_search_recursive(sorted_list: List[Any], item: Any) -> Tuple[int, int]:
    """
    Recursive implementation of binary search.
    
    Args:
        sorted_list: A sorted list to search in
        item: The item to search for
        
    Returns:
        Tuple containing (index, steps) where index is the position of the found item
        (-1 if not found) and steps is the number of operations performed
    """
    def _search(low: int, high: int, steps: int = 0) -> Tuple[int, int]:
        # Base case: element not found
        if low > high:
            return -1, steps
            
        steps += 1
        mid = (low + high) // 2
        guess = sorted_list[mid]
        
        if guess == item:
            return mid, steps
        elif guess > item:
            return _search(low, mid - 1, steps)
        else:
            return _search(mid + 1, high, steps)
            
    return _search(0, len(sorted_list) - 1)

# ------ Experiment 1.1 and 1.2: Maximum steps for different list sizes ------

def max_steps_experiment():
    """
    Experiments demonstrating the maximum number of steps in binary search
    for lists of different sizes.
    
    Returns:
        Dictionary with experiment results
    """
    logger.info("Experiment: Maximum Steps in Binary Search")
    logger.info("-" * 60)
    
    # Test with list of 128 elements (Task 1.1)
    size_128 = list(range(128))
    # Search for the last element to get maximum steps
    _, steps_128 = binary_search(size_128, 127)
    
    # Test with list of 256 elements (Task 1.2)
    size_256 = list(range(256))
    _, steps_256 = binary_search(size_256, 255)
    
    logger.info(f"Maximum steps for list with 128 elements: {steps_128}")
    logger.info(f"Maximum steps for list with 256 elements: {steps_256}")
    logger.info(f"Theoretical formula: log₂(n) steps for n elements")
    logger.info(f"log₂(128) = {np.log2(128):.0f}, log₂(256) = {np.log2(256):.0f}")
    
    # Compare iterative vs recursive implementation
    _, steps_128_recursive = binary_search_recursive(size_128, 127)
    _, steps_256_recursive = binary_search_recursive(size_256, 255)
    
    logger.info(f"Recursive implementation - 128 elements: {steps_128_recursive} steps")
    logger.info(f"Recursive implementation - 256 elements: {steps_256_recursive} steps")
    
    # Visualize maximum steps for different list sizes
    sizes = [2**i for i in range(1, 11)]  # Lists with sizes 2, 4, 8, ..., 1024
    max_steps_iterative = []
    max_steps_recursive = []
    theoretical_steps = []
    
    for size in tqdm(sizes, desc="Testing different list sizes"):
        test_list = list(range(size))
        
        # Iterative implementation
        _, steps_iterative = binary_search(test_list, size-1)  # Search last element
        max_steps_iterative.append(steps_iterative)
        
        # Recursive implementation
        _, steps_recursive = binary_search_recursive(test_list, size-1)
        max_steps_recursive.append(steps_recursive)
        
        # Theoretical expectation
        theoretical_steps.append(np.log2(size))
    
    # Create plots
    plt.figure(figsize=(12, 8))
    plt.plot(sizes, max_steps_iterative, 'o-', label='Iterative Implementation')
    plt.plot(sizes, max_steps_recursive, 's-', label='Recursive Implementation')
    plt.plot(sizes, theoretical_steps, '--', label='Theoretical: log₂(n)')
    plt.xscale('log', base=2)
    plt.xlabel('List Size (n)', fontsize=12)
    plt.ylabel('Maximum Number of Steps', fontsize=12)
    plt.title('Maximum Steps in Binary Search by List Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.savefig('binary_search_steps.png', dpi=300, bbox_inches='tight')
    
    logger.info("Plot saved as 'binary_search_steps.png'")
    
    return {
        "list_sizes": sizes,
        "iterative_steps": max_steps_iterative,
        "recursive_steps": max_steps_recursive,
        "theoretical_steps": theoretical_steps
    }

# ------ Phone Book class for Big O demonstrations (Tasks 1.3 - 1.6) ------

@dataclass
class Contact:
    """Class for storing contact information."""
    name: str
    phone_number: str
    email: str = ""
    address: str = ""

class PhoneBook:
    """
    Class simulating a phone book to demonstrate different operations
    and their Big O complexities.
    """
    def __init__(self, size=1000):
        self.size = size
        logger.info(f"Initializing phone book with {size} contacts...")
        
        # Generate random data for the phone book
        self.contacts = [
            Contact(
                name=f"Person_{i}", 
                phone_number=f"555-{random.randint(1000, 9999)}",
                email=f"person_{i}@example.com",
                address=f"{random.randint(1, 999)} Main St, City {random.randint(1, 50)}"
            ) for i in range(size)
        ]
        
        # Sort by name for binary search
        self.contacts.sort(key=lambda contact: contact.name)
        
        # Dictionary for quick lookup by phone number (not used in O(n) demonstrations)
        self.number_to_contact = {contact.phone_number: contact for contact in self.contacts}
        
        # Group contacts by first letter for task 1.6
        self.by_first_letter = {}
        for contact in self.contacts:
            first_letter = contact.name[0].upper()
            if first_letter not in self.by_first_letter:
                self.by_first_letter[first_letter] = []
            self.by_first_letter[first_letter].append(contact)
            
        logger.info("Phone book initialized successfully")
            
    def find_contact_by_name(self, name: str) -> Tuple[Optional[Contact], int, float]:
        """
        Searches for a contact by name using binary search.
        Complexity: O(log n)
        
        Args:
            name: Name to search for
            
        Returns:
            Tuple of (contact, steps, duration) where contact is the found Contact object
            (None if not found), steps is the number of operations, and duration is the
            time taken in seconds
        """
        low = 0
        high = len(self.contacts) - 1
        steps = 0
        
        start_time = time.time()
        
        while low <= high:
            steps += 1
            mid = (low + high) // 2
            if self.contacts[mid].name == name:
                duration = time.time() - start_time
                return self.contacts[mid], steps, duration
            elif self.contacts[mid].name > name:
                high = mid - 1
            else:
                low = mid + 1
                
        duration = time.time() - start_time
        return None, steps, duration
    
    def find_contact_by_number(self, number: str) -> Tuple[Optional[Contact], int, float]:
        """
        Searches for a contact by phone number using linear search.
        Complexity: O(n)
        
        Args:
            number: Phone number to search for
            
        Returns:
            Tuple of (contact, steps, duration) where contact is the found Contact object
            (None if not found), steps is the number of operations, and duration is the
            time taken in seconds
        """
        steps = 0
        start_time = time.time()
        
        for contact in self.contacts:
            steps += 1
            if contact.phone_number == number:
                duration = time.time() - start_time
                return contact, steps, duration
                
        duration = time.time() - start_time
        return None, steps, duration
    
    def read_all_contacts(self) -> Tuple[List[Contact], int, float]:
        """
        Reads all contacts in the phone book.
        Complexity: O(n)
        
        Returns:
            Tuple of (contacts, steps, duration) where contacts is the list of all contacts,
            steps is the number of operations, and duration is the time taken in seconds
        """
        steps = 0
        start_time = time.time()
        
        result = []
        for contact in self.contacts:
            steps += 1
            result.append(contact)
            
        duration = time.time() - start_time
        return result, steps, duration
    
    def read_contacts_by_letter(self, letter: str) -> Tuple[List[Contact], int, float]:
        """
        Reads all contacts starting with a specific letter.
        Even though we only process a subset of entries, complexity remains O(n).
        
        Args:
            letter: First letter to filter by
            
        Returns:
            Tuple of (contacts, steps, duration) where contacts is the list of matching contacts,
            steps is the number of operations, and duration is the time taken in seconds
        """
        steps = 0
        start_time = time.time()
        
        letter = letter.upper()
        if letter not in self.by_first_letter:
            duration = time.time() - start_time
            return [], steps, duration
            
        result = []
        for contact in self.by_first_letter[letter]:
            steps += 1
            result.append(contact)
            
        duration = time.time() - start_time
        return result, steps, duration

def run_big_o_experiments():
    """
    Demonstrates algorithms with different Big O complexities.
    
    Returns:
        Dictionary with experiment results
    """
    logger.info("\nExperiment: Big O Complexity Analysis")
    logger.info("-" * 60)
    
    # Create phone books with different sizes
    sizes = [100, 1000, 10000]
    if time.time() % 2 == 0:  # Add larger size only sometimes to prevent long execution
        sizes.append(50000)
    
    # Store results for each operation and size
    results = {
        "O(log n) - Name Search": {"sizes": sizes, "steps": [], "times": []},
        "O(n) - Number Search": {"sizes": sizes, "steps": [], "times": []},
        "O(n) - Read All Contacts": {"sizes": sizes, "steps": [], "times": []},
        "O(n) - Read Contacts by Letter": {"sizes": sizes, "steps": [], "times": []}
    }
    
    for size in sizes:
        logger.info(f"\nTesting with phone book of {size} contacts:")
        
        # Create a phone book
        phone_book = PhoneBook(size)
        
        # 1.3 Find contact by name - O(log n)
        name_to_find = phone_book.contacts[random.randint(0, size-1)].name
        _, steps, duration = phone_book.find_contact_by_name(name_to_find)
        results["O(log n) - Name Search"]["steps"].append(steps)
        results["O(log n) - Name Search"]["times"].append(duration)
        logger.info(f"  Find contact by name (O(log n)): {steps} steps, {duration:.6f} seconds")
        
        # 1.4 Find contact by number - O(n)
        number_to_find = phone_book.contacts[random.randint(0, size-1)].phone_number
        _, steps, duration = phone_book.find_contact_by_number(number_to_find)
        results["O(n) - Number Search"]["steps"].append(steps)
        results["O(n) - Number Search"]["times"].append(duration)
        logger.info(f"  Find contact by number (O(n)): {steps} steps, {duration:.6f} seconds")
        
        # 1.5 Read all contacts - O(n)
        _, steps, duration = phone_book.read_all_contacts()
        results["O(n) - Read All Contacts"]["steps"].append(steps)
        results["O(n) - Read All Contacts"]["times"].append(duration)
        logger.info(f"  Read all contacts (O(n)): {steps} steps, {duration:.6f} seconds")
        
        # 1.6 Read contacts by letter - O(n)
        letter = "P"  # Look for contacts starting with P (Person_*)
        contacts, steps, duration = phone_book.read_contacts_by_letter(letter)
        results["O(n) - Read Contacts by Letter"]["steps"].append(steps)
        results["O(n) - Read Contacts by Letter"]["times"].append(duration)
        percentage = len(contacts) / size * 100
        logger.info(f"  Read contacts starting with '{letter}' (O(n)): {steps} steps, {duration:.6f} seconds")
        logger.info(f"    Found {len(contacts)} contacts ({percentage:.1f}% of total)")
    
    # Create visualization of results
    create_big_o_visualizations(results)
    
    return results

def create_big_o_visualizations(results: Dict):
    """
    Creates visualizations for Big O complexity results.
    
    Args:
        results: Dictionary with experiment results
    """
    # Create two plots: one for steps and one for execution time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot steps
    for operation, data in results.items():
        sizes = data["sizes"]
        steps = data["steps"]
        
        if "O(log n)" in operation:
            ax1.plot(sizes, steps, 'o-', linewidth=2, markersize=8, label=operation)
            # Add theoretical log(n) curve
            ax1.plot(sizes, [np.log2(s) for s in sizes], '--', 
                   color=ax1.lines[-1].get_color(), alpha=0.5)
        else:
            ax1.plot(sizes, steps, 'o-', linewidth=2, markersize=8, label=operation)
            
            # For O(n) operations, add theoretical n curve
            if "Read Contacts by Letter" in operation:
                # On average, we only read a portion of the data (e.g., 1/26 for letters)
                ax1.plot(sizes, [s * (1/26) for s in sizes], '--', 
                       color=ax1.lines[-1].get_color(), alpha=0.5,
                       label=f"{operation} (theoretical: n/26)")
    
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=10)
    ax1.set_xlabel('Phone Book Size (n)', fontsize=14)
    ax1.set_ylabel('Number of Steps (log scale)', fontsize=14)
    ax1.set_title('Big O Complexity - Operation Steps', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Plot execution times
    for operation, data in results.items():
        sizes = data["sizes"]
        times = data["times"]
        ax2.plot(sizes, times, 'o-', linewidth=2, markersize=8, label=operation)
    
    ax2.set_xscale('log', base=10)
    ax2.set_xlabel('Phone Book Size (n)', fontsize=14)
    ax2.set_ylabel('Execution Time (seconds)', fontsize=14)
    ax2.set_title('Big O Complexity - Execution Time', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('big_o_complexity.png', dpi=300, bbox_inches='tight')
    
    logger.info("\nVisualization saved as 'big_o_complexity.png'")
    
    # Create a third visualization: ratio of operations to list size
    plt.figure(figsize=(12, 8))
    
    # For each operation, plot the ratio of steps to input size
    for operation, data in results.items():
        sizes = data["sizes"]
        steps = data["steps"]
        
        # Calculate ratio of steps to input size
        ratios = []
        if "O(log n)" in operation:
            # For log n operations, divide by log(n)
            ratios = [steps[i] / np.log2(sizes[i]) for i in range(len(sizes))]
            plt.plot(sizes, ratios, 'o-', linewidth=2, markersize=8, 
                   label=f"{operation} - steps/log(n)")
        else:
            # For linear operations, divide by n
            ratios = [steps[i] / sizes[i] for i in range(len(sizes))]
            plt.plot(sizes, ratios, 'o-', linewidth=2, markersize=8, 
                   label=f"{operation} - steps/n")
    
    plt.xscale('log', base=10)
    plt.xlabel('Phone Book Size (n)', fontsize=14)
    plt.ylabel('Ratio (steps/complexity)', fontsize=14)
    plt.title('Normalized Big O Complexity', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('normalized_complexity.png', dpi=300, bbox_inches='tight')
    
    logger.info("Additional visualization saved as 'normalized_complexity.png'")
    
    # Create a summary explanation of Big O notation
    create_big_o_summary_visualization()

def create_big_o_summary_visualization():
    """Creates a visual summary of common Big O complexities."""
    plt.figure(figsize=(14, 10))
    
    # Define range for x-axis
    n = np.arange(1, 101)
    
    # Plot different complexity functions
    plt.plot(n, np.ones_like(n), label='O(1) - Constant')
    plt.plot(n, np.log2(n), label='O(log n) - Logarithmic')
    plt.plot(n, n, label='O(n) - Linear')
    plt.plot(n, n * np.log2(n), label='O(n log n) - Linearithmic')
    plt.plot(n, n**2, label='O(n²) - Quadratic')
    plt.plot(n, 2**n / 10**6, label='O(2ⁿ) - Exponential (scaled)')
    
    # Annotate the plot
    plt.annotate('Binary Search', xy=(80, np.log2(80)), 
               xytext=(60, np.log2(80)+5), 
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Linear Search', xy=(80, 80), 
               xytext=(60, 90), 
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Quick Sort\n(average case)', xy=(80, 80*np.log2(80)), 
               xytext=(40, 80*np.log2(80)+50), 
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Naive String Matching', xy=(50, 50**2), 
               xytext=(20, 50**2+500), 
               arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Add titles and labels
    plt.title('Common Big O Complexities', fontsize=16)
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Operations', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Add text explanation
    text_explanation = """
    Big O Notation Cheat Sheet:
    
    O(1): Constant Time
    - Array access, hash table lookup
    - Performance does not change with input size
    
    O(log n): Logarithmic Time
    - Binary search, balanced search trees
    - Input size is repeatedly divided (e.g., by 2)
    
    O(n): Linear Time
    - Linear search, traversing a list
    - Each element must be processed exactly once
    
    O(n log n): Linearithmic Time
    - Efficient sorting algorithms (merge sort, quick sort)
    - Each element processed, but with logarithmic operations
    
    O(n²): Quadratic Time
    - Nested loops, bubble sort, selection sort
    - For each element, process all elements again
    
    O(2ⁿ): Exponential Time
    - Recursive algorithms without memoization
    - Brute force solutions to NP-complete problems
    """
    
    plt.figtext(0.5, -0.05, text_explanation, ha='center', fontsize=12, 
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('big_o_cheat_sheet.png', dpi=300, bbox_inches='tight', 
              pad_inches=3.0)
    
    logger.info("Big O cheat sheet saved as 'big_o_cheat_sheet.png'")

# ------ Main function to run all experiments ------

def main():
    """Main function to run all experiments."""
    logger.info("=" * 80)
    logger.info("Binary Search and Big O Notation Experiments")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Experiment 1.1 and 1.2: Maximum steps in binary search
        binary_search_results = max_steps_experiment()
        
        # Experiments 1.3 - 1.6: Big O complexities
        big_o_results = run_big_o_experiments()
        
        logger.info("\nAll experiments completed successfully!")
        
        # Save all results to file
        import json
        
        # Convert numpy values to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.float64):
                return float(obj)
            return obj
        
        with open('outputs/experiment_results.json', 'w') as f:
            serializable_results = {
                'binary_search': {k: convert_to_serializable(v) for k, v in binary_search_results.items()},
                'big_o': {k: {k2: convert_to_serializable(v2) for k2, v2 in v.items()} 
                          for k, v in big_o_results.items()}
            }
            json.dump(serializable_results, f, indent=2)
            
        logger.info("Results saved to 'outputs/experiment_results.json'")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
if __name__ == "__main__":
    main()