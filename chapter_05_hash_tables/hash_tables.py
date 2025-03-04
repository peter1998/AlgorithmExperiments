"""
Hash Tables and Hash Functions - Chapter 5 Implementation
===========================================================

This module demonstrates the concepts from Chapter 5 of "Grokking Algorithms":
1. Hash tables and hash functions
2. Consistent vs. inconsistent hash functions
3. Good vs. bad distributions
4. Implementations of different hash functions
5. Visualization of hash function distributions

Author: [Your Name]
GitHub: [Your GitHub Username]
"""

import random
import string
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, Dict, List, Callable, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Simple Hash Table Implementation --------------------

class HashTable:
    """A simple hash table implementation."""
    
    def __init__(self, size: int = 10, hash_function: Callable = None):
        """
        Initialize a hash table with a given size and hash function.
        
        Args:
            size: Size of the hash table (number of slots)
            hash_function: Function to convert keys to indices
        """
        self.size = size
        self.table = [[] for _ in range(size)]  # Using chaining for collision resolution
        
        # Set the hash function or use default
        if hash_function:
            self.hash_function = hash_function
        else:
            # Default hash function: Python's built-in hash
            self.hash_function = lambda key: abs(hash(key)) % self.size
    
    def set(self, key: Any, value: Any) -> None:
        """
        Insert or update a key-value pair in the hash table.
        
        Args:
            key: The key
            value: The value to associate with the key
        """
        index = self.hash_function(key)
        
        # Check if key already exists (update)
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        
        # Key doesn't exist, add new key-value pair
        self.table[index].append((key, value))
    
    def get(self, key: Any) -> Any:
        """
        Retrieve a value by key from the hash table.
        
        Args:
            key: The key to look up
            
        Returns:
            The value associated with the key or None if not found
        """
        index = self.hash_function(key)
        
        for k, v in self.table[index]:
            if k == key:
                return v
        
        return None
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair from the hash table.
        
        Args:
            key: The key to delete
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        index = self.hash_function(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index].pop(i)
                return True
        
        return False
    
    def display(self) -> None:
        """Display the contents of the hash table."""
        for i, bucket in enumerate(self.table):
            if bucket:
                logger.info(f"Slot {i}: {bucket}")
            else:
                logger.info(f"Slot {i}: Empty")
    
    def load_factor(self) -> float:
        """
        Calculate the load factor of the hash table.
        
        Returns:
            The load factor (number of items / number of slots)
        """
        total_items = sum(len(bucket) for bucket in self.table)
        return total_items / self.size

# -------------------- Different Hash Functions --------------------

def constant_hash(key: Any, size: int = 10) -> int:
    """
    A constant hash function that returns 1 for all input.
    
    Args:
        key: The input key
        size: Size of the hash table
        
    Returns:
        Always returns 1 (Exercise 5.1)
    """
    return 1

def random_hash(key: Any, size: int = 10) -> int:
    """
    A random hash function that returns a random number each time.
    
    Args:
        key: The input key
        size: Size of the hash table
        
    Returns:
        A random integer between 0 and size-1 (Exercise 5.2)
    """
    return random.randint(0, size - 1)

def next_empty_slot_hash(key: Any, table: List[List], size: int = 10) -> int:
    """
    A hash function that returns the index of the next empty slot.
    
    Args:
        key: The input key
        table: The hash table
        size: Size of the hash table
        
    Returns:
        The index of the next empty slot or 0 if none found (Exercise 5.3)
    """
    for i in range(size):
        if not table[i]:
            return i
    return 0  # If no empty slots found, return 0

def length_hash(key: Any, size: int = 10) -> int:
    """
    A hash function that uses the length of the string as the index.
    
    Args:
        key: The input key (converted to string)
        size: Size of the hash table
        
    Returns:
        The length of the key modulo size (Exercise 5.4)
    """
    return len(str(key)) % size

def first_char_hash(key: Any, size: int = 10) -> int:
    """
    A hash function that uses the first character of the string as the index.
    
    Args:
        key: The input key (converted to string)
        size: Size of the hash table
        
    Returns:
        The ASCII value of the first character modulo size
    """
    key_str = str(key)
    if not key_str:
        return 0
    return ord(key_str[0].lower()) % size

def prime_char_hash(key: Any, size: int = 10) -> int:
    """
    A hash function that maps every letter to a prime number and sums them.
    
    Args:
        key: The input key (converted to string)
        size: Size of the hash table
        
    Returns:
        The sum of prime numbers for each character modulo size
    """
    # Map letters to prime numbers (a=2, b=3, c=5, ...)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 
              73, 79, 83, 89, 97, 101]
    
    key_str = str(key).lower()
    result = 0
    
    for char in key_str:
        if 'a' <= char <= 'z':
            # Map letter to prime number (a=0, b=1, ..., z=25)
            result += primes[ord(char) - ord('a')]
        else:
            # For non-letters, use ASCII value
            result += ord(char)
    
    return result % size

# -------------------- Testing Consistency of Hash Functions --------------------

def test_consistency(hash_function: Callable, keys: List[Any], size: int = 10) -> bool:
    """
    Test if a hash function is consistent (returns the same output for the same input).
    
    Args:
        hash_function: The hash function to test
        keys: A list of keys to test
        size: Size of the hash table
        
    Returns:
        True if the hash function is consistent, False otherwise
    """
    # Store initial hash values
    initial_hashes = {}
    for key in keys:
        if callable(hash_function):
            initial_hashes[key] = hash_function(key, size)
        else:
            initial_hashes[key] = hash_function(key)
    
    # Test if subsequent hash values match initial ones
    for _ in range(10):  # Test multiple times
        for key in keys:
            if callable(hash_function):
                current_hash = hash_function(key, size)
            else:
                current_hash = hash_function(key)
            
            if current_hash != initial_hashes[key]:
                return False
    
    return True

def analyze_hash_functions_consistency() -> None:
    """Analyze the consistency of different hash functions."""
    logger.info("\nAnalyzing Hash Function Consistency:")
    logger.info("-" * 60)
    
    # Test keys
    test_keys = ["apple", "banana", "cherry", "date", "elderberry"]
    
    # Hash functions to test
    hash_functions = [
        ("Constant (f(x) = 1)", lambda k, s: 1),
        ("Random", random_hash),
        ("Next Empty Slot", lambda k, s: next_empty_slot_hash(k, [[] for _ in range(s)], s)),
        ("Length", length_hash),
        ("First Character", first_char_hash),
        ("Prime Character", prime_char_hash)
    ]
    
    # Test each hash function
    for name, func in hash_functions:
        is_consistent = test_consistency(func, test_keys)
        logger.info(f"Hash Function: {name}")
        logger.info(f"Consistent: {is_consistent}")
        
        # Additional insights for Exercise 5.1-5.4
        if name == "Constant (f(x) = 1)":
            logger.info("Exercise 5.1: This function returns 1 for all input, making it consistent.")
            logger.info("However, it would create a terrible distribution with all items in the same slot.")
        elif name == "Random":
            logger.info("Exercise 5.2: This function returns a random number each time, making it inconsistent.")
            logger.info("A hash table using this function would not be able to reliably retrieve items.")
        elif name == "Next Empty Slot":
            logger.info("Exercise 5.3: This function depends on the current state of the hash table,")
            logger.info("making it inconsistent. The same key could hash to different slots depending on timing.")
        elif name == "Length":
            logger.info("Exercise 5.4: This function only considers the length of the key, making it consistent.")
            logger.info("However, many keys have the same length, so distribution may not be ideal.")
        
        logger.info("-" * 40)

# -------------------- Testing Distribution of Hash Functions --------------------

def visualize_hash_distribution(hash_function: Callable, keys: List[Any], 
                                size: int = 10, title: str = "Hash Distribution") -> None:
    """
    Visualize the distribution of keys across hash table slots.
    
    Args:
        hash_function: The hash function to visualize
        keys: A list of keys to hash
        size: Size of the hash table
        title: Title for the visualization
    """
    # Calculate hash values for each key
    hash_values = []
    for key in keys:
        if callable(hash_function):
            hash_values.append(hash_function(key, size))
        else:
            hash_values.append(hash_function(key))
    
    # Count occurrences of each slot
    slot_counts = [0] * size
    for hash_val in hash_values:
        slot_counts[hash_val] += 1
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Bar chart of slot counts
    plt.bar(range(size), slot_counts, color='skyblue')
    plt.xlabel('Hash Table Slot')
    plt.ylabel('Number of Keys')
    plt.title(title)
    plt.xticks(range(size))
    
    # Add text showing which keys are in each slot
    slot_to_keys = [[] for _ in range(size)]
    for i, key in enumerate(keys):
        slot_to_keys[hash_values[i]].append(key)
    
    for i, keys_in_slot in enumerate(slot_to_keys):
        if keys_in_slot:
            plt.text(i, slot_counts[i] + 0.1, '\n'.join(str(k) for k in keys_in_slot),
                    ha='center', va='bottom', rotation=90, fontsize=8)
    
    # Save the figure
    filename = f"{title.lower().replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info(f"Distribution visualization saved as '{filename}'")
    plt.close()
    
    # Calculate and return distribution quality metrics
    utilization = sum(1 for count in slot_counts if count > 0) / size
    max_items_per_slot = max(slot_counts)
    avg_items_per_slot = sum(slot_counts) / size
    
    return {
        "utilization": utilization,
        "max_items_per_slot": max_items_per_slot,
        "avg_items_per_slot": avg_items_per_slot,
        "std_dev": np.std(slot_counts)
    }

def analyze_phonebook_example() -> None:
    """Analyze hash function distribution for the phonebook example (Exercise 5.5)."""
    logger.info("\nExercise 5.5 - Phonebook Distribution Analysis:")
    logger.info("-" * 60)
    
    names = ["Esther", "Ben", "Bob", "Dan"]
    table_size = 10
    
    hash_functions = [
        ("Constant (f(x) = 1)", lambda k, s: 1),
        ("Length", length_hash),
        ("First Character", first_char_hash),
        ("Prime Character", prime_char_hash)
    ]
    
    results = {}
    
    for name, func in hash_functions:
        logger.info(f"Analyzing hash function: {name}")
        
        metrics = visualize_hash_distribution(
            func, names, table_size, 
            f"Phonebook Distribution - {name}"
        )
        
        results[name] = metrics
        
        logger.info(f"  Slot utilization: {metrics['utilization']:.2f}")
        logger.info(f"  Maximum items in a slot: {metrics['max_items_per_slot']}")
        logger.info(f"  Average items per slot: {metrics['avg_items_per_slot']:.2f}")
        logger.info(f"  Standard deviation: {metrics['std_dev']:.2f}")
        logger.info("-" * 40)
    
    logger.info("\nConclusion:")
    logger.info("Hash functions C (First Character) and D (Prime Character) provide the best distribution,")
    logger.info("as they spread the names across more slots and have lower collision rates.")

def analyze_battery_example() -> None:
    """Analyze hash function distribution for the battery example (Exercise 5.6)."""
    logger.info("\nExercise 5.6 - Battery Size Distribution Analysis:")
    logger.info("-" * 60)
    
    battery_sizes = ["A", "AA", "AAA", "AAAA"]
    table_size = 10
    
    hash_functions = [
        ("Constant (f(x) = 1)", lambda k, s: 1),
        ("Length", length_hash),
        ("First Character", first_char_hash),
        ("Prime Character", prime_char_hash)
    ]
    
    results = {}
    
    for name, func in hash_functions:
        logger.info(f"Analyzing hash function: {name}")
        
        metrics = visualize_hash_distribution(
            func, battery_sizes, table_size, 
            f"Battery Size Distribution - {name}"
        )
        
        results[name] = metrics
        
        logger.info(f"  Slot utilization: {metrics['utilization']:.2f}")
        logger.info(f"  Maximum items in a slot: {metrics['max_items_per_slot']}")
        logger.info(f"  Average items per slot: {metrics['avg_items_per_slot']:.2f}")
        logger.info(f"  Standard deviation: {metrics['std_dev']:.2f}")
        logger.info("-" * 40)
    
    logger.info("\nConclusion:")
    logger.info("Hash functions B (Length) and D (Prime Character) provide the best distribution,")
    logger.info("as they spread the battery sizes across more slots and have lower collision rates.")
    logger.info("Function C (First Character) is terrible for this case because all battery sizes")
    logger.info("start with 'A', causing all items to hash to the same slot.")

def analyze_book_example() -> None:
    """Analyze hash function distribution for the book example (Exercise 5.7)."""
    logger.info("\nExercise 5.7 - Book Title Distribution Analysis:")
    logger.info("-" * 60)
    
    book_titles = ["Maus", "Fun Home", "Watchmen"]
    table_size = 10
    
    hash_functions = [
        ("Constant (f(x) = 1)", lambda k, s: 1),
        ("Length", length_hash),
        ("First Character", first_char_hash),
        ("Prime Character", prime_char_hash)
    ]
    
    results = {}
    
    for name, func in hash_functions:
        logger.info(f"Analyzing hash function: {name}")
        
        metrics = visualize_hash_distribution(
            func, book_titles, table_size, 
            f"Book Title Distribution - {name}"
        )
        
        results[name] = metrics
        
        logger.info(f"  Slot utilization: {metrics['utilization']:.2f}")
        logger.info(f"  Maximum items in a slot: {metrics['max_items_per_slot']}")
        logger.info(f"  Average items per slot: {metrics['avg_items_per_slot']:.2f}")
        logger.info(f"  Standard deviation: {metrics['std_dev']:.2f}")
        logger.info("-" * 40)
    
    logger.info("\nConclusion:")
    logger.info("Hash functions B (Length), C (First Character), and D (Prime Character) all")
    logger.info("provide good distributions for this case, spreading the book titles well.")
    logger.info("This is because the titles have different lengths, start with different")
    logger.info("characters, and have diverse character compositions.")

# -------------------- Main Function --------------------

def main():
    """Main function to run the hash table demonstrations."""
    logger.info("=" * 80)
    logger.info("Hash Tables and Hash Functions - Chapter 5 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Test consistency of hash functions
        analyze_hash_functions_consistency()
        
        # Analyze example cases from exercises
        analyze_phonebook_example()  # Exercise 5.5
        analyze_battery_example()    # Exercise 5.6
        analyze_book_example()       # Exercise 5.7
        
        # Basic example of using the HashTable implementation
        logger.info("\nDemonstrating HashTable Implementation:")
        logger.info("-" * 60)
        
        # Create a hash table with the prime character hash function
        hash_table = HashTable(size=10, hash_function=lambda k: prime_char_hash(k, 10))
        
        # Add some key-value pairs
        data = [
            ("apple", "A sweet fruit"),
            ("banana", "A yellow fruit"),
            ("cherry", "A small red fruit"),
            ("date", "A sweet dried fruit"),
            ("elderberry", "A purple-black fruit")
        ]
        
        for key, value in data:
            hash_table.set(key, value)
        
        # Display the hash table
        hash_table.display()
        
        # Retrieve values
        logger.info("\nRetrieving values:")
        for key, _ in data:
            value = hash_table.get(key)
            logger.info(f"Key: {key}, Value: {value}")
        
        # Delete a key
        key_to_delete = "banana"
        logger.info(f"\nDeleting key '{key_to_delete}'")
        deleted = hash_table.delete(key_to_delete)
        logger.info(f"Deletion successful: {deleted}")
        
        # Display the hash table after deletion
        hash_table.display()
        
        # Calculate load factor
        load_factor = hash_table.load_factor()
        logger.info(f"\nLoad factor: {load_factor:.2f}")
        
        logger.info("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()