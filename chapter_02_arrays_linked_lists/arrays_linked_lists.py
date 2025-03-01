"""
Arrays vs Linked Lists - Implementation and Exercises
=====================================================

This module demonstrates the concepts from Chapter 2 of "Grokking Algorithms":
1. Comparison between arrays and linked lists
2. Implementation of both data structures
3. Performance analysis for different operations
4. Solutions to the Chapter 2 exercises

Author: Petar Matov
GitHub: petar1998
"""

import time
import random
import matplotlib.pyplot as plt
import numpy as np
from typing import Any, List, Optional, Union, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
random.seed(42)

# -------------------- Linked List Implementation --------------------

class Node:
    """A node in a linked list."""
    def __init__(self, data: Any):
        self.data = data
        self.next = None
    
    def __repr__(self) -> str:
        return f"Node({self.data})"

class LinkedList:
    """Implementation of a singly linked list."""
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def append(self, data: Any) -> None:
        """Add an element to the end of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node
        self.size += 1
    
    def prepend(self, data: Any) -> None:
        """Add an element to the beginning of the list."""
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node
        self.size += 1
    
    def insert_at(self, index: int, data: Any) -> bool:
        """Insert an element at a specific position."""
        if index < 0 or index > self.size:
            return False
        
        if index == 0:
            self.prepend(data)
            return True
        
        if index == self.size:
            self.append(data)
            return True
        
        new_node = Node(data)
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        new_node.next = current.next
        current.next = new_node
        self.size += 1
        return True
    
    def delete_at(self, index: int) -> bool:
        """Delete an element at a specific position."""
        if index < 0 or index >= self.size or self.head is None:
            return False
        
        if index == 0:
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return True
        
        current = self.head
        for _ in range(index - 1):
            current = current.next
        
        if current.next == self.tail:
            self.tail = current
        
        current.next = current.next.next
        self.size -= 1
        return True
    
    def get(self, index: int) -> Optional[Any]:
        """Get element at a specific position."""
        if index < 0 or index >= self.size or self.head is None:
            return None
        
        current = self.head
        for _ in range(index):
            current = current.next
        
        return current.data
    
    def search(self, data: Any) -> int:
        """Search for an element and return its index or -1 if not found."""
        current = self.head
        index = 0
        
        while current:
            if current.data == data:
                return index
            current = current.next
            index += 1
        
        return -1
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.data))
            current = current.next
        return "[" + " -> ".join(values) + "]"

# -------------------- Dynamic Array Implementation --------------------

class DynamicArray:
    """Implementation of a dynamic array (similar to Python list)."""
    def __init__(self):
        self.capacity = 1
        self.size = 0
        self.array = [None] * self.capacity
    
    def append(self, data: Any) -> None:
        """Add an element to the end of the array."""
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        self.array[self.size] = data
        self.size += 1
    
    def prepend(self, data: Any) -> None:
        """Add an element to the beginning of the array."""
        self.insert_at(0, data)
    
    def insert_at(self, index: int, data: Any) -> bool:
        """Insert an element at a specific position."""
        if index < 0 or index > self.size:
            return False
        
        if self.size == self.capacity:
            self._resize(2 * self.capacity)
        
        # Shift elements to the right
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]
        
        self.array[index] = data
        self.size += 1
        return True
    
    def delete_at(self, index: int) -> bool:
        """Delete an element at a specific position."""
        if index < 0 or index >= self.size:
            return False
        
        # Shift elements to the left
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]
        
        self.array[self.size - 1] = None
        self.size -= 1
        
        # Shrink if necessary
        if self.size < self.capacity // 4 and self.capacity > 1:
            self._resize(self.capacity // 2)
            
        return True
    
    def get(self, index: int) -> Optional[Any]:
        """Get element at a specific position (random access)."""
        if index < 0 or index >= self.size:
            return None
        return self.array[index]
    
    def search(self, data: Any) -> int:
        """Search for an element and return its index or -1 if not found."""
        for i in range(self.size):
            if self.array[i] == data:
                return i
        return -1
    
    def _resize(self, new_capacity: int) -> None:
        """Resize the internal array."""
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity
    
    def __len__(self) -> int:
        return self.size
    
    def __repr__(self) -> str:
        return "[" + ", ".join(str(self.array[i]) for i in range(self.size)) + "]"

# -------------------- Performance Comparison --------------------

def compare_operations(sizes: List[int], operations: List[str], num_trials: int = 5) -> Dict:
    """
    Compare the performance of arrays and linked lists for different operations.
    
    Args:
        sizes: List of data structure sizes to test
        operations: List of operations to test ('append', 'prepend', 'insert_middle', 
                  'delete_middle', 'get_random', 'search')
        num_trials: Number of trials for each test to average results
        
    Returns:
        Dictionary with performance results
    """
    results = {
        "array": {op: [] for op in operations},
        "linked_list": {op: [] for op in operations}
    }
    
    for size in sizes:
        logger.info(f"Testing with data structure size: {size}")
        
        for operation in operations:
            # Test dynamic array
            array_times = []
            for _ in range(num_trials):
                array = DynamicArray()
                # Prefill the data structure
                for i in range(size):
                    array.append(i)
                
                # Measure operation time
                start_time = time.time()
                
                if operation == "append":
                    array.append(size)
                elif operation == "prepend":
                    array.prepend(-1)
                elif operation == "insert_middle":
                    array.insert_at(size // 2, -1)
                elif operation == "delete_middle":
                    array.delete_at(size // 2)
                elif operation == "get_random":
                    for _ in range(100):  # Do multiple gets to measure accurately
                        array.get(random.randint(0, size - 1))
                elif operation == "search":
                    array.search(random.randint(0, size - 1))
                
                end_time = time.time()
                array_times.append(end_time - start_time)
            
            # Average the times
            avg_array_time = sum(array_times) / len(array_times)
            results["array"][operation].append(avg_array_time)
            
            # Test linked list
            ll_times = []
            for _ in range(num_trials):
                ll = LinkedList()
                # Prefill the data structure
                for i in range(size):
                    ll.append(i)
                
                # Measure operation time
                start_time = time.time()
                
                if operation == "append":
                    ll.append(size)
                elif operation == "prepend":
                    ll.prepend(-1)
                elif operation == "insert_middle":
                    ll.insert_at(size // 2, -1)
                elif operation == "delete_middle":
                    ll.delete_at(size // 2)
                elif operation == "get_random":
                    for _ in range(100):  # Do multiple gets to measure accurately
                        ll.get(random.randint(0, size - 1))
                elif operation == "search":
                    ll.search(random.randint(0, size - 1))
                
                end_time = time.time()
                ll_times.append(end_time - start_time)
            
            # Average the times
            avg_ll_time = sum(ll_times) / len(ll_times)
            results["linked_list"][operation].append(avg_ll_time)
            
            logger.info(f"  {operation}: Array = {avg_array_time:.6f}s, Linked List = {avg_ll_time:.6f}s")
    
    return results

def plot_comparison(sizes: List[int], results: Dict) -> None:
    """
    Plot the performance comparison between arrays and linked lists.
    
    Args:
        sizes: List of data structure sizes
        results: Dictionary with performance results
    """
    operations = list(results["array"].keys())
    num_ops = len(operations)
    
    # Create subplots
    fig, axes = plt.subplots(nrows=(num_ops + 1) // 2, ncols=2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, operation in enumerate(operations):
        ax = axes[i]
        
        # Plot array performance
        ax.plot(sizes, results["array"][operation], 'o-', label='Dynamic Array')
        
        # Plot linked list performance
        ax.plot(sizes, results["linked_list"][operation], 's-', label='Linked List')
        
        ax.set_title(f"{operation} Performance")
        ax.set_xlabel("Data Structure Size")
        ax.set_ylabel("Time (seconds)")
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide any unused subplots
    for j in range(num_ops, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig('array_vs_linkedlist_performance.png', dpi=300, bbox_inches='tight')
    logger.info("Performance comparison plot saved as 'array_vs_linkedlist_performance.png'")

    # Create a second plot to show the Big O relationships
    plt.figure(figsize=(10, 8))
    
    n = np.arange(1, 101)
    plt.plot(n, np.ones_like(n), label='O(1) - Constant')
    plt.plot(n, np.log2(n), label='O(log n) - Logarithmic')
    plt.plot(n, n, label='O(n) - Linear')
    plt.plot(n, n * np.log2(n), label='O(n log n) - Linearithmic')
    plt.plot(n, n**2, label='O(n²) - Quadratic')
    
    # Annotate with common operations
    plt.annotate('Array: Random Access', xy=(80, 1), 
                xytext=(60, 5), 
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Array: Append (amortized)', xy=(80, 1), 
                xytext=(30, 8), 
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('LinkedList: Insert at beginning/end', xy=(80, 1), 
                xytext=(30, 15), 
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('Array: Insert in middle', xy=(50, 50), 
                xytext=(15, 60), 
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate('LinkedList: Random Access', xy=(50, 50), 
                xytext=(15, 80), 
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.title('Common Time Complexities and Data Structure Operations', fontsize=16)
    plt.xlabel('Input Size (n)', fontsize=14)
    plt.ylabel('Operations', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('big_o_data_structures.png', dpi=300, bbox_inches='tight')
    logger.info("Big O comparison chart saved as 'big_o_data_structures.png'")

# -------------------- Example Use Cases --------------------

def demonstrate_finance_app() -> None:
    """
    Demonstrates a finance app scenario (Exercise 2.1).
    Many inserts (daily expenses) and few reads (monthly summary).
    """
    logger.info("\nDemonstration: Finance App (Exercise 2.1)")
    logger.info("-" * 60)
    
    # Simulate a month of expenses
    days = 30
    transactions_per_day = 5
    
    # Using a linked list
    start_time = time.time()
    finance_ll = LinkedList()
    
    # Insert daily expenses
    for day in range(1, days + 1):
        for _ in range(transactions_per_day):
            amount = round(random.uniform(1, 100), 2)
            finance_ll.append((day, amount))
    
    # Calculate monthly summary (read operation)
    total = 0
    current = finance_ll.head
    while current:
        total += current.data[1]
        current = current.next
    
    ll_time = time.time() - start_time
    
    # Using an array
    start_time = time.time()
    finance_array = DynamicArray()
    
    # Insert daily expenses
    for day in range(1, days + 1):
        for _ in range(transactions_per_day):
            amount = round(random.uniform(1, 100), 2)
            finance_array.append((day, amount))
    
    # Calculate monthly summary (read operation)
    total = 0
    for i in range(len(finance_array)):
        total += finance_array.get(i)[1]
    
    array_time = time.time() - start_time
    
    logger.info(f"Performance for {days} days with {transactions_per_day} transactions per day:")
    logger.info(f"Linked List time: {ll_time:.6f} seconds")
    logger.info(f"Dynamic Array time: {array_time:.6f} seconds")
    logger.info(f"For this use case, {'Linked List' if ll_time < array_time else 'Dynamic Array'} is faster.")
    logger.info(f"The answer to Exercise 2.1 is: {'Linked List' if ll_time < array_time else 'Dynamic Array'}")
    logger.info("Note: This may vary based on implementation details and system performance.")

def demonstrate_restaurant_queue() -> None:
    """
    Demonstrates a restaurant order queue scenario (Exercise 2.2).
    Orders are added to the back and processed from the front (FIFO).
    """
    logger.info("\nDemonstration: Restaurant Order Queue (Exercise 2.2)")
    logger.info("-" * 60)
    
    # Simulate a busy restaurant day
    num_orders = 100
    
    # Using a linked list
    start_time = time.time()
    order_queue_ll = LinkedList()
    
    # Add orders to the back
    for i in range(num_orders):
        order_queue_ll.append(f"Order #{i+1}")
    
    # Process orders from the front
    while order_queue_ll.head:
        order = order_queue_ll.head.data
        order_queue_ll.delete_at(0)  # Remove from front
    
    ll_time = time.time() - start_time
    
    # Using an array
    start_time = time.time()
    order_queue_array = DynamicArray()
    
    # Add orders to the back
    for i in range(num_orders):
        order_queue_array.append(f"Order #{i+1}")
    
    # Process orders from the front
    while len(order_queue_array) > 0:
        order = order_queue_array.get(0)
        order_queue_array.delete_at(0)  # Remove from front
    
    array_time = time.time() - start_time
    
    logger.info(f"Performance for processing {num_orders} orders:")
    logger.info(f"Linked List time: {ll_time:.6f} seconds")
    logger.info(f"Dynamic Array time: {array_time:.6f} seconds")
    logger.info(f"For this use case, {'Linked List' if ll_time < array_time else 'Dynamic Array'} is faster.")
    logger.info(f"The answer to Exercise 2.2 is: {'Linked List' if ll_time < array_time else 'Dynamic Array'}")
    logger.info("Note: Linked lists are typically better for queue operations because removing from the front")
    logger.info("of an array requires shifting all remaining elements, which is O(n).")

def demonstrate_facebook_login() -> None:
    """
    Demonstrates the Facebook login scenario (Exercise 2.3).
    Frequent searches using binary search.
    """
    logger.info("\nDemonstration: Facebook Login (Exercise 2.3)")
    logger.info("-" * 60)
    
    # Note: Binary search requires random access, which linked lists don't provide efficiently
    
    # Generate random usernames
    num_users = 10000
    usernames = [f"user_{i}" for i in range(num_users)]
    usernames.sort()  # Binary search requires sorted data
    
    # Using an array with binary search
    def binary_search(arr, target):
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    # Using a linked list with linear search
    def linear_search(ll, target):
        current = ll.head
        index = 0
        while current:
            if current.data == target:
                return index
            current = current.next
            index += 1
        return -1
    
    # Create data structures
    user_array = DynamicArray()
    for username in usernames:
        user_array.append(username)
    
    user_ll = LinkedList()
    for username in usernames:
        user_ll.append(username)
    
    # Test searches
    num_searches = 1000
    search_targets = [usernames[random.randint(0, num_users - 1)] for _ in range(num_searches)]
    
    # Array with binary search
    start_time = time.time()
    for target in search_targets:
        binary_search(usernames, target)  # Using the Python list for simplicity
    array_time = time.time() - start_time
    
    # Linked list with linear search
    start_time = time.time()
    for target in search_targets:
        linear_search(user_ll, target)
    ll_time = time.time() - start_time
    
    logger.info(f"Performance for {num_searches} login searches with {num_users} users:")
    logger.info(f"Array with binary search: {array_time:.6f} seconds")
    logger.info(f"Linked list with linear search: {ll_time:.6f} seconds")
    logger.info(f"Binary search is approximately {ll_time/array_time:.2f}x faster")
    logger.info(f"The answer to Exercise 2.3 is: Array (for binary search capabilities)")
    
    # Discuss Exercise 2.4 - downsides of arrays for inserts with binary search
    logger.info("\nDiscussion for Exercise 2.4 - Downsides of arrays for inserts with binary search:")
    logger.info("1. When inserting into a sorted array, you need to shift elements to maintain order.")
    logger.info("2. This shifting operation is O(n) in the worst case (inserting at the beginning).")
    logger.info("3. If the array needs to resize, this is an additional O(n) operation.")
    logger.info("4. Binary search benefits are offset by expensive insertion costs if frequent inserts occur.")

def demonstrate_hybrid_structure() -> None:
    """
    Demonstrates the hybrid data structure (Exercise 2.5).
    An array of linked lists.
    """
    logger.info("\nDemonstration: Hybrid Data Structure - Array of Linked Lists (Exercise 2.5)")
    logger.info("-" * 60)
    
    class ArrayOfLinkedLists:
        def __init__(self, num_buckets=26):
            self.buckets = [LinkedList() for _ in range(num_buckets)]
        
        def insert(self, username):
            """Insert a username into the appropriate bucket."""
            if not username:
                return
            
            # Get first character and convert to index (a=0, b=1, etc.)
            first_char = username[0].lower()
            index = ord(first_char) - ord('a')
            
            # Handle non-alphabetic characters
            if index < 0 or index >= len(self.buckets):
                index = len(self.buckets) - 1  # Use last bucket for special chars
            
            self.buckets[index].append(username)
        
        def search(self, username):
            """Search for a username."""
            if not username:
                return False
            
            # Get first character and convert to index
            first_char = username[0].lower()
            index = ord(first_char) - ord('a')
            
            # Handle non-alphabetic characters
            if index < 0 or index >= len(self.buckets):
                index = len(self.buckets) - 1
            
            # Search in the specific bucket
            return self.buckets[index].search(username) != -1
    
    # Generate random usernames
    num_users = 10000
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    usernames = []
    
    for _ in range(num_users):
        first_char = random.choice(alphabet)
        username = f"{first_char}{random.randint(1000, 9999)}"
        usernames.append(username)
    
    # Create data structures for comparison
    sorted_array = sorted(usernames)  # For binary search
    linked_list = LinkedList()
    hybrid = ArrayOfLinkedLists()
    
    for username in usernames:
        linked_list.append(username)
        hybrid.insert(username)
    
    # Test searches
    num_searches = 1000
    search_targets = [usernames[random.randint(0, num_users - 1)] for _ in range(num_searches)]
    
    # Binary search in sorted array
    start_time = time.time()
    for target in search_targets:
        # Using Python's built-in binary search for simplicity
        index = bisect_left(sorted_array, target)
        found = index < len(sorted_array) and sorted_array[index] == target
    array_time = time.time() - start_time
    
    # Linear search in linked list
    start_time = time.time()
    for target in search_targets:
        linked_list.search(target)
    ll_time = time.time() - start_time
    
    # Hybrid structure search
    start_time = time.time()
    for target in search_targets:
        hybrid.search(target)
    hybrid_time = time.time() - start_time
    
    logger.info(f"Performance for {num_searches} searches with {num_users} users:")
    logger.info(f"Sorted array with binary search: {array_time:.6f} seconds")
    logger.info(f"Linked list with linear search: {ll_time:.6f} seconds")
    logger.info(f"Hybrid structure (array of linked lists): {hybrid_time:.6f} seconds")
    
    # Test insertions
    num_inserts = 1000
    new_usernames = []
    
    for _ in range(num_inserts):
        first_char = random.choice(alphabet)
        username = f"{first_char}{random.randint(10000, 99999)}"
        new_usernames.append(username)
    
    # Insert into sorted array (need to maintain sorting)
    start_time = time.time()
    for username in new_usernames:
        # Find insertion point
        index = bisect_left(sorted_array, username)
        sorted_array.insert(index, username)
    array_insert_time = time.time() - start_time
    
    # Insert into linked list
    start_time = time.time()
    for username in new_usernames:
        linked_list.append(username)  # Just append for simplicity
    ll_insert_time = time.time() - start_time
    
    # Insert into hybrid structure
    start_time = time.time()
    for username in new_usernames:
        hybrid.insert(username)
    hybrid_insert_time = time.time() - start_time
    
    logger.info(f"\nPerformance for {num_inserts} insertions:")
    logger.info(f"Sorted array: {array_insert_time:.6f} seconds")
    logger.info(f"Linked list: {ll_insert_time:.6f} seconds")
    logger.info(f"Hybrid structure: {hybrid_insert_time:.6f} seconds")
    
    logger.info("\nAnalysis of the hybrid data structure (Exercise 2.5):")
    logger.info("1. Search performance:")
    if hybrid_time < array_time and hybrid_time < ll_time:
        logger.info("   - Hybrid structure is faster than both array and linked list for searching")
    elif hybrid_time < ll_time:
        logger.info("   - Hybrid structure is faster than linked list but slower than array for searching")
    else:
        logger.info("   - Hybrid structure is slower than array for searching")
    
    logger.info("2. Insertion performance:")
    if hybrid_insert_time < array_insert_time and hybrid_insert_time < ll_insert_time:
        logger.info("   - Hybrid structure is faster than both array and linked list for insertion")
    elif hybrid_insert_time < array_insert_time:
        logger.info("   - Hybrid structure is faster than array but slower than linked list for insertion")
    else:
        logger.info("   - Hybrid structure is slower than linked list for insertion")
    
    logger.info("\nConclusion for Exercise 2.5:")
    logger.info("The hybrid data structure provides a compromise between arrays and linked lists:")
    logger.info("- Faster than linear search in a linked list as you only search within a specific bucket")
    logger.info("- Faster than binary search in an array when the array is very large")
    logger.info("- Faster insertions than a sorted array since no elements need to be shifted")
    logger.info("- More memory efficient than a hash table for this specific use case")
    logger.info("This is why such hybrid structures are often used in real-world systems like databases")

# -------------------- Main Function --------------------

def main():
    """Main function to run the arrays vs linked lists demonstrations."""
    logger.info("=" * 80)
    logger.info("Arrays vs Linked Lists - Chapter 2 Demonstrations")
    logger.info("=" * 80)
    
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    try:
        # Run performance comparison
        sizes = [100, 1000, 10000]
        operations = ["append", "prepend", "insert_middle", "delete_middle", "get_random", "search"]
        
        logger.info("\nRunning performance comparison between arrays and linked lists...")
        results = compare_operations(sizes, operations)
        
        logger.info("\nGenerating performance visualizations...")
        plot_comparison(sizes, results)
        
        # Demonstrate example use cases for the exercises
        demonstrate_finance_app()  # Exercise 2.1
        demonstrate_restaurant_queue()  # Exercise 2.2
        demonstrate_facebook_login()  # Exercises 2.3 and 2.4
        
        # For Exercise 2.5, we need the bisect module
        from bisect import bisect_left
        demonstrate_hybrid_structure()  # Exercise 2.5
        
        logger.info("\nAll demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()