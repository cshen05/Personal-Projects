from collections import deque

def bfs(graph, start):
    visited = set()  # To keep track of visited nodes
    queue = deque([start])  # Initialize a queue with the start node
    traversal_order = []  # To store the order of traversal

    while queue:
        node = queue.popleft()  # Dequeue a node
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            traversal_order.append(node)  # Add to traversal order

            # Add all unvisited neighbors to the queue
            for neighbor in graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
    
    return traversal_order

def dfs_recursive(graph, node, visited=None, traversal_order=None):
    if visited is None:
        visited = set()  # Initialize visited set
    if traversal_order is None:
        traversal_order = []  # Initialize traversal order

    visited.add(node)  # Mark the node as visited
    traversal_order.append(node)  # Add to traversal order

    # Recur for all unvisited neighbors
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited, traversal_order)
    
    return traversal_order

def dfs_iterative(graph, start):
    visited = set()  # To keep track of visited nodes
    stack = [start]  # Use a stack for DFS
    traversal_order = []  # To store the order of traversal

    while stack:
        node = stack.pop()  # Pop the last element
        if node not in visited:
            visited.add(node)  # Mark the node as visited
            traversal_order.append(node)  # Add to traversal order

            # Add all unvisited neighbors to the stack
            for neighbor in reversed(graph[node]):  # Reverse for correct order
                if neighbor not in visited:
                    stack.append(neighbor)
    
    return traversal_order


def main():
    # Example graph (adjacency list)
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'F'],
        'C': ['A', 'E', 'D'],
        'D': ['C', 'E'],
        'E': ['C', 'D', 'H'],
        'F': ['B', 'G', 'H'],
        'G': ['F'],
        'H': ['F', 'E'],
    }

    # BFS Traversal
    print("BFS Traversal:", bfs(graph, 'C'))

    # DFS Traversal (Recursive)
    print("DFS Traversal (Recursive):", dfs_recursive(graph, 'C'))

    # DFS Traversal (Iterative)
    print("DFS Traversal (Iterative):", dfs_iterative(graph, 'C'))


if __name__ == "__main__":
    main()