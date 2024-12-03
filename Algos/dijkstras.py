from collections import defaultdict
import heapq as heap

def dijkstras(graph, start_vertex):
    #visited set of nodes
    visited = set()
    
    #parent nodes
    parent_map = {}
    
    #list as a priority queue
    pq = []
    
    #dict to hold a group vertex and its distance as a key
    node_distances = defaultdict(lambda:float('inf'))
    node_distances[start_vertex] = 0
    
    heap.heappush(pq, (0,start_vertex))
    
    while pq:
        _, node = heap.heappop(pq)
        visited.add(node)
        
        for adj_node, weight in graph[node]:
            
            if adj_node in visited:
                continue
            
            new_distance = node_distances[node] + weight
            if node_distances[adj_node] > new_distance:
                parent_map[adj_node] = node
                node_distances[adj_node] = new_distance
                
                heap.heappush(pq, (new_distance, adj_node))
                
    return parent_map, node_distances

def main():
    graph = defaultdict()
    graph[0] = [(1, 11), (2, 5)]
    graph[1] = [(3, 2)]
    graph[2] = [(1, 4), (3, 6)]
    graph[3] = []
    
    parentmap, nodeDistance = dijkstras(graph, 0)
    
    print(parentmap)
    print(nodeDistance)

if __name__ == "__main__":
    main()