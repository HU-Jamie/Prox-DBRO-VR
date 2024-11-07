import numpy as np

class Solution:
    def floyd_warshall(self, num_nodes, edges):
        # Initialize the distance matrix with infinity
        dist_matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist_matrix, 0)  # Distance from node to itself is 0
        # Add edges to the distance matrix
        for u, v in edges:
            dist_matrix[u - 1, v - 1] = 1  # Assume each edge has a weight of 1
            dist_matrix[v - 1, u - 1] = 1
        # Floyd-Warshall algorithm to find shortest paths among all pairs
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]

        return dist_matrix

    def calculate_d_mpl(self,Adj):
        num_nodes, edges = self.get_graph_info(Adj)
        # Compute all pairs shortest paths
        dist_matrix = self.floyd_warshall(num_nodes, edges)
        # Find the diameter (maximum shortest path)
        diameter = np.max(dist_matrix[np.isfinite(dist_matrix)])
        # Calculate the mean path length (average of all finite shortest paths)
        mean_path_length = np.sum(dist_matrix[np.isfinite(dist_matrix)]) / (num_nodes * (num_nodes - 1))
        return diameter, mean_path_length

    def get_graph_info(self, Adjacency_matrix):
        num_nodes = Adjacency_matrix.shape[0]
        edges = []
        # Traverse the upper triangle of the matrix to extract edges
        for i in range(num_nodes):
            Adjacency_matrix[i, i] = 0 # the distance of self-loop is zero
            for j in range(i + 1, num_nodes):
                if Adjacency_matrix[i, j] != 0:
                    # If there is an edge, add it to the list
                    edges.append((i, j))
        return num_nodes, edges

    def test_connectivity(self, adjacency_matrix):
        num_nodes = adjacency_matrix.shape[0]
        visited = np.zeros(num_nodes, dtype=bool)
        # Use a stack to implement DFS
        stack = [0]  # Start DFS from the first node (node 0)
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                # Check all neighbors of the current node
                for neighbor, is_connected in enumerate(adjacency_matrix[node]):
                    if is_connected and not visited[neighbor]:
                        stack.append(neighbor)
        # If all nodes are visited, the graph is connected
        return np.all(visited)


if __name__ == "__main__":
    solution = Solution()
    while True:
        try:
            Adj = eval(input("Please input the adjacency matrix of the graph (use numpy array format): "))
            if not solution.test_connectivity(Adj) or Adj.shape[0] != Adj.shape[1]:
                raise ValueError("Your input matrix indicates a disconnected graph or invalid adjacency matrix! Please provide another input:")
            diameter, mean_path_length = solution.calculate_d_mpl(Adj)
            print("The diameter of the graph is %d;" % (diameter), "\n" "The mean path length of the graph is %d." % (mean_path_length))
        except ValueError as e:
            print(f"Error: {e}. Please enter a valid adjacency matrix!")