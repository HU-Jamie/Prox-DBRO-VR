import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Solution:
    def floyd_warshall(self, num_nodes, edges):
        dist_matrix = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist_matrix, 0)
        for u, v in edges:
            dist_matrix[u, v] = 1  # Assume each edge has a weight of 1
            dist_matrix[v, u] = 1
        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if dist_matrix[i, j] > dist_matrix[i, k] + dist_matrix[k, j]:
                        dist_matrix[i, j] = dist_matrix[i, k] + dist_matrix[k, j]
        return dist_matrix

    def calculate_d_mpl(self, num_nodes, edges):
        dist_matrix = self.floyd_warshall(num_nodes, edges)
        diameter = np.max(dist_matrix[np.isfinite(dist_matrix)])
        mean_path_length = np.sum(dist_matrix[np.isfinite(dist_matrix)]) / (num_nodes * (num_nodes - 1))
        return diameter, mean_path_length

    def generate_graphs(self, num_nodes, k):
        # Initialize edges in a circular fashion
        edges_set = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
        # Track the degree of each node
        degree = [2] * num_nodes  # Each node starts with two connections

        # Add extra edges to achieve targeted degree
        for node in range(num_nodes):
            attempts = 0  # Limit attempts to prevent infinite loops
            while degree[node] < k:
                # List potential nodes to connect with
                potential_nodes = [n for n in range(num_nodes) if n != node and degree[n] < k]
                if not potential_nodes:
                    print(f"Failed to find a valid node to connect for node {node}, degree: {degree}")
                    break  # If no potential node left, break the loop

                # Randomly select a node to connect
                np.random.shuffle(potential_nodes)
                for neighbor in potential_nodes:
                    if (node, neighbor) not in edges_set and (neighbor, node) not in edges_set:
                        edges_set.append((node, neighbor))
                        degree[node] += 1
                        degree[neighbor] += 1
                        break
                attempts += 1
                if attempts > num_nodes:  # Avoid infinite attempts
                    print(f"Exceeded maximum attempts for node {node}.")
                    break

        # Check if degree requirements are met
        if any(d != k for d in degree):
            print(f"Invalid graph with degrees: {degree}")
            return None, None

        # Create adjacency matrix
        Adj = np.zeros((num_nodes, num_nodes), dtype=int)
        for u, v in edges_set:
            Adj[u, v] = 1
            Adj[v, u] = 1

        return edges_set, Adj

    def test_connectivity(self, num_nodes, edges):
        adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
        for u, v in edges:
            adjacency_matrix[u, v] = 1
            adjacency_matrix[v, u] = 1
        visited = np.zeros(num_nodes, dtype=bool)
        stack = [0]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                for neighbor, is_connected in enumerate(adjacency_matrix[node]):
                    if is_connected and not visited[neighbor]:
                        stack.append(neighbor)
        return np.all(visited)

    def search_minimal_diameter_mpl_graph(self, num_nodes, k, iterations=1000):
        best_diameter = np.inf
        best_mpl = np.inf
        best_edges = None
        best_adj = None  # Store the best adjacency matrix found

        for iter in range(iterations):
            edges_set, Adj = self.generate_graphs(num_nodes, k)
            if edges_set is None or Adj is None:
                continue  # Skip invalid graphs

            if self.test_connectivity(num_nodes, edges_set):
                diameter, mean_path_length = self.calculate_d_mpl(num_nodes, edges_set)
                if (diameter < best_diameter) or (diameter == best_diameter and mean_path_length < best_mpl):
                    best_diameter = diameter
                    best_mpl = mean_path_length
                    best_edges = edges_set
                    best_adj = Adj

            if iter % (iterations // 10) == 0:
                print(f"The current iteration is {iter}th. Best diameter so far: {best_diameter}, Best MPL: {best_mpl}")

        return best_edges, best_diameter, best_mpl, best_adj


if __name__ == "__main__":
    solution = Solution()
    num_nodes = 16
    k = 3
    best_edges_set, best_diameter, best_mpl, Adj = solution.search_minimal_diameter_mpl_graph(num_nodes, k, iterations=100)

    if best_edges_set is not None:
        print(f"Best diameter: {best_diameter}")
        print(f"Best mean path length: {best_mpl}")
        print(f"Edges of the optimal graph: {best_edges_set}")


        # Visualize the desired graph
        G = nx.from_numpy_array(Adj)
        plt.figure(figsize=(6, 6))
        # Custom labels for nodes, starting from 1
        labels = {i: i + 1 for i in range(num_nodes)}
        nx.draw_circular(G, labels=labels, with_labels=True, font_size=8.5, font_color='w', node_size=120, node_color='#FFA326', width=1, edge_color='#0073BD')
        plt.show()
    else:
        print("Failed to find a valid graph.")