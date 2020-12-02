import networkx as nx
import community as comm


def louvain(graph, partition=None, modularity = -.5):
    partition = comm.best_partition(graph, partition)
    new_modularity = comm.modularity(partition, graph)
    while new_modularity > modularity:
        modularity, new_modularity = new_modularity, -.5
        partition = comm.best_partition(graph, partition)
        new_modularity = community_louvain.modularity(partition, graph)
    return partition, comm