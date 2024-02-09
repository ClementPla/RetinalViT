#include <torch/extension.h>
#include <iostream>
#include <vector>


bool notInSet(const std::set<int>& s, int n){
    return s.find(n) == s.end();
}

at::Tensor clusterIndex(
    torch::Tensor edge_index
){
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);
    int current_cluster = 0;
    auto cluster_index = torch::zeros({num_nodes}, torch::kInt64);
    int last_node = -1;
    std::set<int> nodes_in_clusters;
    for (int i = 0; i < num_edges; i++){
        int nodeA = edge_index.index({0, i}).item<int>();
        int nodeB = edge_index.index({1, i}).item<int>();        
        if (notInSet(nodes_in_clusters, nodeA)) {
            nodes_in_clusters.insert(nodeA);
            cluster_index.index_put_({nodeA}, current_cluster);
            // We are on a new node, thus a new cluster
            current_cluster++;

            // Keep track on the current node/cluster we are working on for now. 
            last_node = nodeA;
        }
        else if (last_node != nodeA){
          // We are still on the same node, thus the same cluster
          continue;
        }
        if (notInSet(nodes_in_clusters, nodeB)) {
            // This node is not in any cluster, thus we add it to its parent cluster
            nodes_in_clusters.insert(nodeB);
            int nodeA_cluster = cluster_index.index({nodeA}).item<int>();
            cluster_index.index_put_({nodeB}, nodeA_cluster);
        }
    }
    return cluster_index;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cluster_index", &clusterIndex, "Cluster edges into clusters based on connected components.");
}
