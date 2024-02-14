#include <torch/extension.h>
#include <iostream>
#include <vector>


bool notInSet(const std::unordered_set<int>& s, int n){
    return s.find(n) == s.end();
}

at::Tensor clusterIndex(
    torch::Tensor edge_index)
{
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);
    int current_cluster = 0;
    auto cluster_index = edge_index.new_zeros({num_nodes});
    int last_node = -1;
    std::unordered_set<int> nodes_in_clusters;
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

at::Tensor fasterClusterIndex(
    torch::Tensor edge_index)
{
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);
    int current_cluster = 1;
    auto cluster_index = edge_index.new_zeros({num_nodes});
    auto last_node = edge_index.new_zeros({1});

    for (int i = 0; i < num_edges; i++){
        auto nodeA = edge_index[0][i];
        auto nodeB = edge_index[1][i];
        
        if ((cluster_index[nodeA] == 0).item<bool>()){
            cluster_index.index_put_({nodeA}, current_cluster);
            // We are on a new node, thus a new cluster
            current_cluster++;
            // Keep track on the current node/cluster wefasterC are working on for now. 
            last_node = nodeA;
        }
        else if ((last_node !=  nodeA).item<bool>()){
          // We changed cluster
          continue;
        }
        if ((cluster_index[nodeB] ==  0).item<bool>()) {
            // This node is not in any cluster, thus we add it to its parent cluster
            int nodeA_cluster = cluster_index.index({nodeA}).item<int>();
            cluster_index.index_put_({nodeB}, nodeA_cluster);
        }
    }
    return cluster_index;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cluster_index", &fasterClusterIndex, "Cluster edges into clusters based on connected components.");
}


