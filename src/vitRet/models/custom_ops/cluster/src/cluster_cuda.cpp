#include <ATen/ATen.h>
#include <cuda_runtime.h> // Include CUDA runtime header to use CUDA features
#include <torch/extension.h>


at::Tensor clusterIndexCuda(torch::Tensor edge_index);

at::Tensor clusterIndexCpu(torch::Tensor edge_index){
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);

    auto clusters = at::full({num_nodes}, -1, edge_index.options());
    auto is_leaf = at::zeros({num_nodes}, edge_index.options().dtype(torch::kBool));
    
    auto clusters_data = clusters.data_ptr<int64_t>();
    auto is_leaf_data = is_leaf.data_ptr<bool>();
    int n_clusters = -1;

    for(int i=0; i<num_edges; i++){
        int64_t nodeA = edge_index[0][i].item<int64_t>();
        int64_t nodeB = edge_index[1][i].item<int64_t>();

        if(clusters_data[nodeA] == -1 && clusters_data[nodeB] == -1){
            is_leaf_data[nodeA] = true;
            n_clusters++;
        }
        if (is_leaf_data[nodeA] && (!is_leaf_data[nodeB] || nodeA == nodeB) && clusters_data[nodeB] == -1){
            clusters_data[nodeA] = n_clusters;
            clusters_data[nodeB] = n_clusters;
        }
    }
    return clusters;

}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor clusterIndex(torch::Tensor edge_index){
    if(edge_index.device().is_cuda()){
        CHECK_CUDA(edge_index);
        return clusterIndexCuda(edge_index);
    }
    else
        return clusterIndexCpu(edge_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_index", &clusterIndex, "Cluster edges into clusters based on connected components.");
}