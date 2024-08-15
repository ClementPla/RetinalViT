#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>


__global__ void clusterAssignmentKernel(
    const int64_t* edge_index, int num_edges, int64_t* clusters_data, int num_nodes) {
    int edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= num_edges) return;

    int64_t nodeA = edge_index[edge];
    int64_t nodeB = edge_index[edge + num_edges]; // Considering edge_index as flattened
    atomicMin((unsigned long long*)&clusters_data[nodeA], nodeA);
    atomicMin((unsigned long long*)&clusters_data[nodeB], nodeA);
    atomicMin((unsigned long long*)&clusters_data[nodeA], nodeB);
    atomicMin((unsigned long long*)&clusters_data[nodeB], nodeB);

}

__global__ void consolidationKernel( 
    const int64_t* edge_index, int num_edges, int64_t* clusters_data, int num_nodes) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= num_nodes) return;

    for (int i = 0; i < num_edges; ++i) {
        int64_t nodeA = edge_index[i];
        int64_t nodeB = edge_index[i + num_edges];

        if (nodeA == node || nodeB == node) {
            int64_t clusterA = clusters_data[nodeA];
            int64_t clusterB = clusters_data[nodeB];
            int64_t minCluster = min(clusterA, clusterB);
            atomicMin((unsigned long long*)&clusters_data[nodeA], minCluster);
            atomicMin((unsigned long long*)&clusters_data[nodeB], minCluster);
        }
    }
}

at::Tensor clusterIndexCuda(torch::Tensor edge_index) {
    auto device = edge_index.device();
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);

    auto clusters = at::full({num_nodes}, num_nodes, edge_index.options());    
    auto clusters_data = clusters.data_ptr<int64_t>();

    // Assuming edge_index is a [2, num_edges] tensor, we need to flatten it for GPU processing
    auto edge_index_contig = edge_index.contiguous();
    auto edge_index_data = edge_index_contig.data_ptr<int64_t>();

    // Kernel invocation
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_edges + threadsPerBlock - 1) / threadsPerBlock;
    clusterAssignmentKernel<<<blocksPerGrid, threadsPerBlock>>>(
        edge_index_data, num_edges, clusters_data, num_nodes);

    cudaDeviceSynchronize();


    return clusters;
}