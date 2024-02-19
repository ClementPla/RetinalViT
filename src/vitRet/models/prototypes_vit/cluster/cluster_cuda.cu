#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>




__global__ void cluster_kernel(int64_t* __restrict__ edge_index_data,
                               int64_t* __restrict__ cluster_index_data,
                               int64_t* __restrict__ num_cluster,
                               int num_edges,
                               int num_nodes) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < num_edges) {
        int64_t nodeA = edge_index_data[index];
        int64_t nodeB = edge_index_data[index + num_edges];

        int64_t old = atomicCAS(cluster_index_data[nodeA], int64_t(0), num_cluster[0]+1);
        if (old == 0) {
            int64_t old_value = atomicAdd(num_cluster[0], int64_t(1));
        }
        atomicCAS(cluster_index_data[nodeB], int64_t(0), cluster_index_data[nodeA]);

    }
}

at::Tensor clusterIndexCuda(torch::Tensor edge_index) {
    int num_nodes = edge_index.max().item<int>() + 1;
    int num_edges = edge_index.size(1);
    auto num_cluster = at::zeros({1}, edge_index.options());
    auto cluster_index = at::zeros({num_nodes}, edge_index.options());
    int64_t* num_cluster_data = cluster_index.data_ptr<int64_t>();
    int64_t* cluster_index_data = cluster_index.data_ptr<int64_t>();

    edge_index = edge_index.to(at::kCUDA);
    int64_t* edge_index_data = edge_index.data_ptr<int64_t>();

    int threads_per_block = 256;
    int blocks = (num_edges + threads_per_block - 1) / threads_per_block;
    cluster_kernel<<<blocks, threads_per_block>>>(edge_index_data, cluster_index_data, num_cluster_data, num_edges, num_nodes);

    cudaDeviceSynchronize();

    return cluster_index;
}