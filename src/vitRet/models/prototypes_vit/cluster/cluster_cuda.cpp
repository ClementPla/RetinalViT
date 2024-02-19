#include <ATen/ATen.h>
#include <cuda_runtime.h> // Include CUDA runtime header to use CUDA features
#include <torch/extension.h>


at::Tensor clusterIndexCuda(torch::Tensor edge_index);

at::Tensor clusterIndex(torch::Tensor edge_index){
      return clusterIndexCuda(edge_index);
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cluster_index", &clusterIndex, "Cluster edges into clusters based on connected components.");
}