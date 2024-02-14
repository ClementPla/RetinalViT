import torch
from torch.utils.cpp_extension import load
from torch_geometric.utils import add_self_loops, dense_to_sparse, sort_edge_index, to_undirected

my_ext = load(name="cluster_cpp", sources=["src/vitRet/models/prototypes_vit/cluster/cluster_cpp.cpp"])
cluster_index = my_ext.cluster_index


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from vitRet.data.fundus import EyePACSDataModule
    from vitRet.models.prototypes_vit.utils import get_superpixels_adjacency, reindex_segments_from_batch
    data_dir = '/home/tmp/clpla/data/eyepacs/'
    dm = EyePACSDataModule(batch_size=2, data_dir=data_dir, num_workers=2, img_size=(1024, 1024),
                           superpixels_min_nb=32,
                           superpixels_scales=1)
    dm.setup('test')

    for batch in dm.test_dataloader():
        segments = batch['segments'].cuda()
        print(segments.max())
        adj = get_superpixels_adjacency(segments)
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(segments[1, 0].cpu(), cmap='RdYlGn')
        adj[:, 0, :] = 0
        adj[:, :, 0] = 0
        adj[:,0,0] = 1
        edge_index, edge_attribute  = dense_to_sparse(adj)
        clustered_nodes = cluster_index(edge_index) - 1 
        segments, batch_index = reindex_segments_from_batch(segments)
        output = torch.gather(clustered_nodes, -1, segments.flatten()).view(segments.shape)
        print(output[0].min())
        axs[1].imshow(output[0, 0].cpu(), cmap='RdYlGn')
        axs[2].imshow(output[1, 0].cpu(), cmap='RdYlGn')

        plt.show()
        break