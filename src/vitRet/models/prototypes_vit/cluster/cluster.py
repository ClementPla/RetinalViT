import cudf
import cugraph
import torch
from torch_geometric.utils import add_self_loops, dense_to_sparse

from vitRet.models.prototypes_vit.utils import consecutive_reindex_batch_of_integers_tensor, normalize_adjacency


def get_community_cluster(
    adj, mask_node=None, max_iter=100, resolution=50.0, theta=0.5, fill_diagonal=False, normalize_adj=False
):
    B = adj.shape[0]
    N = adj.shape[1]
    if fill_diagonal:
        adj = torch.diagonal_scatter(adj, adj.amax(2) + 1e-6, dim1=1, dim2=2)
    if normalize_adj:
        adj = normalize_adjacency(adj, normalize=True, binarize=False)

    edge_index, edge_attr = dense_to_sparse(adj)
    clusters_tensor = get_clusters_from_edges(edge_index, edge_attr, N, B, max_iter, resolution, theta)

    if mask_node is not None:
        clusters_tensor = clusters_tensor * mask_node
    clusters_tensor = consecutive_reindex_batch_of_integers_tensor(clusters_tensor)

    return clusters_tensor

@torch.no_grad()
def get_clusters_from_edges(edge_index, edge_attr, num_nodes, batch_size, max_iter, resolution, theta):
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=num_nodes)

    edge_index = (edge_index).t()

    dlpack_edges = torch.utils.dlpack.to_dlpack(edge_index)

    gdf_edges = cudf.from_dlpack(dlpack_edges)
    gdf_edges.columns = ["source", "destination"]
    dlpack_attributes = torch.utils.dlpack.to_dlpack(edge_attr.float())
    gdf_attributes = cudf.from_dlpack(dlpack_attributes)
    gdf = cudf.concat([gdf_edges, gdf_attributes], axis=1)
    gdf.columns = ["source", "destination", "weight"]
    G = cugraph.Graph()
    G.from_cudf_edgelist(gdf, source="source", destination="destination", edge_attr="weight")
    partition, score = cugraph.leiden(
        G,
        resolution=resolution,
        max_iter=max_iter, theta=theta, 
        random_state=torch.seed(), 
    )
    partition_sorted = partition.sort_values("vertex")
    # unique_clusters = partition_sorted['partition'].unique()
    # map = unique_clusters.to_dict()
    # inv_map = {v: k for k, v in map.items()}
    # partition_sorted['partition'] = partition_sorted['partition'].map(inv_map)
    clusters_series = partition_sorted["partition"]
    dlpack_clusters = clusters_series.to_dlpack()
    clusters_tensor = torch.utils.dlpack.from_dlpack(dlpack_clusters)
    return clusters_tensor.view(batch_size, -1)
