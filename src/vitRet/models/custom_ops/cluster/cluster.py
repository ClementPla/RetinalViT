import warnings

import cudf
import cugraph
import torch
from torch_cluster import graclus_cluster
from torch_geometric.utils import dense_to_sparse
from vitRet.models.superpixels.graph import normalize_adjacency
from vitRet.models.superpixels.indexing import consecutive_reindex_batch_of_integers_tensor

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_community_cluster(
    adj,
    mask_node=None,
    max_iter=100,
    resolution=50.0,
    theta=0.5,
    fill_diagonal=False,
    normalize_adj=False,
    algorithm="leiden",
    remap_cluster: bool = False,
    per_batch: bool = False,
):
    B = adj.shape[0]
    N = adj.shape[1]
    if fill_diagonal:
        adj = torch.diagonal_scatter(adj, adj.amax(2) + 1e-6, dim1=1, dim2=2)
    if normalize_adj:
        adj = normalize_adjacency(adj, normalize=True, binarize=False)
    if per_batch:
        clusters_tensor = get_clusters_from_adj(adj, algorithm, N, B, max_iter, resolution, theta, remap_cluster)
    else:
        c = []
        for a in adj:
            c.append(get_clusters_from_adj(a.unsqueeze(0), algorithm, N, 1, max_iter, resolution, theta, remap_cluster))
        clusters_tensor = torch.cat(c, dim=0)

    if mask_node is not None:
        clusters_tensor = clusters_tensor * mask_node
    clusters_tensor = consecutive_reindex_batch_of_integers_tensor(clusters_tensor)

    return clusters_tensor


def get_clusters_from_adj(adj, algorithm, N, B, max_iter, resolution, theta, remap_cluster: bool = False):
    edge_index, edge_attr = dense_to_sparse(adj)
    if algorithm in [
        "leiden",
        "louvain",
        "spectral_cluster",
        "spectral_modularity_cluster",
        "weakly_connected_components",
        "strongly_connected_components",
    ]:
        clusters_tensor = get_clusters_from_edges_cugraph(
            edge_index, edge_attr, N, B, max_iter, resolution, theta, algorithm=algorithm, remap_cluster=remap_cluster
        )
    elif algorithm in ["graclus"]:
        clusters_tensor = get_clusters_from_edges_others(
            edge_index, edge_attr, num_nodes=N, algorithm=algorithm, batch_size=B
        )
    else:
        raise ValueError(f"Unrecognized algorithm {algorithm}")

    return clusters_tensor


def get_clusters_from_edges_others(edge_index, edge_attr, num_nodes, batch_size, algorithm="graclus"):
    match algorithm:
        case "graclus":
            clusters = graclus_cluster(edge_index[0], edge_index[1], edge_attr)
            clusters = clusters.view(batch_size, -1)
            return clusters


def get_clusters_from_edges_cugraph(
    edge_index,
    edge_attr,
    num_nodes,
    batch_size,
    max_iter,
    resolution,
    theta,
    algorithm="leiden",
    remap_cluster: bool = False,
):
    edge_index = (edge_index).t().type(torch.int)

    dlpack_edges = torch.utils.dlpack.to_dlpack(edge_index)

    gdf_edges = cudf.from_dlpack(dlpack_edges)
    gdf_edges.columns = ["source", "destination"]
    dlpack_attributes = torch.utils.dlpack.to_dlpack(edge_attr.float())
    gdf_attributes = cudf.from_dlpack(dlpack_attributes)
    gdf = cudf.concat([gdf_edges, gdf_attributes], axis=1)
    gdf.columns = ["source", "destination", "weight"]
    G = cugraph.Graph(directed=False)
    G.from_cudf_edgelist(gdf, source="source", destination="destination", renumber=True, edge_attr="weight")

    N = num_nodes * batch_size
    match algorithm:
        case "leiden":
            partition, score = cugraph.leiden(
                G, resolution=resolution, max_iter=max_iter, theta=theta, random_state=torch.seed()
            )
            column = "partition"
        case "louvain":
            partition, score = cugraph.louvain(G, max_iter=max_iter, resolution=resolution)
            column = "partition"
        case "spectral_cluster":
            partition = cugraph.spectralBalancedCutClustering(
                G, num_clusters=int(0.5 * N), num_eigen_vects=min(0.5 * num_nodes, 15)
            )
            column = "cluster"
        case "spectral_modularity_cluster":
            partition = cugraph.spectralModularityMaximizationClustering(
                G, num_clusters=int(0.5 * N), num_eigen_vects=min(0.5 * num_nodes, 15)
            )
            column = "cluster"
        case "weakly_connected_components":
            partition = cugraph.weakly_connected_components(G)
            column = "labels"
        case "strongly_connected_components":
            partition = cugraph.strongly_connected_components(G)
            column = "labels"

    partition_sorted = partition.sort_values("vertex")
    if remap_cluster:
        unique_clusters = partition_sorted[column].unique()
        map = unique_clusters.to_dict()
        inv_map = {v: k for k, v in map.items()}
        partition_sorted[column] = partition_sorted[column].map(inv_map)
    clusters_series = partition_sorted[column]
    dlpack_clusters = clusters_series.to_dlpack()
    clusters_tensor = torch.utils.dlpack.from_dlpack(dlpack_clusters)
    return clusters_tensor.view(batch_size, -1)
