from torch.utils.cpp_extension import load

gaussian_kde_cpp = load(
    name="gaussian_kde_cpp",
    sources=[
        "src/vitRet/models/custom_ops/kde/src/gaussian_kde.cpp",
        "src/vitRet/models/custom_ops/kde/src/gaussian_kde_kernel.cu",
    ],
    verbose=False,
)


def gaussian_KDE(pos, sampling_points_pos, mask, bandwidth):
    result = gaussian_kde_cpp.gaussian_kde(pos, sampling_points_pos.contiguous(), mask, bandwidth)
    return result


if __name__ == "__main__":
    import einops
    import matplotlib.pyplot as plt
    import torch
    from torch_geometric.utils import to_dense_batch
    from vitRet.models.superpixels.indexing import (
        consecutive_reindex_batch_of_integers_tensor,
        reindex_all_segments_across_batch,
    )
    from vitRet.utils.plot import imshow

    datamodule = ...

    batch = datamodule.train_dataloader().__iter__().__next__()
    img = batch["image"].cuda()
    seg = batch["segments"].cuda()
    img = img.repeat(4, 1, 1, 1)
    seg = seg.repeat(4, 1, 1, 1)
    B, C, H, W = img.shape
    grid_x, grid_y = torch.meshgrid(torch.linspace(-1, 1, W), torch.linspace(-1, 1, H), indexing="xy")

    pos = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0)], dim=0).cuda()
    pos = pos.unsqueeze(0).expand(B, -1, -1, -1)
    pos = einops.rearrange(pos, "b c h w -> (b h w) c")

    beta = 128
    grid_x_reduced, grid_y_reduced = torch.meshgrid(
        torch.linspace(-1, 1, beta), torch.linspace(-1, 1, beta), indexing="xy"
    )
    sampling_points_pos = torch.cat([grid_x_reduced.unsqueeze(0), grid_y_reduced.unsqueeze(0)], dim=0).cuda()
    sampling_points_pos = einops.rearrange(sampling_points_pos, "c h w -> (h w) c")
    seg = consecutive_reindex_batch_of_integers_tensor(seg)
    seg, batch_index = reindex_all_segments_across_batch(seg, with_padding=False)
    _, counts = torch.unique(seg, return_counts=True)
    max_node = torch.sort(counts, descending=True).values[B]
    stacked_segment = einops.rearrange(seg.squeeze(1), "b h w -> (b h w)")
    arg_sort = torch.argsort(stacked_segment)
    stacked_segment = stacked_segment[arg_sort]
    pos = pos[arg_sort]
    out_pos, mask = to_dense_batch(pos, stacked_segment, fill_value=0, max_num_nodes=max_node)
    N = out_pos.shape[0]
    print("Starting KDE")
    result = gaussian_kde_cpp.gaussian_kde(out_pos, sampling_points_pos.contiguous(), mask.bool(), 0.1)
    print("KDE done")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    n = 150

    print(result.amin(), result.amax())
    data = result[n].view(beta, beta)
    imshow(data, cmap="jet", ax=axs[0], normalize=False)
    alpha = 0.75
    imshow(
        ((seg[0] == n) * img[0]) * alpha + (1 - alpha) * img[0],
        normalize=True,
        ax=axs[1],
        segment=seg[0],
        border_alpha=0.25,
        border_width=4,
    )
    plt.show(block=True)
