from fast_slic.avx2 import SlicAvx2 as Slic
from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.datamodules import CLASSIF_PATHS, DataHookPosition
from fundus_data_toolkit.datamodules.classification import (
    AptosDataModule,
    DDRDataModule,
    EyePACSDataModule,
    IDRiDDataModule,
)
from fundus_data_toolkit.datamodules.utils import merge_existing_datamodules
from nntools.dataset import nntools_wrapper


def verify_data(data):
    segments = data["segments"]
    assert segments.max() > 0, "No segments found"


def get_datamodules(dataset_paths: dict, dataset_module_args: dict, superpixels_args: dict):
    ds = []

    n_segments = superpixels_args.get("n_segments", 2048)
    min_size = superpixels_args.get("min_size", 0)
    compactness = superpixels_args.get("compactness", 10)
    filter_roi = superpixels_args.get("roi_filter", True)
    convert_to_lab = superpixels_args.get("convert_to_lab", True)
    manhattan_spatial_dist = superpixels_args.get("manhattan_spatial_dist", True)
    num_threads = superpixels_args.get("num_threads", 1)

    print(f"Superpixels: {n_segments} segments, compactness: {compactness}, min_size: {min_size}")
    da_type = dataset_module_args.pop("da_type", DAType.DEFAULT)
    da_type = DAType(da_type)

    dataset_module_args["callbacks"] = [verify_data]

    cache_dir = f"supepixels_{n_segments}_{compactness}_filter_{filter_roi}"

    for k, v in dataset_paths.items():
        match k.upper():
            case "DDR":
                ds.append(
                    DDRDataModule(
                        data_dir=v, cache_dir=cache_dir, data_augmentation_type=da_type, **dataset_module_args
                    )
                )
            case "EYEPACS":
                ds.append(
                    EyePACSDataModule(
                        data_dir=v, cache_dir=cache_dir, data_augmentation_type=da_type, **dataset_module_args
                    )
                )
            case "IDRID":
                ds.append(
                    IDRiDDataModule(
                        data_dir=v, cache_dir=cache_dir, data_augmentation_type=da_type, **dataset_module_args
                    )
                )
            case "APTOS":
                ds.append(
                    AptosDataModule(
                        data_dir=v, cache_dir=cache_dir, data_augmentation_type=da_type, **dataset_module_args
                    )
                )
    datamodule = merge_existing_datamodules(ds)

    slic = Slic(
        num_components=n_segments,
        convert_to_lab=manhattan_spatial_dist,
        min_size_factor=min_size,
        manhattan_spatial_dist=convert_to_lab,
        compactness=compactness,
        num_threads=num_threads,
    )

    @nntools_wrapper
    def get_superpixels(image, roi=None):
        output = {"image": image}
        # image = cv2.bilateralFilter(image, 9, 75, 75)

        # segments = felzenszwalb(image, scale=10, sigma=0.5, min_size=15)

        # gradient = sobel(rgb2gray(image))
        # segments = watershed(gradient, markers=4096, compactness=1e-7)
        # segments = quickshift(image, kernel_size=3, max_dist=6, ratio=0.5)
        segments = slic.iterate(image, 50)
        if filter_roi:
            segments = roi.astype(segments.dtype) * segments
            output["mask"] = roi
        output["segments"] = segments
        return output

    datamodule.set_data_pipeline_hook(get_superpixels, position=DataHookPosition.POST_RESIZE_PRE_CACHE)
    datamodule.setup_all()
    datamodule.add_target({"segments": "mask"})
    return datamodule


def get_dm_from_config(config: dict):
    return get_datamodules(config["data"]["data_path"], config["data"]["dataset"], config["data"]["superpixels"])


def get_test_sample(batch_size=8, img_size=(1024, 1024), shuffle=False, dataloader_idx=0, **kwargs):
    paths = {
        "IDRID": CLASSIF_PATHS.IDRID,
        "APTOS": CLASSIF_PATHS.APTOS,
        "EYEPACS": CLASSIF_PATHS.EYEPACS,
        "DDR": CLASSIF_PATHS.DDR,
        "APTOS": CLASSIF_PATHS.APTOS,
    }
    dataset_args = {
        "batch_size": batch_size,
        "da_type": "heavy",
        "use_cache": False,
        "img_size": img_size,
        "num_workers": 0,
    }

    superpixels_args = {
        "n_segments": 4096,
        "min_size": 0,
        "compactness": 10,
        "roi_filter": True,
        "convert_to_lab": True,
        "manhattan_spatial_dist": True,
    }
    superpixels_args.update(kwargs)

    datamodule = get_datamodules(paths, dataset_args, superpixels_args)
    dataloader = datamodule.test_dataloader(shuffle=shuffle)[dataloader_idx]
    return next(iter(dataloader))
