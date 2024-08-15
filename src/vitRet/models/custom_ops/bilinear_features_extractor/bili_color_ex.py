from torch.utils.cpp_extension import load

bili_extrac_cpp = load(
    name="bilinear_features_extract",
    sources=[
        "src/vitRet/models/custom_ops/bilinear_features_extractor/src/bilinear_features_extract.cpp",
        "src/vitRet/models/custom_ops/bilinear_features_extractor/src/bilinear_features_extract_cuda.cu",
    ],
    with_cuda=True,
    verbose=True,
    is_standalone=False,
    is_python_module=True,
)


def color_extractor(img, superpixels, bounding_boxes, beta=128, isolateSuperpixel=False):
    result = bili_extrac_cpp.bilinear_features_extract(img, superpixels, bounding_boxes, beta, isolateSuperpixel)
    return result
