import cv2
from nntools.dataset import ClassificationDataset, Composition, ImageDataset, nntools_wrapper
from python_color_transfer.color_transfer import ColorTransfer

from vitRet.data.fundus import FundusDataModule


class DeepDRIDDataModule(FundusDataModule):
    def __init__(
        self,
        data_dir,
        img_size=(512, 512),
        valid_size=0.1,
        batch_size=64,
        num_workers=32,
        use_cache=False,
        use_superpixels=True,
        superpixels_scales=4,
        superpixels_max_nb=2048,
        superpixels_min_nb=32,
        superpixels_filter_black=True,
        superpixels_num_threads=1,
        superpixels_compactness=10,
        superpixels_min_size_factor=0,
        ref_image='/home/clement/Documents/data/IDRID/B. Disease Grading/1. Original Images/a. Training Set/IDRiD_072.jpg'
    ):
        super().__init__(
            data_dir=data_dir,
            img_size=img_size,
            valid_size=valid_size,
            batch_size=batch_size,
            num_workers=num_workers,
            use_cache=use_cache,
            use_superpixels=use_superpixels,
            superpixels_scales=superpixels_scales,
            superpixels_max_nb=superpixels_max_nb,
            superpixels_min_nb=superpixels_min_nb,
            superpixels_filter_black=superpixels_filter_black,
            superpixels_num_threads=superpixels_num_threads,
            superpixels_compactness=superpixels_compactness,
            superpixels_min_size_factor=superpixels_min_size_factor,
        )
        self.ref_img = cv2.imread(ref_image)[:,:,::-1]
        self.colorTransfer = ColorTransfer()
    
    def setup(self, stage: str) -> None:
        if stage == 'fit':
            root_dir = self.root_img + 'ultra-widefield-training/'
            d = ClassificationDataset(root_dir+'Images/', shape=self.img_size,
                          label_filepath=root_dir+'ultra-widefield-training.csv',
                          gt_column='DR_level', file_column='image_id'
                          )
    
            d.composer = Composition()
            d.composer.add(self.color_cvt(), *self.img_size_ops(), *self.normalize_and_cast_op())
            d.remap('DR_level', 'label')
            self.train = d


        if stage in ['test', 'validate']:

            root_dir = self.root_img + 'ultra-widefield-validation/'
            d = ClassificationDataset(root_dir+'Images/', shape=self.img_size,
                          label_filepath=root_dir+'ultra-widefield-validation.csv',
                          gt_column='DR_level', file_column='image_id'
                          )
            d.remap('DR_level', 'label')
            d.composer = Composition()
            d.composer.add(self.color_cvt(), *self.img_size_ops(), *self.normalize_and_cast_op())
            self.test = self.val = d

    def color_cvt(self):

        @nntools_wrapper
        def colorTransfer(image):
            image = self.colorTransfer.lab_transfer(image, self.ref_img)
            return {'image':image}
        return colorTransfer