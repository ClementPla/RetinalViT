from vitRet.data.segmentation import MaplesDR
from nntools.dataset.utils import class_weighting


maplesdr_datamodule = MaplesDR(data_dir='/home/tmp/clpla/data/Maples-DR/')
maplesdr_datamodule.setup('fit')
dataset = maplesdr_datamodule.train

class_count = dataset.get_class_count(load=False, save=True, key='label')

print(class_count)
class_weights = class_weighting(class_count=class_count)
print(class_weights)