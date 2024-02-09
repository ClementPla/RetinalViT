import tqdm
from vitRet.data.fundus import EyePACSDataModule

data_dir = '/home/tmp/clpla/data/eyepacs/'
img_size = (1024, 1024)
scales = 1
datamodule = EyePACSDataModule(data_dir, img_size=img_size, batch_size=16, num_workers=16, superpixels_scales=scales, 
                               valid_size=4000,
                               superpixels_max_nb=8000,
                               superpixels_min_nb=2048,
                               superpixels_num_threads=-1)
datamodule.setup('fit')
datamodule.setup('test')


train_loader = datamodule.train_dataloader()
valid_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()



print(f'Train loader ({len(train_loader.dataset)} images)')
for batch in tqdm.tqdm(train_loader, total=len(train_loader)):
    pass

print(f'Valid loader ({len(valid_loader.dataset)} images)')
for batch in tqdm.tqdm(valid_loader, total=len(valid_loader)):
    pass

print(f'Test loader ({len(test_loader.dataset)} images)')
for batch in tqdm.tqdm(test_loader, total=len(test_loader)):
    pass