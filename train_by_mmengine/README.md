# 模型训练(mmengine)
## 数据集准备(以kinetics400-tiny为例)
```c
wget https://download.openmmlab.com/mmaction/kinetics400_tiny.zip
unzip kinetics400_tiny.zip
// 文件夹结构
'''
kinetics400_tiny
├── kinetics_tiny_train_video.txt
├── kinetics_tiny_val_video.txt
├── train
│   ├── 27_CSXByd3s.mp4
│   ├── 34XczvTaRiI.mp4
│   ├── A-wiliK50Zw.mp4
│   ├── D32_1gwq35E.mp4
│   ├── D92m0HsHjcQ.mp4
...
|   ├── u4Rm6srmIS8.mp4
|   └── y5Iu7XkTqV0.mp4
'''
```

## 修改配置文件
### 调用mmengine的Config
```c
from mmengine.config import Config
// 官方配置文件
cfg = Config.fromfile('[/home/wenjun.feng/action_recognize/mmaction2_library/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-50e_sthv2-rgb.py](https://github.com/open-mmlab/mmaction2/blob/main/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x16-50e_sthv2-rgb.py)')

```
### 修改对应的config
```c
// Modify dataset type and path
cfg.data_root = 'kinetics400_tiny/train/'
cfg.data_root_val = 'kinetics400_tiny/val/'
cfg.ann_file_train = 'kinetics400_tiny/kinetics_tiny_train_video.txt'
cfg.ann_file_val = 'kinetics400_tiny/kinetics_tiny_val_video.txt'

cfg.test_dataloader.dataset.ann_file = 'kinetics400_tiny/kinetics_tiny_val_video.txt'
cfg.test_dataloader.dataset.data_prefix.video = 'kinetics400_tiny/val/'

cfg.train_dataloader.dataset.ann_file = 'kinetics400_tiny/kinetics_tiny_train_video.txt'
cfg.train_dataloader.dataset.data_prefix.video = 'kinetics400_tiny/train/'

cfg.val_dataloader.dataset.ann_file = 'kinetics400_tiny/kinetics_tiny_val_video.txt'
cfg.val_dataloader.dataset.data_prefix.video  = 'kinetics400_tiny/val/'

// Modify num classes of the model in cls_head
// For kinetics-400-tiny only has 2 classes
cfg.model.cls_head.num_classes = 2
// We can use the pre-trained TSN model
// Here is pre-trained model weights url which can find in mmaction website
cfg.load_from = 'https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

// Set up working dir to save files and logs.
cfg.work_dir = './train_tsn_2'

// The original learning rate (LR) is set for 8-GPU training.
// We divide it by 8 since we only use one GPU.
// You can set LR scale here.
cfg.train_dataloader.batch_size = cfg.train_dataloader.batch_size 
cfg.val_dataloader.batch_size = cfg.val_dataloader.batch_size 
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr
cfg.train_cfg.max_epochs = 10

cfg.train_dataloader.num_workers = 2
cfg.val_dataloader.num_workers = 2
cfg.test_dataloader.num_workers = 2

// We can initialize the logger for training and have a look
// at the final config used for training
print(f'Config:\n{cfg.pretty_text}')
```
关于配置文件中的细节参考:https://github.com/Vvvvvvsysy/Action-Recognize/blob/main/README.md
## 模型训练&测试
### 调用mmengine的Runner
```c
import os.path as osp
import mmengine
from mmengine.runner import Runner

// Create work_dir
mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

// build the runner from config
runner = Runner.from_cfg(cfg)

// start training
runner.train()

// start testing
runner.test()
```










