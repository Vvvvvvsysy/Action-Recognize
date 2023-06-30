# MMAction2---Action Recognize
## 环境配置
参考官方文档：https://mmaction2.readthedocs.io/zh_CN/0.x/install.html

1、新建python环境
```c
conda create -n mmaction python=3.8
conda activate mmaction
```
2、安装torch、torchvision
```c
pip install torch==1.13 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torchvision==0.14.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
3、安装MMAction2

3.1 方法1--使用mim安装
```c
pip install git+https://github.com/open-mmlab/mim.git
mim install mmaction2 -f https://github.com/open-mmlab/mmaction2.git
// 安装对应mmcv
// 参考：https://mmcv.readthedocs.io/en/latest/get_started/installation.html
min install mmcv -i https://pypi.tuna.tsinghua.edu.cn/simple
```
3.2 方法2---手动git安装
```c
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -r requirements/build.txt
python setup.py develop
// 安装mmcv
min install mmcv -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 测试环境是否搭建成功
```c
// git mmaction2项目文件
git clone https://github.com/open-mmlab/mmaction2.git

// 测试mmaction2 demo(以slowfast为例)
// 进入mmaction2目录
cd mmaction2
// 命令行输入
python demo/demo.py configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth demo/demo.mp4 tools/data/kinetics/label_map_k400.txt --out-filename demo/demo_out1.mp4
// 输出如下：
"""
The top-5 labels with corresponding scores are:
arm wrestling:  0.9999974966049194
dying hair:  4.0872896533983294e-07
opening bottle:  2.254595159456585e-07
drinking:  1.9572580356452818e-07
shaking hands:  1.944649739016313e-07
---demo/demo_out1.mp4是输出文件
"""
```
输出以上结果，证明环境安装成功。

## 准备训练数据(以ucf101为例)
动作识别的数据集类别和介绍整理到：https://github.com/Vvvvvvsysy/Action-Recognize/tree/main/dataset

```c
// 进入mmaction2/tools/data/ucf101目录
// 下载各种数据的脚本都在mmaction2/tools/data
cd mmaction2/tools/data/ucf101
// 下载annotations(.sh文件中的路径要适当修改)
bash download_annotations.sh
// 生成train_val_split
bash generate_videos_filelist.sh
// 下载videos(.sh文件中的路径要适当修改)
bash download_videos.sh

// 最后数据集文件夹目录结构如下
"""
/data（自己创建）
----/videos（存放原始视频）
--------/class_1
------------Class_1Video_1.mp4
------------Class_1Video_2.mp4
--------/class_2
------------Class_2Video_1.mp4
------------Class_2Video_2.mp4
----/annotations（存放标注信息）
--------classInd.txt  // 分类类别
--------trainlist.txt // 训练集
--------testlist.txt  // 测试集
----/train_val_split  // 数据划分文件使用trainlist.txt和testlist.txt需要通过generate_videos_filelist.sh得到
--------ucf101_train_split_1_videos.txt // training data
--------ucf101_val_split_1_videos.txt   // testing data
"""
```
## 模型训练(用slowfast在ucf101数据上微调)
进入mmaction2目录，修改对应模型的配置文件，以slowfast为例：
```c
_base_ = ['../../_base_/default_runtime.py'] // 继承的hook类

num_classe = 101  // 分类类别数

// model settings 这部分除了cls_head以外一般不做修改
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  // tau
        speed_ratio=8,  // alpha
        channel_ratio=8,  // beta_inv
        slow_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            type='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False)),
    cls_head=dict(
        type='SlowFastHead',
        in_channels=2304,  // 2048+256
        num_classes=num_classe, // 这里需要根据使用的数据进行修改
        spatial_type='avg',
        dropout_ratio=0.5,
        average_clips='prob'),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        format_shape='NCTHW'))

// 数据集配置
dataset_type = 'VideoDataset'  // VideoDataset / FrameDataset
data_root = '/dataset/xcyuan/ucf101/UCF-101/' // 训练集根目录
data_root_val = '/dataset/xcyuan/ucf101/UCF-101/' // 验证集根目录
// 数据划分文件
ann_file_train = '/home/wenjun.feng/action_recognize/ucf101/ucf101_train_split_1_videos.txt'
ann_file_val = '/home/wenjun.feng/action_recognize/ucf101/ucf101_val_split_1_videos.txt'
ann_file_test = '/home/wenjun.feng/action_recognize/ucf101/ucf101_val_split_1_videos.txt'

file_client_args = dict(io_backend='disk')
// 训练数据前处理pipeline
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    // 视频帧采样策略 采样1个视频片段，片段长度为32，间隔2帧采样
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1), 
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)), 
    dict(type='RandomResizedCrop'),
    dict(type='Resize', scale=(224, 224), keep_ratio=False), // resize
    dict(type='Flip', flip_ratio=0.5), // 翻转
    dict(type='FormatShape', input_format='NCTHW'), // 输出格式
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='SampleFrames', clip_len=32, frame_interval=2, num_clips=1, test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    // 测试数据采样策略会和训练时不同
    // 往往采样的视频片段数更多，这里是10段
    dict(type='SampleFrames',clip_len=32,frame_interval=2,num_clips=10,test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='ThreeCrop', crop_size=256),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='PackActionInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type=dataset_type,ann_file=ann_file_train,data_prefix=dict(video=data_root),pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type,ann_file=ann_file_val,data_prefix=dict(video=data_root_val),pipeline=val_pipeline,test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type=dataset_type,ann_file=ann_file_test,data_prefix=dict(video=data_root_val),pipeline=test_pipeline,test_mode=True))

val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

// val_begin：验证开始轮次
// val_interval：验证模型的间隔轮次
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_begin=1, val_interval=2)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

// 优化器设置
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=1e-4),
    clip_grad=dict(max_norm=40, norm_type=2))

// lr更新策略
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=True,
        begin=0,
        end=34,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=256,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=256)
]
// 模型保存间隔和允许保存的最多模型  logger输出间隔
default_hooks = dict(checkpoint=dict(interval=2, max_keep_ckpts=3), logger=dict(interval=100))
// 对应模型参数下载url，可以在mmaction官方github上找到
load_from = "https://download.openmmlab.com/mmaction/v1.0/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb_20220901-701b0f6f.pth"
```
模型训练，参数依次为 配置文件-使用的GPU数量-工作区保存地址
```c
bash tools/dist_train.sh configs/recognition/slowfast/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py 4 --work-dir /home/wenjun.feng/action_recognize/ucf101
```
模型测试，参数依次为 配置文件-模型checkpoint-GPU数量-工作区保存地址
```c
bash mmaction2_library/tools/dist_test.sh ucf101/slowfast_100epoch/slowfast_r50_8xb8-4x16x1-256e_kinetics400-rgb.py ucf101/slowfast_100epoch/20230627_192647/best_acc_top1_epoch_20.pth 8  --work-dir ucf101/slowfast_100epoch/test_res/
// 验证集结果："acc/top1": 0.9497753106000528, "acc/top5": 0.9955062120010574
// 测试集结果："acc/top1": 0.9608776103621465, "acc/top5": 0.9952418715305313
// 这里因为模型验证和测试时对视频的帧采样策略不同
// 测试数据的采样更密集 结果会更好一点
```
模型复杂度分析
```c
// 输出模型的参数量和在指定输入数据下的计算量
python tools/analysis_tools/get_flops.py {config path}
```
## some baselines
| model        | frame sampleing strategy | resoution      | backbone         | pretrain               | epoch  | val protocol                       | top1-acc-val | top5-acc-val | test protocol                | top1-acc-test | top5-acc-test | parameters | FLOPs             |
| ------------ | ------------------------ | -------------- | ---------------- | ---------------------- | ------ | ---------------------------------- | ------------ | ------------ | ---------------------------- | ------------- | ------------- | ---------- | ----------------- |
| I3D          | 32×2×1                   | 224×224        | Res50-3D         | ImageNet + Kinetics400 | 84/100 | 1 clips x 1 center crop × 32frames | 91.01        | 98.30        | 10 clips × 3 crop × 32frames | 91.91         | 98.78         | 27.431M    | 33.271G(32frames) |
| SlowFast     | 32×2×1                   | 224×224        | Res50-3D         | Kinetics400            | 20/100 | 1 clips x 1 center crop × 32frames | 94.97        | 99.55        | 10 clips × 3 crop × 32frames | 96.08         | 99.52         | 33.79M     | 27.817G(32frames) |
| TimeSformer  | 8×32×1                   | 224×224        | ViT-B/16         | ImageNet-21K           | 19/30  | 1 clip × 1 center crop × 8frames   | 95.37        | 99.23        | 1 clip × 3 crop × 8frames    | 94.98         | 99.41         | 123.904M   | 200.7.4G(8frames) |
| Uniformer-v2 | uniform sampling-8frames | short side 320 | UniformerV2-B/16 | clip                   | 28/30  | 1 clip × 1 center crop × 8frames   | 97.36        | 99.78        | 4 clips × 3 crop × 8frames   | 97.51         | 99.78         | 116.736M   | 151.552G(8frames) |

## some problems
### 1、模型验证和测试用同样数据却得到不同的指标？
这个是因为模型对验证集和测试集的前处理pipeline不一样。一般在测试集处理上会抽取更多clips，对视频帧的采样更密集，想对来说会更有优势一点。
```c
// 以slowfast验证和测试采样策略为例
// 验证
dict(type='SampleFrames',clip_len=32,frame_interval=2,num_clips=1,test_mode=True)
// 测试
dict(type='SampleFrames',clip_len=32,frame_interval=2,num_clips=10,test_mode=True)
// 可以看出测试数据在帧采样时，采样的帧数和采样帧间隔一样，但是采样的片段数是验证数据的10倍。
```
### 2、在一台机器上同时训练两个模型时，报端口重复占用的错。
因为模型训练都是用的DDP的框架，启动DDP时会设置占用的端口号，一个端口号只能给一次训练使用。报这个错可以在在/home/mmaction2_library/tools/dist_train.sh文件中修改PORT参数，随便换一个没被占用的即可。
### 3、训练时日志输出的lr和配置文件中有差别。
```c
// 训uniformer v1的时候，学习率设置的1e-5但是实际上一直在1e-7左右，后面发现配置文件中有一项
auto_scale_lr = dict(enable=False, base_batch_size=256)
// auto_scale_lr如果是enable的话，实际训练的lr会对照现有的batchsize和base_batch_size来相应scale
```




