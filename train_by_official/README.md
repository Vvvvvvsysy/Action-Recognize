# 模型训练(官方脚本)
## 数据集准备
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
参考slowfast_config.py

配置文件参数解释参考:https://github.com/Vvvvvvsysy/Action-Recognize/blob/main/README.md
## 训练脚本
```c
cd train_script_official
python train_official.py --config ./slowfast_config.py --work-dir ./slowfast
```
## 测试脚本
```c
python test_official.py --config ./slowfast_config.py --checkpoint ./slowfast/epoch_10.pth --work-dir ./slowfast
```





