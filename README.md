# Deep-learning-based-point-cloud-alignment-and-feature-extraction-work
point_net, Data_read
本项目是基于点云的配准任务，通过回归旋转矩阵，从而完成配准目标。
特征提取backbone基于point_net，如需迁移到其他任务中，需要自己对网络做调整。
项目中封装了一些常用工具可供使用，在train_utils文件下。
dataset文件夹下的dataset.py文件可以用来读取以及划分验证集合，此处感谢霹雳巴拉Wz大佬的b站教学视频以及开源代码。
train.py文件是训练文件，直接运行即可。
