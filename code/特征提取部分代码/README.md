# 特征提取代码说明
## feature.py 和 vggish
    vggish为其项目源码的拷贝，且有单独下载的vggish_pca_params.npz文件作为模型输入需要，而vggish_model.ckpt文件由于大小超过github上所能承受的空间，故未上传。
    可参考tensorflow项目地址 https://github.com/tensorflow/models.git 中vggish相关部分。
    feature.py为特征提取代码，引用了viggish相关方法和模型，读取wav音频文件后按照平均值和整合值分别输出特征向量。
## librosa
    librosa为具体的音频特征分析代码，其中也包括了数据运行结果，具体见代码注释。
    sptg和mfcc提取的csv文件占用空间较大，暂不共享，如需要可通过代码中方法获取。
## pyaudio
    pyaudio为项目初始阶段的实验代码，并未纳入最终的使用。