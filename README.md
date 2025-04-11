# lookhere
1.模型来自：  
NeurIPS2024的论文《LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate》  
论文网址：https://proceedings.neurips.cc/paper_files/paper/2024/file/22d72e9f55bc29cafcca6814a7feac8c-Paper-Conference.pdf  
2.模型简介：  
VIT模型通过使用Transformer的位置编码方式来增加图片的位置信息，LookHere使用Directed Attention的方法来为模型提供位置信息，具体实现方法为(Q @ K^T)/(d_model**0.5) - LH。  
3.代码说明：  
原论文中注意力方向有更多组合，本代码中只有了["right","left","up","down"]四个方向，ViT模型把数据预处理之后的注释去掉也可以运行。数据集使用的MNIST。
