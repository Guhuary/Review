# 记录大四学年的论文阅读经历

## 写在前面

&nbsp;&nbsp;&nbsp;&nbsp;这个项目是用来对前两年读过的论文做一下整理，回顾本科期间所阅读过的论文以及整理之后要读的论文，并给出论文链接以及代码，其中文本图像生成是一个大专题，比较完整，整理后放在了`./text2image/README.md`文件中。

&nbsp;&nbsp;&nbsp;&nbsp;本人本科期间主要从事人工智能在图像处理上的应用，在图像增强上有做过单图超分（Single Image Super-Resolution, SISR），单图去雨（Single Image Deraining）,图神经网络（Graph Convolutional Networks, GCNs）以及跨模态图像文本生成（Text2Image，T2I）。现从事多模态（Multi-Model），生成对抗网络（Generative Adversarial Network, GAN）以及小样本（Few-shot Learning）的研究。`GAN`以及`Sampling Methods`的论文放在了`./GAN&GAN and Sampling Methods/README.md`中。

&nbsp;&nbsp;&nbsp;&nbsp;正常情况下每周对论文会有所更新。

## 这里先介绍几个很好用的学习项目(不需要全学)

1. [卷积的概念以及基础](https://www.bilibili.com/video/BV1F4411y7o7?from=search&seid=4145305518394585094)，主要是前两个章节，有兴趣的可以多看看。

2. [吴恩达机器学习](https://www.bilibili.com/video/BV164411b7dx?p=112)，主要是第9-11章节梯度回传部分，有兴趣的可以考虑都看完。机器学习完整版的学习看吴恩达和李宏毅的均可以。

3. [Pytorch一小时入门](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)，这是Pytorch的简单入门，一些基础操作，很快就可以看完了。

4. [ResNet](https://arxiv.org/abs/1512.03385)，ResNet是CNN中的经典框架，`ResNet_CIFAR10.py`是一个简易的图像分类代码。

`CUDA_VISIBLE_DEVICES=0 python ResNet_CIFAR10.py`

5. [PyTorch example](https://github.com/pytorch/examples)，这是PyTorch的官方模板平台，学习PyTorch高端操作最好的样例。

6. [如何读论文——论文精读](https://www.bilibili.com/video/BV1H44y1t75x)，这是李沐大神在B站分享的项目，持续跟进可以很好规范自己的科研习惯。以及[李沐大神教你学AI](https://space.bilibili.com/1567748478?spm_id_from=333.788.b_765f7570696e666f.1)。

7. 李宏毅老师关于GAN的介绍，[B站视频链接](https://www.bilibili.com/video/BV1Up411R7Lk)


## Entity Resolution (Table) 
### 杂七杂八的文章

|Year   | Title  | Model  | Key Notes |Publication| Paper | Code |Have Read?(Y/N)|
|-------|--------|--------|--------|--------|--------|--------|--------|
|2016|**Deep Residual Learning for Image Recognition**| ResNet|ResBlock|```CVPR```|[PDF](https://arxiv.org/abs/1512.03385)||Y|
|2016|**Identity Mappings in Deep Residual Networks**|||`arXiv`|[PDF](https://arxiv.org/abs/1603.05027)|[Code](https://github.com/KaimingHe/resnet-1k-layers)|Y
|2020|**PointRend: Image Segmentation as Rendering**| PointRend|Segmentation|```CVPR```|[PDF](https://arxiv.org/abs/1912.08193)|[Code](https://github.com/zsef123/PointRend-PyTorch)|N|
|2018|**The Perception-Distortion Tradeoff**|||```CVPR```|[PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf)||Y
|2017|**Feedback Networks**||Feedback|```CVPR```|[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zamir_Feedback_Networks_CVPR_2017_paper.pdf)|[Code](https://github.com/StanfordVL/feedback-networks)|Y
|2018|**Feedback Convolutional Neural Network for Visual Localization and Segmentation**|FBCNN||```TPAMI```|[PDF](https://ieeexplore.ieee.org/document/8370896)|[Code](https://github.com/caochunshui/Feedback-CNN)|Y
|2020|**Batch Normalization Biases Deep Residual Networks Towards Shallow Paths**|||```NIPS```|[PDF](https://proceedings.neurips.cc/paper/2020/file/e6b738eca0e6792ba8a9cbcba6c1881d-Paper.pdf)||Y
|2020|**Designing Network Design Spaces**|||`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Radosavovic_Designing_Network_Design_Spaces_CVPR_2020_paper.pdf)|[Code](https://github.com/facebookresearch/pycls)|Y
|2017|**Wasserstein GAN**|WGAN|GAN|`arXiv`|[PDF](https://arxiv.org/abs/1701.07875)|[Code](https://github.com/martinarjovsky/WassersteinGAN)|Y
|2017|**Improved Training of Wasserstein GANs**|WGAN-GP|GAN|`arXiv`|[PDF](https://arxiv.org/abs/1704.00028)|[Code](https://github.com/igul222/improved_wgan_training)|Y
|2018|**Spectral Normalization for Generative Adversarial Networks**|SPNorm|GAN|`ICLR`|[PDF](https://openreview.net/forum?id=B1QRgziT-)||Y
|2017|**Densely Connected Convolutional Networks**|DenseNet||`CVPR`|[PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)|[Code](https://github.com/liuzhuang13/DenseNet)|Y
|2021|**Hybrid-attention guided network with multiple resolution features for person re-identification**||ReID|`INS`|[PDF](https://www.sciencedirect.com/science/article/pii/S0020025521007489)||N
|2020|**ActBERT: Learning Global-Local Video-Text Representations**|ActBERT||`arXiv`|[PDF](https://arxiv.org/abs/2011.07231)||Y
|2020|**Unbiased Scene Graph Generation from Biased Training**|SGG|Scene Graph|`CVPR`|[PDF](https://arxiv.org/abs/2002.11949)|[Code](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)|Y
|2019|**Adversarial Feedback Loop**|AFL|GAN|`ICCV`|[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Shama_Adversarial_Feedback_Loop_ICCV_2019_paper.pdf)|[Code](https://github.com/shamafiras/AFL)|Y
|2020|**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**|||`ECCV`|[PDF](https://arxiv.org/abs/2003.08934)|[Code](https://github.com/bmild/nerf)|Y
|2021|**Rethinking “Batch” in BatchNorm**|||`arXiv`|[PDF](https://arxiv.org/abs/2105.07576)||Y
|2018|**Unsupervised Feature Learning via Non-Parametric Instance Discrimination**|||`CVPR`|[PDF](https://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)|[Code](https://link.zhihu.com/?target=https%3A//github.com/zhirongw/lemniscate.pytorch)|Y





### GCN
|Year   | Title  | Model  | Key Notes |Publication| Paper | Code |Have Read?(Y/N)|
|-------|--------|--------|--------|--------|--------|--------|--------|
|2019|**Can GCNs Go as Deep as CNNs?**|DeepGCN|Point cloud segmentation|```ICCV```|[PDF](https://arxiv.org/abs/1904.03751)|[Code](https://github.com/lightaime/deep_gcns)|Y
|2020|**Feedback Graph Convolutional Network for Skeleton-based Action Recognition**||GCN, FB|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.html)||Y
|2020|**PairNorm: Tackling Oversmoothing in GNNs**|PairNorm|GCN|`ICLR`|[PDF](https://openreview.net/forum?id=rkecl1rtwB)|[Code](https://github.com/LingxiaoShawn/PairNorm)|Y
|2018|**Residual Gated Graph ConvNets**||GCN|`ICLR`|[PDF](https://openreview.net/forum?id=HyXBcYg0b)|[Code](https://github.com/xbresson/spatial_graph_convnets)|Y
|2019|**On Asymptotic Behaviors of Graph CNNs from Dynamical Systems Perspective**||GCN|`CoRR`|[PDF](https://openreview.net/forum?id=mPLlSuOPxJh)||N
|2020|**Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View**|MADGAP|GCN|`AAAI`|[PDF](https://arxiv.org/abs/1909.03211)|[Code](https://github.com/victorchen96/MadGap)|Y
|2017|**Inductive Representation Learning on Large Graphs**|GraphSAGE|GCN|`NIPS`|[PDF](https://proceedings.neurips.cc/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf)|[Code](http://snap.stanford.edu/graphsage/)|Y
|2018|**Graph Attention Networks**|GAT|GCN|`ICLR`|[PDF](https://arxiv.org/abs/1710.10903)|[Code](https://github.com/PetarV-/GAT)|Y
|2020|**Benchmarking Graph Neural Networks**||GCN|`TNNLS`|[PDF](https://arxiv.org/abs/2003.00982)|[Code](https://github.com/graphdeeplearning/benchmarking-gnns)|Y
|2019|**DropEdge: Towards Deep Graph Convolutional Networks on Node Classification**|DropEdge|GCN|`ICLR`|[PDF](https://openreview.net/forum?id=Hkx1qkrKPr)|[Code](https://github.com/DropEdge/DropEdge)|Y
|2020|**When Does Self-Supervision Help Graph Convolutional Networks?**||GCN|`ICML`|[PDF](https://arxiv.org/abs/2006.09136)|[Code](https://github.com/Shen-Lab/SS-GCNs)|Y
|2019|**Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks**|Cluster-GCN|GCN|`KDD`|[PDF](https://arxiv.org/abs/1905.07953)|[Code](https://github.com/zhengjingwei/cluster_GCN)|Y
|2020|**Simple and Deep Graph Convolutional Networks**|GCNII|GCN|`ICML`|[PDF](https://arxiv.org/abs/2007.02133)|[Code](https://github.com/chennnM/GCNII)|Y
|2021|**Isometric Transformation Invariant and Equivariant Graph Convolutional Networks**|IsoGCN|GCN|`ICLR`|[PDF](https://openreview.net/forum?id=FX0vR39SJ5q)|[Code](https://github.com/yellowshippo/isogcn-iclr2021)|N
|2019|**Attention Models in Graphs: A Survey**||GCN|`KDD`|[PDF](https://arxiv.org/abs/1807.07984)||N
|2019|**Graph Convolutional Networks for Temporal Action Localization**|PGCN|GCN|`ICCV`|[PDF](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zeng_Graph_Convolutional_Networks_for_Temporal_Action_Localization_ICCV_2019_paper.pdf)|[Code](https://github.com/Alvin-Zeng/PGCN)|N
|2018|**Attention-based Graph Neural Network for Semi-supervised Learning**|AGNN|GCN|`ICLR`|[PDF](https://openreview.net/forum?id=rJg4YGWRb)||Y
|2019|**Understanding Attention and Generalization in Graph Neural Networks**||GCN|`arXiv`|[PDF](https://arxiv.org/abs/1905.02850)|[Code](https://github.com/bknyaz/graph_attention_pool)|Y
|2021|**Fast Graph Attention Networks Using Effective Resistance Based Graph Sparsification**||GAT|`ICLR`|[PDF](https://openreview.net/forum?id=sjGBjudWib)||N
|2021|**Non-Local Graph Neural Networks**|NLGCN|GCN|`ICLR`|[PDF](https://openreview.net/forum?id=heqv8eIweMY)||N
|2021|**When Do GNNs Work: Understanding and Improving Neighborhood Aggregation**||GCN|`IJCAI`|[PDF](https://www.ijcai.org/proceedings/2020/181)||Y
|2019|**SPAGAN: Shortest Path Graph Attention Network**|SPAGAN|GCN|`IJCAI`|[PDF](https://arxiv.org/abs/2101.03464)|[Code](https://github.com/ihollywhy/SPAGAN)|Y
|2020|**LFGCN: Levitating over Graphs with Levy Flights**|LFGCN|GCN|`ICDM`|[PDF](https://arxiv.org/abs/2009.02365)||N
|2019|**Graph Representation Learning via Hard and Channel-Wise Attention Networks**||GCN|`KDD`|[PDF](https://dl.acm.org/doi/10.1145/3292500.3330897)||Y
|2021|**MULTI-HOP ATTENTION GRAPH NEURAL NETWORKS**|MAGNA|GCN|`ICLR`|[PDF](https://openreview.net/pdf?id=muppfCkU9H1)||Y
|2020|**MGAT: Multimodal Graph Attention Network for Recommendation**|MGAT||`IPM`|[PDF](https://www.sciencedirect.com/science/article/pii/S0306457320300182)||Y
|2018|**Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning**||GCN|`AAAI`|[PDF](https://arxiv.org/abs/1801.07606)||N
|2019|**Simplifying Graph Convolutional Networks**|SGC|GCN|`ICML`|[PDF](https://arxiv.org/abs/1902.07153)|[Code](https://github.com/Tiiiger/SGC)|Y
|2021|**Should Graph Convolution Trust Neighbors? A Simple Causal Inference Method**||GCN|`SIGIR`|[PDF](https://arxiv.org/abs/2010.11797)||Y
|2017|**SEMI-SUPERVISED CLASSIFICATION WITH GRAPH CONVOLUTIONAL NETWORKS**|GCN|GCN|`ICLR`|[PDF](https://openreview.net/pdf?id=SJU4ayYgl)|[Code](https://github.com/tkipf/gcn)|Y
|2020|**Open Graph Benchmark: Datasets for Machine Learning on Graphs**|OGB|GCN|`arXiv`|[PDF](https://arxiv.org/abs/2005.00687)|[Code](https://github.com/snap-stanford/ogb)|Y
|2021|**Zero-shot Synthesis with Group-Supervised Learning**||GCN|`ICLR`|[PDF](https://openreview.net/forum?id=8wqCDnBmnrT)||Y





### Image Enhencement
|Year   | Title  | Model  | Key Notes |Publication| Paper | Code |Have Read?(Y/N)|
|-------|--------|--------|--------|--------|--------|--------|--------|
|2021|**Distributed feedback network for single-image deraining**|DFN|Derain|`INS`|[PDF](https://www.sciencedirect.com/science/article/pii/S0020025521002371)|[Code](https://github.com/Guhuary/DFN)|Y
|2019|**EDVR: Video Restoration with Enhanced Deformable Convolutional Networks**|EDVR|SR|```CVPR```|[PDF](https://arxiv.org/pdf/1905.02716.pdf)|[Code](https://github.com/xinntao/EDVR)|Y
|2018|**ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks**|ESRGAN|SR|`ECCV`|[PDF](https://arxiv.org/abs/1809.00219)|[Code](https://github.com/xinntao/ESRGAN)|Y
|2020|**Self-Learning Video Rain Streak Removal: When Cyclic Consistency Meets Temporal Correspondence**|SLDNet|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_Self-Learning_Video_Rain_Streak_Removal_When_Cyclic_Consistency_Meets_Temporal_CVPR_2020_paper.html)|[Code](https://pythonrepo.com/repo/flyywh-CVPR-2020-Self-Rain-Removal-python-deep-learning)|Y
|2020|**Single Image Deraining Using Bilateral Recurrent Network**|BRN|Derain|`TIP`|[PDF](https://ieeexplore.ieee.org/document/9096546)|[Code](https://github.com/csdwren/RecDerain)|Y
|2020|**Multi-Scale Progressive Fusion Network for Single Image Deraining**|MSPFN|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Jiang_Multi-Scale_Progressive_Fusion_Network_for_Single_Image_Deraining_CVPR_2020_paper.pdf)|[Code](https://github.com/kuijiang0802/MSPFN)|Y
|2018|**Residual Dense Network for Image Super-Resolution**|RDB|SR|`CVPR`|[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Residual_Dense_Network_CVPR_2018_paper.pdf)|[Code](https://github.com/yulunzhang/RDN)|Y
|2019|**Second-order Attention Network for Single Image Super-Resolution**|SAN|SR|`CVPR`|[PDF](https://ieeexplore.ieee.org/document/8954252)|[Code](https://github.com/daitao/SAN)|Y
|2019|**Spatial Attentive Single-Image Deraining with a High Quality Real Rain Dataset**|SPANet|Derain|`CVPR`|[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Spatial_Attentive_Single-Image_Deraining_With_a_High_Quality_Real_Rain_CVPR_2019_paper.pdf)|[Code](https://github.com/stevewongv/SPANet)|Y
|2012|**Making a ‘Completely Blind’ Image Quality Analyzer**||NIQE|`SPL`|[PDF](https://ieeexplore.ieee.org/document/6353522)|[Code](https://github.com/csjunxu/Bovik_NIQE_SPL2013)|Y
|2020|**DRD-Net: Detail-recovery Image Deraining via Context Aggregation Networks**|DRD-Net|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/papers/Deng_Detail-recovery_Image_Deraining_via_Context_Aggregation_Networks_CVPR_2020_paper.pdf)|[Code](https://github.com/Dengsgithub/DRD-Net)|Y
|2018|**Density-aware Single Image De-raining using a Multi-stream Dense Network**|DID-MDN|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Density-Aware_Single_Image_CVPR_2018_paper.pdf)|[Code](https://github.com/hezhangsprinter/DID-MDN)|Y
|2020|**Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images**||Deraining|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2020/html/Zou_Deep_Adversarial_Decomposition_A_Unified_Framework_for_Separating_Superimposed_Images_CVPR_2020_paper.html)|[Code](https://github.com/jiupinjia/Deep-adversarial-decomposition)|Y
|2014|**No-reference image quality assessment based on spatial and spectralentropies**||SSEQ|`Signal Processing: Image Communication`|[PDF](https://www.sciencedirect.com/science/article/pii/S0923596514000927)||Y
|2009|**Single Image Haze Removal Using Dark Channel Prior**||Dehazing|`CVPR`|[PDF](https://ieeexplore.ieee.org/document/5206515)|[Code](https://github.com/joyeecheung/dark-channel-prior-dehazing)|Y
|2020|**Cross-Scale Internal Graph Neural Network for image super resolution**||SR|`NIPS`|[PDF](https://arxiv.org/abs/2006.16673)|[Code](https://github.com/sczhou/IGNN)|Y
|2017|**Enhanced Deep Residual Networks for Single Image Super-Resolution**|EDSR|SR|`CVPR`|[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)|[Code](https://github.com/sanghyun-son/EDSR-PyTorch)|Y
|2019|**Gated Multiple Feedback Network for Image Super-Resolution**|GMFN|SR|`BMVC`|[PDF](https://arxiv.org/abs/1907.04253)|[Code](https://github.com/liqilei/GMFN)|Y
|2019|**Image Super-Resolution by Neural Texture Transfer**||SR|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Image_Super-Resolution_by_Neural_Texture_Transfer_CVPR_2019_paper.pdf)|[Code](https://github.com/ZZUTK/SRNTT)|Y
|2019|**Progressive Image Deraining Networks: A Better and Simpler Baseline**|PReNet|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Ren_Progressive_Image_Deraining_Networks_A_Better_and_Simpler_Baseline_CVPR_2019_paper.pdf)|[Code](https://github.com/csdwren/PReNet)|Y
|2019|**Feedback Network for Image Super-Resolution**|SRFBN|Derain|`CVPR`|[PDF](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.pdf)|[Code](https://github.com/Paper99/SRFBN_CVPR19)|Y
|2019|**Semi-supervised Transfer Learning for Image Rain Removal**|SSIR|Derain|`CVPR`|[PDF](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wei_Semi-Supervised_Transfer_Learning_for_Image_Rain_Removal_CVPR_2019_paper.pdf)|[Code](https://github.com/wwzjer/Semi-supervised-IRR)|Y
|2020|**Confidence Measure Guided Single Image De-raining**||Derain|`TIP`|[PDF](https://arxiv.org/abs/1909.04207)||N
|2020|**Rain O'er Me: Synthesizing real rain to derain with data distillation**||Deraining|`TIP`|[PDF](https://arxiv.org/abs/1904.04605)||N

