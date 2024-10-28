# 🚀 Awesome-Graph-OOD-Learning

This repository for the paper 📘: [A Survey of Deep Graph Learning under Distribution Shifts: from Graph Out-of-Distribution Generalization to Adaptation](https://arxiv.org/pdf/2410.19265).
The README file here maintains a list of papers on graph out-of-distribution learning, covering three primary scenarios: **graph OOD generalization**, **training-time graph OOD adaptation**, and **test-time graph OOD adaptation**.

Check out our existing survey 📄: [Beyond Generalization: A Survey of Out-Of-Distribution Adaptation on Graphs](https://arxiv.org/pdf/2402.11153.pdf), which contains a list of papers on graph out-of-distribution adaptation.


## ❖ Contents 
- [Graph OOD Generalization](#Graph-OOD-Generalization)
    - [Model-centric Approaches](#Model-centric-Approaches)
    - [Data-centric Approaches](#Data-centric-Approaches)
- [Training-time Graph OOD Adaptation](#Training-time-Graph-OOD-Adaptation)
    - [Model-centric Approaches](#Model-centric-Approaches-1)
    - [Data-centric Approaches](#Data-centric-Approaches-1)
- [Test-time Graph OOD Adaptation](#Test-time-Graph-OOD-Adaptation)
    - [Model-centric Approaches](#Model-centric-Approaches-2)
    - [Data-centric Approaches](#Data-centric-Approaches-2)
- [Transferability evaluation](#Related-Transferability-evaluation)

## ❖ Graph OOD Generalization

### Model-centric Approaches
|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **DIR** | Invariant Representation Learning    | [[ICLR 2022] Discovering invariant rationales for graph neural networks](https://arxiv.org/pdf/2201.12872) | [Code](https://github.com/Wuyxin/DIR-GNN) |
| **GIL** | Invariant Representation Learning    | [[NeurIPS 2022] Learning invariant graph representations for out-of-distribution generalization](https://proceedings.neurips.cc/paper_files/paper/2022/file/4d4e0ab9d8ff180bf5b95c258842d16e-Paper-Conference.pdf) | [N/A] |
| **GSAT** | Invariant Representation Learning    | [[ICML 2022] Interpretable and generalizable graph learning via stochastic attention mechanism](https://proceedings.mlr.press/v162/miao22a/miao22a.pdf) | [Code](https://github.com/Graph-COM/GSAT) |
| **IS-GIB** | Invariant Representation Learning    | [[TKDE 2023] Individual and structural graph information bottlenecks for out-of-distribution generalization](https://arxiv.org/pdf/2306.15902) | [Code](https://github.com/YangLing0818/GraphOOD) |
| **MARIO** | Invariant Representation Learning    | [[TheWebConf 2024] Mario: Model agnostic recipe for improving ood generalization of graph contrastive learning](https://arxiv.org/pdf/2307.13055) | [Code](https://github.com/ZhuYun97/MARIO) |
| **DIVE** | Invariant Representation Learning    | [[KDD 2024] DIVE: Subgraph Disagreement for Graph Out-of-Distribution Generalization](https://arxiv.org/pdf/2408.04400) | [N/A] |
| **CAP** | Invariant Representation Learning    | [[arXiv] CAP: Co-Adversarial Perturbation on Weights and Features for Improving Generalization of Graph Neural Networks](https://arxiv.org/pdf/2110.14855) | [N/A] |
| **GraphAT** | Invariant Representation Learning    | [[TKDE 2019] Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure](https://arxiv.org/pdf/1902.08226) | [Code](https://github.com/fulifeng/GraphAT) |
| **GNN-DRO** | Invariant Representation Learning    | [[arXiv] Distributionally robust semi-supervised learning over graphs](https://arxiv.org/pdf/2110.10582) | [N/A] |
| **WT-AWP** | Invariant Representation Learning    | [[AAAI 2023] Adversarial weight perturbation improves generalization in graph neural networks](https://ojs.aaai.org/index.php/AAAI/article/view/26239/26011) | [N/A] |
| **DisenGCN** | Invariant Representation Learning    | [[ICML 2019] Disentangled graph convolutional networks](https://proceedings.mlr.press/v97/ma19a/ma19a.pdf) | [N/A] |
| **IPGDN** | Invariant Representation Learning    | [[AAAI 2020] Independence promoted graph disentangled networks](https://ojs.aaai.org/index.php/AAAI/article/view/5929/5785) | [N/A] |
| **FactorGCN** | Invariant Representation Learning    | [[NeurIPS 2020] Factorizable graph convolutional networks](https://proceedings.neurips.cc/paper_files/paper/2020/file/ea3502c3594588f0e9d5142f99c66627-Paper.pdf) | [Code](https://github.com/ihollywhy/FactorGCN.PyTorch) |
| **NED-VAE** | Invariant Representation Learning    | [[KDD 2020] Interpretable deep graph generation with node-edge co-disentanglement](https://dl.acm.org/doi/pdf/10.1145/3394486.3403221) | [Code](https://github.com/xguo7/NED-VAE) |
| **DGCL** | Invariant Representation Learning    | [[NeurIPS 2021] Disentangled contrastive learning on graphs](https://proceedings.neurips.cc/paper_files/paper/2021/file/b6cda17abb967ed28ec9610137aa45f7-Paper.pdf) | [N/A] |
| **IDGCL** | Invariant Representation Learning    | [[TKDE 2022] Disentangled graph contrastive learning with independence promotion](https://ieeexplore.ieee.org/abstract/document/9893319) | [N/A] |
| **I-DIDA** | Invariant Representation Learning    | [[arXiv] Out-of-distribution generalized dynamic graph neural network with disentangled intervention and invariance promotion](https://arxiv.org/pdf/2311.14255) | [N/A] |
| **L2R-GNN** | Invariant Representation Learning    | [[AAAI 2024] Learning to reweight for generalizable graph neural network](https://ojs.aaai.org/index.php/AAAI/article/view/28673/29307) | [N/A] |
| **OOD-GNN** | Invariant Representation Learning    | [[TKDE 2022] OOD-GNN: Out-of-Distribution Generalized Graph Neural Network](https://arxiv.org/pdf/2112.03806) | [N/A] |
| **EQuAD** | Invariant Representation Learning    | [[ICML 2024] Empowering Graph Invariance Learning with Deep Spurious Infomax](https://arxiv.org/pdf/2407.11083) | [Code](https://github.com/tianyao-aka/EQuAD) |
| **DIDA** | Invariant Representation Learning    | [[NeurIPS 2022] Dynamic graph neural networks under spatio-temporal distribution shift](https://proceedings.neurips.cc/paper_files/paper/2022/file/2857242c9e97de339ce642e75b15ff24-Paper-Conference.pdf) | [Code](https://github.com/wondergo2017/DIDA) |
| **EAGLE** | Invariant Representation Learning    | [[NeurIPS 2024] Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/9bf12308ece130daa083fb21f7faf1b6-Paper-Conference.pdf) | [Code](https://github.com/RingBDStack/EAGLE) |
| **CauSTG** | Invariant Representation Learning    | [[KDD 2023] Maintaining the status quo: Capturing invariant relations for ood spatiotemporal learning](https://dl.acm.org/doi/pdf/10.1145/3580305.3599421) | [Code](https://github.com/zzyy0929/KDD23-CauSTG) |
| **STONE** | Invariant Representation Learning    | [[KDD 2024] STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts](http://home.ustc.edu.cn/~zzy0929/Home/Paper/KDD24_STONE.pdf) | [Code](https://github.com/PoorOtterBob/STONE-KDD-2024) |
| **E-invariant GR** | Causality-based Learning   | [[ICML 2021] Size-invariant graph representations for graph classification extrapolations](https://proceedings.mlr.press/v139/bevilacqua21a/bevilacqua21a.pdf) | [Code](https://github.com/PurdueMINDS/size-invariant-GNNs) |
| **CIGA** | Causality-based Learning   | [[NeurIPS 2022] Learning causally invariant representations for out-of-distribution generalization on graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/8b21a7ea42cbcd1c29a7a88c444cce45-Paper-Conference.pdf) | [Code](https://github.com/LFhase/CIGA) |
| **CAL** | Causality-based Learning   | [[KDD 2022] Causal attention for interpretable and generalizable graph classification](https://arxiv.org/pdf/2112.15089) | [Code](https://github.com/yongduosui/CAL) |
| **DisC** | Causality-based Learning   | [[NeurIPS 2022] Debiasing Graph Neural Networks via Learning Disentangled Causal Substructure](https://proceedings.neurips.cc/paper_files/paper/2022/file/9e47a0bc530cc88b09b7670d2c130a29-Paper-Conference.pdf) | [Code](https://github.com/googlebaba/DisC) |
| **GALA** | Causality-based Learning   | [[NeurIPS 2024] Does invariant graph learning via environment augmentation learn invariance?](https://proceedings.neurips.cc/paper_files/paper/2023/file/e21a7b668ce3ea2c9c964c52d1c9f161-Paper-Conference.pdf) | [Code](https://github.com/LFhase/GALA) |
| **LECI** | Causality-based Learning  | [[NeurIPS 2024] Joint Learning of Label and Environment Causal Independence for Graph Out-of-Distribution Generalization](https://proceedings.neurips.cc/paper_files/paper/2023/file/0c6c92a0c5237761168eafd4549f1584-Paper-Conference.pdf) | [Code](https://github.com/divelab/LECI) |
| **StableGNN** | Causality-based Learning  | [[TPAMI 2023] Generalizing graph neural networks on out-of-distribution graphs](https://arxiv.org/pdf/2111.10657) | [Code](https://github.com/googlebaba/StableGNN) |
| **Pretraining-GNN** | Graph Self-supervised Learning    | [[ICLR 2020] Strategies for pre-training graph neural networks](https://arxiv.org/pdf/1905.12265)  | [N/A] |
| **PATTERN** | Graph Self-supervised Learning    | [[ICML 2021] From local structures to size generalization in graph neural networks](https://proceedings.mlr.press/v139/yehudai21a/yehudai21a.pdf) | [N/A] |
| **OOD-GCL** | Graph Self-supervised Learning    | [[ICML 2024] Disentangled Graph Self-supervised Learning for Out-of-Distribution Generalization](https://openreview.net/pdf?id=OS0szhkPmF) | [N/A] |
| **GPPT** | Graph Self-supervised Learning   | [[KDD 2022] GPPT: Graph Pre-training and Prompt Tuning to Generalize Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3534678.3539249) | [Code](https://github.com/MingChen-Sun/GPPT) |
| **GPF** | Graph Self-supervised Learning    | [[NeurIPS 2024] Universal prompt tuning for graph neural networks](https://proceedings.neurips.cc/paper_files/paper/2023/file/a4a1ee071ce0fe63b83bce507c9dc4d7-Paper-Conference.pdf) | [Code](https://github.com/zjunet/GPF) |
| **GraphControl** | Graph Self-supervised Learning   | [[TheWebConf 2024] GraphControl: Adding Conditional Control to Universal Graph Pre-trained Models for Graph Domain Transfer Learning](https://arxiv.org/pdf/2310.07365) | [Code](https://github.com/wykk00/GraphControl) |

### Data-centric Approaches

|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **AIA** | Graph Data Augmentation    | [[NeurIPS 2023] Unleashing the power of graph data augmentation on covariate distribution shift](https://proceedings.neurips.cc/paper_files/paper/2023/file/3a33ddacb2798fc7d83b8334d552e05a-Paper-Conference.pdf) | [Code](https://github.com/yongduosui/AIA) |
| **G-Splice** | Graph Data Augmentation    | [[arXiv] Graph structure and feature extrapolation for out-of-distribution generalization](https://arxiv.org/pdf/2306.08076) | [N/A] |
| **LiSA** | Graph Data Augmentation    | [[CVPR 2023] Mind the Label Shift of Augmentation-based Graph OOD Generalization](https://openaccess.thecvf.com/content/CVPR2023/papers/Yu_Mind_the_Label_Shift_of_Augmentation-Based_Graph_OOD_Generalization_CVPR_2023_paper.pdf) | [Code](https://github.com/Samyu0304/LiSA) |
| **DLG** | Graph Data Augmentation    | [[ICDM 2024] Enhancing Distribution and Label Consistency for Graph Out-of-Distribution Generalization] | [N/A] |
| **Pattern-PT** | Graph Data Augmentation    | [[ICML 2021] From local structures to size generalization in graph neural networks](https://proceedings.mlr.press/v139/yehudai21a/yehudai21a.pdf) | [N/A] |
| **P-gMPNN** | Graph Data Augmentation    | [[NeurIPS 2022] OOD link prediction generalization capabilities of message-passing GNNs in larger test graphs](https://proceedings.neurips.cc/paper_files/paper/2022/file/7f88a8478c4ae97819ccffa1e80e7a7b-Paper-Conference.pdf) | [Code](https://github.com/yangzez/OOD-Link-Prediction-Generalization-MPNN) |
| **GraphMix** | Graph Data Augmentation    | [[AAAI 2021] Graphmix: Improved training of gnns for semi-supervised learning](https://ojs.aaai.org/index.php/AAAI/article/view/17203/17010) | [Code](https://github.com/vikasverma1077/GraphMix) |
| **G-Mixup** | Graph Data Augmentation    | [[TheWebConf 2021] Mixup for node and graph classification](https://dl.acm.org/doi/pdf/10.1145/3442381.3449796) | [N/A] |
| **$\mathcal{G}$-Mixup** | Graph Data Augmentation    | [[ICML 2022] G-mixup: Graph data augmentation for graph classification](https://proceedings.mlr.press/v162/han22c/han22c.pdf) | [Code](https://github.com/ahxt/g-mixup) |
| **OOD-GMixup** | Graph Data Augmentation    | [[TKDE 2024] Graph out-of-distribution generalization with controllable data augmentation](https://arxiv.org/pdf/2308.08344) | [N/A] |
| **GREA** | Distribution Augmentation    | [[KDD 2022] Graph rationalization with environment-based augmentations](https://dl.acm.org/doi/pdf/10.1145/3534678.3539347) | [Code](https://github.com/liugangcode/GREA) |
| **EERM** | Distribution Augmentation    | [[ICLR 2022] Handling distribution shifts on graphs: An invariance perspective](https://arxiv.org/pdf/2202.02466) | [Code](https://github.com/qitianwu/GraphOOD-EERM) |
| **FLOOD** | Distribution Augmentation    | [[KDD 2023] FLOOD: A Flexible Invariant Learning Framework for Out-of-Distribution Generalization on Graphs](https://dl.acm.org/doi/pdf/10.1145/3580305.3599355) | [N/A] |
| **DPS** | Distribution Augmentation    | [[arXiv] Finding Diverse and Predictable Subgraphs for Graph Domain Generalization](https://arxiv.org/pdf/2206.09345) | [N/A] |
| **MoleOOD** | Distribution Augmentation    | [[NeurIPS 2022] Learning substructure invariance for out-of-distribution molecular representations](https://proceedings.neurips.cc/paper_files/paper/2022/file/547108084f0c2af39b956f8eadb75d1b-Paper-Conference.pdf) | [Code](https://github.com/yangnianzu0515/MoleOOD) |
| **ERASE** | Distribution Augmentation    | [[CIKM 2024] ERASE: Error-Resilient Representation Learning on Graphs for Label Noise Tolerance](https://arxiv.org/pdf/2312.08852) | [Code](https://github.com/eraseai/erase) |
| **IGM** | Distribution Augmentation    | [[AAAI 2024] Graph invariant learning with subgraph co-mixup for out-of-distribution generalization](https://ojs.aaai.org/index.php/AAAI/article/download/28700/29356) | [Code](https://github.com/BUPT-GAMMA/IGM) |




## ❖ Training-time Graph OOD Adaptation

### Model-centric Approaches
|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **DAGNN** | Invariant Representation Learning    | [[ICDM 2019] Domain-Adversarial Graph Neural Networks for Text Classification](https://shiruipan.github.io/publication/icdm-19-wu/icdm-19-wu.pdf) | [N/A] |
| **DANE**  | Invariant Representation Learning    |  [[ICJAI 2019] DANE: Domain Adaptive Network Embedding](https://arxiv.org/pdf/1906.00684.pdf)  |  [Unofficial](https://github.com/Jerry2398/DANE-Simple-implementation?tab=readme-ov-file) |
| **CDNE** | Invariant Representation Learning    |    [[TNNLS 2020] Network Together: Node Classification via Cross-Network Deep Network Embedding](https://arxiv.org/pdf/1901.07264.pdf)      |  [Code](https://github.com/shenxiaocam/CDNE) |
| **ACDNE** |  Invariant Representation Learning    |  [[AAAI 2020] Adversarial Deep Network Embedding for Cross-network Node Classification](https://arxiv.org/pdf/2002.07366.pdf)  | [Code](https://github.com/shenxiaocam/ACDNE) |
| **UDA-GCN** | Invariant Representation Learning    |  [[TheWebConf 2020] Unsupervised Domain Adaptive Graph Convolutional Networks](https://shiruipan.github.io/publication/www-2020-wu/www-2020-wu.pdf)   | [Code](https://github.com/GRAND-Lab/UDAGCN) |
| **DGDA** |  Invariant Representation Learning    |  [[TKDD 2024] Graph domain adaptation: A generative view.](https://arxiv.org/pdf/2106.07482.pdf)  |  [Code](https://github.com/rynewu224/GraphDA) |
| **SR-GNN** | Invariant Representation Learning    |  [[NeurIPS 2021] Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data](https://arxiv.org/pdf/2108.01099.pdf)  | [Code](https://github.com/GentleZhu/Shift-Robust-GNNs) |
| **ASN** | Invariant Representation Learning    |  [ [CIKM 2021] Adversarial separation network for cross-network node classification](https://dl.acm.org/doi/abs/10.1145/3459637.3482228) | [Code](https://github.com/yuntaodu/ASN) |
| **AdaGCN** |Invariant Representation Learning    |  [[TKDE 2022] Graph transfer learning via adversarial domain adaptation with graph convolution](https://arxiv.org/pdf/1909.01541.pdf) | [Code](https://github.com/daiquanyu/AdaGCN_TKDE) |
| **GraphAE** |Invariant Representation Learning    |  [[TKDE 2023] Learning adaptive node embeddings across graphs](https://ieeexplore.ieee.org/document/9737419) |  [N/A] |
| **GRADE** |Invariant Representation Learning    |  [ [AAAI 2023] Non-iid transfer learning on graphs](https://arxiv.org/pdf/2212.08174.pdf)| [Code](https://github.com/jwu4sml/GRADE) | 
| **JHGDA** |Invariant Representation Learning    |  [ [CIKM 2023] Improving graph domain adaptation with network hierarchy](https://dl.acm.org/doi/pdf/10.1145/3583780.3614928) | [Code](https://github.com/Skyorca/JHGDA) |
| **SGDA** | Invariant Representation Learning    | [ [ICJAI 2023] Semi-supervised Domain Adaptation in Graph Transfer Learning](https://www.ijcai.org/proceedings/2023/0253.pdf) | [Code](https://github.com/joe817/SGDA) |
| **MTDF** | Invariant Representation Learning | [[ICDE 2024] Multi-View Teacher with Curriculum Data Fusion for Robust Unsupervised Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/10597690/) | [N/A] |
| **SDA** | Invariant Representation Learning    | [ [AAAI 2024] Open-set graph domain  adaptation via separate domain alignment](https://ojs.aaai.org/index.php/AAAI/article/download/28765/29469) | [N/A] |
| **JDA-GCN** | Invariant Representation Learning    | [ [ICJAI 2024] Joint domain adaptive graph convolutional network](https://www.ijcai.org/proceedings/2024/0276.pdf) | [N/A] |
| **HC-GST** | Invariant Representation Learning    | [ [CIKM 2024] HC-GST: Heterophily-aware Distribution Consistency based Graph Self-training](https://dl.acm.org/doi/pdf/10.1145/3627673.3679622) | [N/A] |
| **DREAM** | Invariant Representation Learning    | [ [ICLR 2024] DREAM: Dual structured exploration with mixup for open-set graph domain adaption](https://openreview.net/pdf?id=4olqbTBt1Y) | [N/A] |
| **SelMAG** | Invariant Representation Learning    | [ [KDD 2024] Multi-source Unsupervised Domain Adaptation on Graphs with Transferability Modeling](https://arxiv.org/pdf/2406.10425) | [N/A] |
| **SRNC** | Concept-shift Aware Representation Learning    | [ [NeurIPS 2022] Shift-Robust Node Classification via Graph Clustering Co-training](https://www.cs.emory.edu/~jyang71/files/srnc.pdf) | [N/A] |
| **StruRW** | Concept-shift Aware Representation Learning    | [[ICML 2023] Structural re-weighting improves graph domain adaptation](https://arxiv.org/pdf/2306.03221.pdf) | [Code](https://github.com/Graph-COM/StruRW) |
| **Pair-Align** | Concept-shift Aware Representation Learning    | [[ICML 2024] Pairwise Alignment Improves Graph Domain Adaptation](https://arxiv.org/pdf/2403.01092) | [Code](https://github.com/Graph-COM/Pair-Align) |
| **GCONDA++** | Concept-shift Aware Representation Learning    |[[arXiv] Explaining and Adapting Graph Conditional Shift](https://arxiv.org/pdf/2306.03256.pdf) | [N/A] |
| **KDGA** | Model Regularization   |[[NeurIPS 2022] Knowledge distillation improves graph structure augmentation for graph neural networks](https://openreview.net/pdf?id=7yHte3tH8Xh) | [Code](https://github.com/LirongWu/KDGA)
| **SS/MFR-Reg** |Model Regularization    | [[ICLR 2023] Graph domain adaptation via theory-grounded spectral regularization](https://openreview.net/pdf?id=OysfLgrk8mk) | [Code](https://github.com/Shen-Lab/GDA-SpecReg) |
| **KTGNN** | Model Regularization    | [[TheWebConf 2023] Predicting the Silent Majority on Graphs: Knowledge Transferable Graph Neural Network](https://arxiv.org/pdf/2302.00873.pdf) | [Code](https://github.com/wendongbi/KT-GNN)
| **A2GNN** |Model Regularization    | [[AAAI 2024] Rethinking propagation for unsupervised graph domain adaptation](https://ojs.aaai.org/index.php/AAAI/article/download/29304/30460) | [Code](https://github.com/Meihan-Liu/24AAAI-A2GNN) |

### Data-centric Approaches

|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **IW** | Instance Weighting    | [[TheWebConf 2013] Predicting positive and negative links in signed social networks by transfer learning](https://dl.acm.org/doi/abs/10.1145/2488388.2488517) | [N/A] |
| **NES-TL** | Instance Weighting    |[[TNSE 2020] Nes-tl: Network embedding similarity-based transfer learning](http://www.xuanqi-net.com/Papers/TNSE19-NES.pdf)| [N/A] |
| **RSS-GNN** |Instance Weighting    | [[BIBM 2022] Reinforced Sample Selection for Graph Neural Networks Transfer Learning](https://ieeexplore.ieee.org/document/9995652) | [N/A] |
| **DR-GST** | Instance Weighting    |[[TheWebConf 2022] Confidence may cheat: Self-training on graph neural networks under distribution shift](https://arxiv.org/pdf/2201.11349.pdf) | [Code](https://github.com/bupt-gamma/dr-gst) |
| **FakeEdge** | Graph Data Augmentation  | [[LoG 2022] Fakeedge: Alleviate dataset shift in link prediction](https://arxiv.org/pdf/2211.15899.pdf) | [Code](https://github.com/Barcavin/FakeEdge)
| **Bridged-GNN** | Graph Data Augmentation  | [[CIKM 2023] Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer](https://arxiv.org/pdf/2308.09499v1.pdf) | [Code](https://github.com/wendongbi/Bridged-GNN)
| **DC-GST** | Graph Data Augmentation  | [[WSDM 2024] Distribution consistency based self-training for graph neural networks with sparse labels](https://arxiv.org/pdf/2401.10394.pdf) | [N/A] |
| **LTLP** | Graph Data Augmentation  | [[KDD 2024] Optimizing Long-tailed Link Prediction in Graph Neural Networks through Structure Representation Enhancement](https://arxiv.org/pdf/2407.20499) | [N/A] |

## ❖ Test-time Graph OOD Adaptation

### Model-centric Approaches

|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **GraphControl**  | Semi-supervised Fine-tuning | [[arXiv] GraphControl: Adding Conditional Control to Universal Graph Pre-trained Models for Graph Domain Transfer Learning](https://arxiv.org/pdf/2310.07365.pdf)    | [N/A] |
| **G-Adapter** | Semi-supervised Fine-tuning |  [[AAAI 2024] G-Adapter: Towards Structure-Aware Parameter-Effcient Transfer Learning for Graph Transformer Networks](https://ojs.aaai.org/index.php/AAAI/article/download/29112/30103) |   [N/A] |
| **AdapterGNN** | Semi-supervised Fine-tuning |  [[AAAI 2024] Adaptergnn: Parameter-efficient fine-tuning improves generalization in gnns](https://ojs.aaai.org/index.php/AAAI/article/download/29264/30385)      |  [Code](https://github.com/Lucius-lsr/AdapterGNN) |
| **PROGRAM** | Semi-supervised Fine-tuning |  [[ICLR 2024] PROGRAM: PROtotype GRAph Model based Pseudo-Label Learning for Test-Time Adaptation](https://openreview.net/pdf?id=x5LvBK43wg) |   [N/A] |
| **SOGA** | Self-supervised Adaptation |  [[WSDM 2024] Source free unsupervised graph domain adaptation](https://arxiv.org/pdf/2112.00955.pdf)      |  [Code](https://github.com/HaitaoMao/SOGA) |
| **GAPGC** | Self-supervised Adaptation |  [[ICML 2022] GraphTTA: Test Time Adaptation on Graph Neural Networks](https://arxiv.org/pdf/2208.09126.pdf)    | [N/A] |
| **GT3** | Self-supervised Adaptation |  [[arXiv] Test-time training for graph neural networks](https://arxiv.org/pdf/2210.08813.pdf)       |   [N/A] |
| **GraphGLOW** | Self-supervised Adaptation |  [[KDD 2023] GraphGLOW: Universal and Generalizable Structure Learning for Graph Neural Networks](https://arxiv.org/pdf/2306.11264.pdf)  | [Code](https://github.com/WtaoZhao/GraphGLOW)
| **RNA** | Self-supervised Fine-tuning |  [[IJCAI 2024] Rank and Align: Towards Effective Source-free Graph Domain Adaptation](https://www.ijcai.org/proceedings/2024/520) |   [N/A] |

### Data-centric Approaches

|Name|Category|Paper|Code|
| :------------ |:---------------:| :---------------| :---------------| 
| **FRGNN** | Feature Reconstruction | [[arXiv] FRGNN: Mitigating the Impact of Distribution Shift on Graph Neural Networks via Test-Time Feature Reconstruction](https://arxiv.org/pdf/2308.09259.pdf)   | [N/A] |
| **GTRANS** | Graph Data Augmentation  | [[ICLR 2023] Empowering graph representation learning with test-time graph transformation](https://openreview.net/pdf?id=Lnxl5pr018)    | [Code](https://github.com/ChandlerBang/GTrans)
| **GraphCTA** | Graph Data Augmentation | [[TheWebConf 2024] Collaborate to Adapt: Source-Free Graph Domain Adaptation via Bi-directional Adaptation](https://dl.acm.org/doi/pdf/10.1145/3589334.3645507) |  [Code](https://github.com/cszhangzhen/GraphCTA) |
| **SGOOD** | Graph Data Augmentation | [[arXiv] SGOOD: Substructure-enhanced Graph-Level Out-of-Distribution Detection](https://arxiv.org/pdf/2310.10237) | [N/A] |
| **GALA** | Graph Data Augmentation  | [[TPAMI 2024] GALA: Graph Diffusion-based Alignment with Jigsaw for Source-free Domain Adaptation](https://www.computer.org/csdl/journal/tp/5555/01/10561561/1XSjvvkWZhu)    | [Code](https://github.com/luo-junyu/GALA) |


## ❖ Related: Transferability evaluation 

|Name|Paper|Code|
| :------------ |:---------------| :---------------| 
| **EGI** |  [[NeurIPS 2021] Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization](https://proceedings.neurips.cc/paper/2021/file/0dd6049f5fa537d41753be6d37859430-Paper.pdf) | [Code](https://github.com/GentleZhu/EGI) |
| **WNN** | [[NeurIPS 2020] Graphon Neural Networks and the Transferability of Graph Neural Networks](https://arxiv.org/pdf/2006.03548.pdf) | [N/A] |
| **TMD** | [[NeurIPS 2022] Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks](https://arxiv.org/pdf/2210.01906.pdf) | [Code](https://github.com/chingyaoc/TMD) |
|**W2PGNN** | [[KDD 2023] When to Pre-Train Graph Neural Networks? From Data Generation Perspective!](https://arxiv.org/pdf/2303.16458.pdf)| [Code](https://github.com/caoyxuan/W2PGNN) |



## 📚 Citing This Work 
If you find this repository helpful to your work, please kindly star it and cite our survey paper as follows:
```bibtex
@article{zhang2024surveygraph,
      title={A Survey of Deep Graph Learning under Distribution Shifts: from Graph Out-of-Distribution Generalization to Adaptation}, 
      author={Kexin Zhang and Shuhan Liu and Song Wang and Weili Shi and Chen Chen and Pan Li and Sheng Li and Jundong Li and Kaize Ding},
      journal={arXiv preprint arXiv:2410.19265},
      year={2024}
}
```


