<p align="center"><img width="40%" src="the_incredible_pytorch.png" /></p>

--------------------------------------------------------------------------------
<p align="center">
	<img src="https://img.shields.io/badge/stars-1000+-blue.svg"/>
	<img src="https://img.shields.io/badge/forks-300+-blue.svg"/>
	<img src="https://img.shields.io/badge/license-MIT-blue.svg"/>
</p>

This is a curated list of tutorials, projects, libraries, videos, papers, books and anything related to the incredible [PyTorch](http://pytorch.org/).

## Tutorials
- [Official PyTorch Tutorials](https://github.com/pytorch/tutorials)
	1. [Deep Learning with PyTorch: a 60-minute blitz](http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
		- A perfect introduction to PyTorch's torch, autograd, nn and optim APIs
   		- If you are a former Torch user, you can check out this instead: [Introduction to PyTorch for former Torchies](http://pytorch.org/tutorials/beginner/former_torchies_tutorial.html)
	2. Custom C extensions
   		- [Write your own C code that interfaces into PyTorch via FFI](http://pytorch.org/tutorials/advanced/c_extension.html)
	3. [Writing your own neural network module that uses numpy and scipy](http://pytorch.org/tutorials/advanced/numpy_extensions_tutorial.html)
	4. [Reinforcement (Q-)Learning with PyTorch](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Official PyTorch Examples](https://github.com/pytorch/examples)
	- MNIST Convnets
	- Word level Language Modeling using LSTM RNNs
	- Training Imagenet Classifiers with Residual Networks
	- Generative Adversarial Networks (DCGAN)
	- Variational Auto-Encoders
	- Superresolution using an efficient sub-pixel convolutional neural network
	- Hogwild training of shared ConvNets across multiple processes on MNIST
	- Training a CartPole to balance in OpenAI Gym with actor-critic
	- Natural Language Inference (SNLI) with GloVe vectors, LSTMs, and torchtext
- [Practical PyTorch](https://github.com/spro/practical-pytorch)
	- This focuses on using RNNs for NLP
	- [Classifying Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb)
	- [Generating Shakespeare with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)
	- [Generating Names with a Conditional Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb)
	- [Translation with a Sequence to Sequence Network and Attention](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [Simple Examples to Introduce PyTorch](https://github.com/jcjohnson/pytorch-examples)
- [Mini Tutorials in PyTorch](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials)
	- Tensor Multiplication, Linear Regresison, Logistic Regression, Neural Network, Modern Neural Network, and Convolutional Neural Network
- [Deep Learning for NLP](https://github.com/rguthrie3/DeepLearningForNLPInPytorch)
	1. Introduction to Torch's Tensor Library
	2. Computation Graphs and Automatic Differentiation
	3. Deep Learning Building Blocks: Affine maps, non-linearities, and objectives
	4. Optimization and Training
	5. Creating Network Components in Pytorch
	  * Example: Logistic Regression Bag-of-Words text classifier
	6. Word Embeddings: Encoding Lexical Semantics
	  * Example: N-Gram Language Modeling
	  * Exercise: Continuous Bag-of-Words for learning word embeddings
	7. Sequence modeling and Long-Short Term Memory Networks
	  * Example: An LSTM for Part-of-Speech Tagging
	  * Exercise: Augmenting the LSTM tagger with character-level features
	8. Advanced: Making Dynamic Decisions
	  * Example: Bi-LSTM Conditional Random Field for named-entity recognition
	  * Exercise: A new loss function
- [Deep Learning Tutorial for Researchers](https://github.com/yunjey/pytorch-tutorial)
	1. PyTorch Basics
	2. Linear Regression
	3. Logistic Regression
	4. Feedforward Neural Network
	5. Convolutional Neural Network
	6. Deep Residual Network
	7. Recurrent Neural Network
	8. Bidirectional Recurrent Neural Network
	9. Language Model (RNNLM)
	10. Image Captioning (CNN-RNN)
	11. Generative Adversarial Network
	12. Deep Q-Network and Q-learning (WIP)
- [Fully Convolutional Networks implemented with PyTorch](https://github.com/wkentaro/pytorch-fcn)
- [Simple PyTorch Tutorials Zero to ALL](https://github.com/hunkim/PyTorchZeroToAll)
- [DeepNLP-models-Pytorch](https://github.com/DSKSD/DeepNLP-models-Pytorch)
	1. Skip-gram-Naive-Softmax
	2. Skip-gram-Negative-Sampling
	3. GloVe
	4. Window-Classifier-for-NER
	5. Neural-Dependancy-Parser
	6. RNN-Language-Model
	7. Neural-Machine-Translation-with-Attention
	8. CNN-for-Text-Classification
	9. Recursive-NN-for-Sentiment-Classification
	10. Dynamic-Memory-Network-for-Question-Answering
- [MILA PyTorch Welcome Tutorials](https://github.com/mila-udem/welcome_tutorials)

## Papers Originally Implemented with PyTorch
- [https://github.com/DavexPro/pytorch-pose-estimation](https://github.com/DavexPro/pytorch-pose-estimation)
- [Faster Faster R-CNN Implementation](https://github.com/jwyang/faster-rcnn.pytorch)
- [In-Place Activated BatchNorm for Memory-Optimized Training of DNNs](https://github.com/mapillary/inplace_abn)
- [Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
- [OptNet: Differentiable Optimization as a Layer in Neural Networks](https://github.com/locuslab/optnet)
- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://github.com/szagoruyko/attention-transfer)
- [Wide ResNet model in PyTorch](https://github.com/szagoruyko/functional-zoo)
- [Task-based End-to-end Model Learning](https://github.com/locuslab/e2e-model-learning)
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](https://github.com/bgshih/crnn)
- [Scaling the Scattering Transform: Deep Hybrid Networks](https://github.com/edouardoyallon/pyscatwave)
- [Adversarial Generator-Encoder Network](https://github.com/DmitryUlyanov/AGE)
- [Conditional Similarity Networks](https://github.com/andreasveit/conditional-similarity-networks)
- [Multi-style Generative Network for Real-time Transfer](https://github.com/zhanghang1989/PyTorch-Style-Transfer)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
- [Inferring and Executing Programs for Visual Reasoning](https://github.com/facebookresearch/clevr-iep)
- [On the Effects of Batch and Weight Normalization in Generative Adversarial Networks](https://github.com/stormraiser/GAN-weight-norm)
- [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://github.com/eladhoffer/bigBatch)
- [Neural Message Passing for Quantum Chemistry](https://github.com/priba/nmp_qc)
- [DiracNets: Training Very Deep Neural Networks Without Skip-Connections](https://github.com/szagoruyko/diracnets)
- [Deal or No Deal? End-to-End Learning for Negotiation Dialogues](https://github.com/facebookresearch/end-to-end-negotiator)
- [Visual Question Answering in Pytorch](https://github.com/Cadene/vqa.pytorch)
- [Principled Detection of Out-of-Distribution Examples in Neural Networks](https://github.com/ShiyuLiang/odin-pytorch)
- [Attention is all you need](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [FreezeOut: Accelerate Training by Progressively Freezing Layers](https://github.com/ajbrock/FreezeOut)
- [CortexNet: a Generic Network Family for Robust Visual Temporal Representations](https://github.com/e-lab/pytorch-CortexNet)
- [VSE++: Improved Visual-Semantic Embeddings](https://github.com/fartashf/vsepp)
- [Reading Wikipedia to Answer Open-Domain Questions](https://github.com/facebookresearch/DrQA)
- [A Structured Self-Attentive Sentence Embedding](https://github.com/ExplorerFreda/Structured-Self-Attentive-Sentence-Embedding)
- [Efficient Densenet](https://github.com/gpleiss/efficient_densenet_pytorch)
- [Averaged Stochastic Gradient Descent with Weight Dropped LSTM](https://github.com/salesforce/awd-lstm-lm)
- [Oriented Response Networks](https://github.com/ZhouYanzhao/ORN)
- [Video Frame Interpolation via Adaptive Separable Convolution](https://github.com/sniklaus/pytorch-sepconv)
- [Learning local feature descriptors with triplets and shallow convolutional neural networks](https://github.com/edgarriba/examples/tree/master/triplet)
- [Training RNNs as Fast as CNNs](https://github.com/taolei87/sru)
- [How far are we from solving the 2D & 3D Face Alignment problem? (and a dataset of 230,000 3D facial landmarks)](https://github.com/1adrianb/face-alignment)
- [Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition](https://github.com/kenshohara/3D-ResNets-PyTorch)
- [Reinforcement Learning for Bandit Neural Machine Translation with Simulated Human Feedback](https://github.com/khanhptnk/bandit-nmt)
- [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://github.com/clcarwin/sphereface_pytorch)
- [Quasi-Recurrent Neural Network (QRNN)](https://github.com/salesforce/pytorch-qrnn)
- [Enhanced Deep Residual Networks for Single Image Super-Resolution](https://github.com/thstkdgus35/EDSR-PyTorch)
- [DiracNets](https://github.com/szagoruyko/diracnets)

## Papers with Third-Party PyTorch Implementations
- [Binary Stochastic Neurons](https://github.com/Wizaron/binary-stochastic-neurons)
- [ReSeg: A Recurrent Neural Network-based Model for Semantic Segmentation](https://github.com/Wizaron/reseg-pytorch)
- [mixup: Beyond Empirical Risk Minimization](https://github.com/leehomyc/mixup_pytorch)
- [Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://github.com/xingyizhou/pytorch-pose-hg-3d)
- [Hierarchical Attention Network for Document Classification](https://github.com/cedias/HAN-pytorch)
- [Unsupervised Learning of Depth and Ego-Motion from Video](https://github.com/ClementPinard/SfmLearner-Pytorch)
- [Learning to learn by gradient descent by gradient descent](https://github.com/ikostrikov/pytorch-meta-optimizer)
- [Densely Connected Convolutional Networks](https://github.com/bamos/densenet.pytorch)
- [A Neural Algorithm of Artistic Style](https://github.com/alexis-jacq/Pytorch-Tutorials)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://github.com/jcjohnson/pytorch-vgg)
	- VGG model in PyTorch.
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and \<0.5MB model size](https://github.com/gsp-27/pytorch_Squeezenet)
- [Network In Network](https://github.com/szagoruyko/functional-zoo)
- [Deep Residual Learning for Image Recognition](https://github.com/szagoruyko/functional-zoo)
	- ResNet model in PyTorch.
- [Training Wide ResNets for CIFAR-10 and CIFAR-100 in PyTorch](https://github.com/xternalz/WideResNet-pytorch)
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://github.com/longcw/faster_rcnn_pytorch)
- [FlowNet: Learning Optical Flow with Convolutional Networks](https://github.com/ClementPinard/FlowNetPytorch)
- [Asynchronous Methods for Deep Reinforcement Learning](https://github.com/ikostrikov/pytorch-a3c)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://github.com/bengxy/FastNeuralStyle)
- [Highway Networks](https://github.com/c0nn3r/pytorch_highway_networks)
- [Collection of Generative Models with PyTorch](https://github.com/wiseodd/generative-models)
	- Generative Adversarial Nets (GAN)
		1. [Vanilla GAN](https://arxiv.org/abs/1406.2661)
		2. [Conditional GAN](https://arxiv.org/abs/1411.1784)
		3. [InfoGAN](https://arxiv.org/abs/1606.03657)
		4. [Wasserstein GAN](https://arxiv.org/abs/1701.07875)
		5. [Mode Regularized GAN](https://arxiv.org/abs/1612.02136)
	- Variational Autoencoder (VAE)
		1. [Vanilla VAE](https://arxiv.org/abs/1312.6114)
		2. [Conditional VAE](https://arxiv.org/abs/1406.5298)
		3. [Denoising VAE](https://arxiv.org/abs/1511.06406)
		4. [Adversarial Autoencoder](https://arxiv.org/abs/1511.05644)
		5. [Adversarial Variational Bayes](https://arxiv.org/abs/1701.04722)
- [A Recurrent Latent Variable Model for Sequential Data (VRNN)](https://github.com/emited/VariationalRecurrentNeuralNetwork)
- [Hybrid computing using a neural network with dynamic external memory](https://github.com/ypxie/pytorch-NeuCom)
- [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://github.com/SeanNaren/deepspeech.pytorch)
- [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation](https://github.com/mattmacy/vnet.pytorch)
- [Value Iteration Networks](https://github.com/onlytailei/PyTorch-value-iteration-networks)
- [YOLOv2: Real-Time Object Detection](https://github.com/longcw/yolo2-pytorch)
- [Convolutional Neural Fabrics](https://github.com/vabh/convolutional-neural-fabrics)
- [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://github.com/dasguptar/treelstm.pytorch)
- [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://github.com/fxia22/pointnet.pytorch)
- [Deformable Convolutional Network](https://github.com/oeway/pytorch-deform-conv)
- [Continuous Deep Q-Learning with Model-based Acceleration](https://github.com/ikostrikov/pytorch-naf)
- [Hierarchical Attention Networks for Document Classification](https://github.com/EdGENetworks/attention-networks-for-classification)
- [Spatial Transformer Networks](https://github.com/fxia22/stn.pytorch)
- [Decoupled Neural Interfaces using Synthetic Gradients](https://github.com/andrewliao11/dni.pytorch)
- [Improved Training of Wasserstein GANs](https://github.com/caogang/wgan-gp)
- [CycleGAN and Semi-Supervised GAN](https://github.com/yunjey/mnist-svhn-transfer)
- [Automatic chemical design using a data-driven continuous representation of molecules](https://github.com/cxhernandez/molencoder)
- [Differentiable Neural Computer](https://github.com/jingweiz/pytorch-dnc)
- [Asynchronous Methods for Deep Reinforcement Learning for Atari 2600](https://github.com/dgriff777/rl_a3c_pytorch)
- [Trust Region Policy Optimization](https://github.com/mjacar/pytorch-trpo)
- [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://github.com/jacobgil/pytorch-grad-cam)
- [Improving Variational Auto-Encoders using Householder Flow and using convex combination linear Inverse Autoregressive Flow](https://github.com/jmtomczak/vae_vpflows)
- [SSD: Single Shot MultiBox Detector](https://github.com/amdegroot/ssd.pytorch)
- [Neural Combinatorial Optimization with Reinforcement Learning](https://github.com/pemami4911/neural-combinatorial-rl-pytorch)
- [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://github.com/jacobgil/pytorch-pruning)
- [A Neural Representation of Sketch Drawings](https://github.com/alexis-jacq/Pytorch-Sketch-RNN)
- [Convolutional LSTM Network](https://github.com/automan000/Convolution_LSTM_pytorch)
- [Noisy Networks for Exploration](https://github.com/Kaixhin/NoisyNet-A3C)
- [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://github.com/potterhsu/SVHNClassifier-PyTorch)
- [Distributed Proximal Policy Optimization](https://github.com/alexis-jacq/Pytorch-DPPO)
- [Single Shot MultiBox Detector](https://github.com/amdegroot/ssd.pytorch)
- [Deformable Convolutional Networks in PyTorch](https://github.com/1zb/deformable-convolution-pytorch)
- [Dilated ResNet combination with Dilated Convolutions](https://github.com/fyu/drn)
- [Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling](https://github.com/DSKSD/RNN-for-Joint-NLU)
- [Attentive Recurrent Comparators](https://github.com/sanyam5/arc-pytorch)
- [PyTorch GAN Collection](https://github.com/znxlwm/pytorch-generative-model-collections)
- [Compact Bilinear Pooling](https://github.com/DeepInsight-PCALab/CompactBilinearPooling-Pytorch)
- [Striving for Simplicity: The All Convolutional Net](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
- [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
- [Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images](https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)
 - [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization - 2](https://github.com/utkuozbulak/pytorch-cnn-visualizations)
 - [FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://github.com/NVIDIA/flownet2-pytorch)
 - [Optical Flow Estimation using a Spatial Pyramid Network](https://github.com/sniklaus/pytorch-spynet)
 - [Skip-Thought Vectors](https://github.com/sanyam5/skip-thoughts)
 - [Explaining and Harnessing Adversarial Examples](https://github.com/utkuozbulak/pytorch-cnn-adversarial-attacks)

## Projects Implemented with Pytorch
- [Collection of Sequence to Sequence Models with PyTorch](https://github.com/MaximumEntropy/Seq2Seq-PyTorch)
	1. Vanilla Sequence to Sequence models
	2. Attention based Sequence to Sequence models
	3. Faster attention mechanisms using dot products between the final encoder and decoder hidden states
- [Reinforcement learning models in ViZDoom environment with PyTorch](https://github.com/akolishchak/doom-net-pytorch)
- [Neuraltalk 2, Image Captioning Model, in PyTorch](https://github.com/ruotianluo/neuraltalk2.pytorch)
- [Recurrent Neural Networks for Sentiment Analysis (Aspect-Based) on SemEval 2014](https://github.com/vanzytay/pytorch_sentiment_rnn)
- [PyTorch Image Classification with Kaggle Dogs vs Cats Dataset](https://github.com/rdcolema/pytorch-image-classification)
- [CNN Based Text Classification](https://github.com/xiayandi/Pytorch_text_classification)
- [Open-source (MIT) Neural Machine Translation (NMT) System](https://github.com/OpenNMT/OpenNMT-py)
- [Pytorch Poetry Generation](https://github.com/jhave/pytorch-poetry-generation)
- [Data Augmentation and Sampling for Pytorch](https://github.com/ncullen93/torchsample)
- [CIFAR-10 on Pytorch with VGG, ResNet and DenseNet](https://github.com/kuangliu/pytorch-cifar)
- [Generate captions from an image with PyTorch](https://github.com/eladhoffer/captionGen)
- [Generative Adversarial Networks, focusing on anime face drawing](https://github.com/jayleicn/animeGAN)
- [Simple Generative Adversarial Networks](https://github.com/mailmahee/pytorch-generative-adversarial-networks)
- [Fast Neural Style Transfer](https://github.com/darkstar112358/fast-neural-style)
- [Pixel-wise Segmentation on VOC2012 Dataset using PyTorch](https://github.com/bodokaiser/piwise)
- [Draw like Bob Ross](https://github.com/kendricktan/drawlikebobross)
- [Reinforcement learning models using Gym and Pytorch](https://github.com/jingweiz/pytorch-rl)
- [Open-Source Neural Machine Translation in PyTorch](https://github.com/OpenNMT/OpenNMT-py)
- [Deep Video Analytics](https://github.com/AKSHAYUBHAT/DeepVideoAnalytics)
- [Adversarial Auto-encoders](https://github.com/fducau/AAE_pytorch)
- [Whale Detector](https://github.com/TarinZ/whale-detector)
- [Base pretrained models and datasets in pytorch (MNIST, SVHN, CIFAR10, CIFAR100, STL10, AlexNet, VGG16, VGG19, ResNet, Inception, SqueezeNet)](https://github.com/aaron-xichen/pytorch-playground)
- [Open Source Chatbot with PyTorch](https://github.com/jinfagang/pytorch_chatbot)
- [Seq2Seq Intent Parsing](https://github.com/spro/pytorch-seq2seq-intent-parsing)
- [OpenFace in PyTorch](https://github.com/thnkim/OpenFacePytorch)
- [Complete Suite for Training Seq2Seq Models in PyTorch](https://github.com/eladhoffer/seq2seq.pytorch)
- [Probabilistic Programming and Statistical Inference in PyTorch](https://github.com/stepelu/ptstat)
- [Graphics code generating model using Processing](https://github.com/jtoy/sketchnet)
- [MUSE: Multilingual Unsupervised and Supervised Embeddings](https://github.com/facebookresearch/MUSE)

## PyTorch Video Tutorials
- [Practical Deep Learning with PyTorch](https://www.udemy.com/practical-deep-learning-with-pytorch/?couponCode=DEEPWIZARD)
- [PyTorch Zero to All Lectures (Free)](http://bit.ly/PyTorchVideo)
## Useful PyTorch Tools
- [Load Audio files directly into PyTorch Tensors](https://github.com/pytorch/audio)
- [Weight Initializations](https://github.com/pytorch/pytorch/blob/master/torch/nn/init.py)
- [Spatial transformer implemented in PyTorch](https://github.com/fxia22/stn.pytorch)
- [PyTorch AWS AMI, run PyTorch with GPU support in less than 5 minutes](https://github.com/ritchieng/dlami)
- [Use tensorboard with PyTorch](https://github.com/lanpa/tensorboard-pytorch)
- [Simple Fit Module in PyTorch, similar to Keras](https://github.com/henryre/pytorch-fitmodule)

## Community
- [PyTorch Discussion Forum](https://discuss.pytorch.org/)
	- This is actively maintained by [Adam Paszke](https://github.com/apaszke)
- [StackOverflow PyTorch Tags](http://stackoverflow.com/questions/tagged/pytorch)
- [Gloqo](https://www.gloqo.com/)
	- Add, discover and discuss paper implementations in PyTorch and other frameworks.


## Contributions
Do feel free to contribute!

You can raise an issue or submit a pull request, whichever is more convenient for you. The guideline is simple: just follow the format of the previous bullet point.
