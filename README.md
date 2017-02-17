# The Incredible PyTorch

## What is this?
This is inspired by the famous [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow) repository where this repository would hold tutorials, projects, libraries, videos, papers, books and anything related to the incredible [PyTorch](http://pytorch.org/).


## Tutorials
- [Official PyTorch Tutorials](https://github.com/pytorch/tutorials)
	1. [Deep Learning with PyTorch: a 60-minute blitz](Deep Learning with PyTorch.ipynb)
		- A perfect introduction to PyTorch's torch, autograd, nn and optim APIs
   		- If you are a former Torch user, you can check out this instead: [Introduction to PyTorch for former Torchies](Introduction to PyTorch for former Torchies.ipynb)
	2. Custom C extensions
   		- [Write your own C code that interfaces into PyTorch via FFI](Creating%20Extensions%20using%20FFI.md)
	3. [Writing your own neural network module that uses numpy and scipy](Creating extensions using numpy and scipy.ipynb)
	4. [Reinforcement (Q-)Learning with PyTorch](Reinforcement (Q-)Learning with PyTorch.ipynb)
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
	- [Generating Names with a Character-Level RNN](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb)
	- [Translation with a Sequence to Sequence Network and Attention](https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb)
- [Simple Examples to Introduce PyTorch](https://github.com/colesbury/pytorch-examples)
- [Mini Tutorials in PyTorch](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials)
	- Tensor Multiplication, Linear Regresison, Logistic Regression, Neural Network, Modern Neural Network, and Convolutional Neural Network

## Papers in PyTorch
- [Learning to learn by gradient descent by gradient descent](https://github.com/ikostrikov/pytorch-meta-optimizer)
- [Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer](https://github.com/szagoruyko/attention-transfer)
- [Wasserstein GAN](https://github.com/martinarjovsky/WassersteinGAN)
- [Densely Connected Convolutional Networks](https://github.com/bamos/densenet.pytorch)
- [A Neural Algorithm of Artistic Style](https://github.com/alexis-jacq/Pytorch-Tutorials)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://github.com/jcjohnson/pytorch-vgg)
	- VGG model in PyTorch.
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://github.com/gsp-27/pytorch_Squeezenet)
- [Network In Network](https://github.com/szagoruyko/functional-zoo)
- [Deep Residual Learning for Image Recognition](https://github.com/szagoruyko/functional-zoo)
	- ResNet model in PyTorch.
- [Wide Residual Networks](https://github.com/szagoruyko/functional-zoo)
	- Wide ResNet model in PyTorch
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://github.com/longcw/faster_rcnn_pytorch)
- [FlowNet: Learning Optical Flow with Convolutional Networks](https://github.com/ClementPinard/FlowNetPytorch)
- [Asynchronous Methods for Deep Reinforcement Learning](https://github.com/ikostrikov/pytorch-a3c)


## Projects in Pytorch
- [Reinforcement learning models in ViZDoom environment with PyTorch](https://github.com/akolishchak/doom-net-pytorch)
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

## Community
- [PyTorch Discussion Forum](https://discuss.pytorch.org/)
	- This is actively maintained by [Adam Paszke](https://github.com/apaszke)
- [StackOverFlow PyTorch Tags](http://stackoverflow.com/questions/tagged/pytorch)

## Contributions
Do feel free to contribute! 

You can raise an isssue or submit a pull request, whichever is more convenient for you. The guideline is simple: just follow the format of the previous bullet point. 