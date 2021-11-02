# holbertonschool-machine_learning
This repository contains projects for the Machine Learning scpecialization at Holberton School. Holberton School's Machine Learning scpecialization teaches the fundamentals of an emerging and exciting field of study that has implications in almost every industry. The projects included in this repository are mathematical concepts needed for Machine Learning, Supervised Learning, Unsupervised Learning, and Reinforement Learning. Below you will see all four directories and the projects within, I have highlighted important projects inside each directory that you can click on to find out more.

## Python Libraries
- Tensorflow
- Numpy
- Pandas
- Keras
- Matplotlib

## [Math](math)
In this directory we work through many of the required mathematic skills and concepts needed for Machine Learning. A thorough mathematical understanding of many of these techniques and concepts is necessary for a good grasp of the inner workings of the algorithms and achieving desired results.
- [0x00-linear_algebra](math/0x00-linear_algebra)
- [0x01-plotting](math/0x01-plotting)
- [0x02-calculus](math/0x02-calculus)
- [0x03-probability](math/0x03-probability)
- [0x04-convolutions_and_pooling](math/0x04-convolutions_and_pooling)
- [0x05-advanced_linear_algebra](math/0x05-advanced_linear_algebra)
- [0x06-multivariate_prob](math/0x06-multivariate_prob)
- [0x07-bayesian_prob](math/0x07-bayesian_prob)

## [Supervised Learning](supervised_learning)
In this directory we work through Binary Classification, Multiclass Classification, Optimization Techniques, Regularization Techniques, Convolutional Neural Networks, Deep Convolutional Architectures and others for building Supervised Learning Models.
- [0x01-classification](supervised_learning/0x01-classification)
- [0x02-tensorflow](supervised_learning/0x02-tensorflow)
- [0x03-optimization](supervised_learning/0x03-optimization)
- [0x04-error_analysis](supervised_learning/0x04-error_analysis)
- [0x05-regularization](supervised_learning/0x05-regularization)
- [0x06-keras](supervised_learning/0x06-keras)
- [0x07-cnn](supervised_learning/0x07-cnn)
- [0x08-deep_cnns](supervised_learning/0x08-deep_cnns)
- [0x09-transfer_learning](supervised_learning/0x09-transfer_learning)
- [0x0A-object_detection](supervised_learning/0x0A-object_detection)
```
YOLO Model Summary:
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 416, 416, 3)  0                                            
__________________________________________________________________________________________________
conv2d (Conv2D)                 (None, 416, 416, 32) 864         input_1[0][0]                    
__________________________________________________________________________________________________
batch_normalization (BatchNorma (None, 416, 416, 32) 128         conv2d[0][0]                     
__________________________________________________________________________________________________

...

reshape (Reshape)               (None, 13, 13, 3, 85 0           conv2d_58[0][0]                  
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 26, 26, 3, 85 0           conv2d_66[0][0]                  
__________________________________________________________________________________________________
reshape_2 (Reshape)             (None, 52, 52, 3, 85 0           conv2d_74[0][0]                  
==================================================================================================
Total params: 62,001,757
Trainable params: 61,949,149
Non-trainable params: 52,608
__________________________________________________________________________________________________
Class names: ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
Class threshold: 0.6
NMS threshold: 0.5
Anchor boxes: [[[116  90]
  [156 198]
  [373 326]]

 [[ 30  61]
  [ 62  45]
  [ 59 119]]

 [[ 10  13]
  [ 16  30]
  [ 33  23]]]
```
- [0x0D-RNNs](supervised_learning/0x0D-RNNs)
- [0x0F-word_embeddings](supervised_learning/0x0F-word_embeddings)
- [0x10-nlp_metrics](supervised_learning/0x10-nlp_metrics)
- [0x0E-time_series](supervised_learning/0x0E-time_series)
- [0x11-attention](supervised_learning/0x11-attention)
- [0x12-transformer_apps](supervised_learning/0x12-transformer_apps)
- [0x13-qa_bot](supervised_learning/0x13-qa_bot)

## [Unsupervised Learning](unsupervised_learning)
In this directory we work through Dimensionality Reduction, Clustering, Embeddings, Autoencoders, Generative Adversarial Networks, Hyperparameter Optimization, Hidden Markov Models and others for building Unsupervised Learning Models.
- [0x00-dimensionality_reduction](unsupervised_learning/0x00-dimensionality_reduction)
- [0x01-clustering](unsupervised_learning/0x01-clustering)
- [0x02-hmm](unsupervised_learning/0x02-hmm)
- [0x03-hyperparameter_tuning](unsupervised_learning/0x03-hyperparameter_tuning)
- [0x04-autoencoders](unsupervised_learning/0x04-autoencoders)

## [Reinforcement Learning](reinforcement_learning)
In this directory we work through Agent-Environment Framework, Multi-armed Bandit problems, Markov decision process, Exploration vs Exploitation, Policy and Value Functions, Temporal Difference Learning, Deep Reinforcement Learning and others for building Reinforcement Learning Models.
- [0x00-q_learning](reinforcement_learning/0x00-q_learning)


## Authors

#### Gunter Pearson:
I am currently studying Machine Learning at Holberton School in Tulsa, Oklahoma and will graduate in January 2022. Machine learning has been able to revolutionize so many fields, ranging from medicine to social media, to food, to security. I am excited by the possibilities and my future in Machine Learning. You can keep up with my progress and find out more about me by clicking on the following links:
- [LinkedIn](www.linkedin.com/in/gunter-pearson-0611b81a1)
- [Git Hub](https://github.com/GunterPearson)
