# deep-learning-python-sungkim

Deep Learning in Python by Sung Kim

# Details

<details open>
  <summary>Click to Contract/Expend</summary>

## 1. Lec 00 - Machine/Deep learning intro

- Basic understanding of machine learning algorithms
  - Regression
    - Lenear regression, Logistic regression (classification)
  - Neutral networks
    - Convolutional Nueral Network
    - Recurrent Neural Networks

### references

- Andrew Ng's ML class
  - https://www.coursera.org/learn/machine-learning
  - https://holehouse.org/mlclass : note
- Good website
  - https://cs231n.github.io/
- TensorFlow
  - https://www.tensorflow.org/
  - https://github.com/aymericdamien/TensorFlow-Examples

### Schedule

- Machine learning basic concepts
- Linear regression
- Logistic regression (classification)
- Multivariable (Vector) linear/logistic regression
- Neural networks
- Deep learning
  - CNN
  - RNN
  - Bidirectional Neural networks

## 2. ML lec 01 - Machine Learning basic concepts

- Supervised Learning
  - learning with labeled examples
  - Most common problem type in ML
    - Image labeling: cat, dog, car, etc
    - Email spam filter
    - Predicting exam score
- Unsupervised Learning
  - Google news grouping
  - Word clustering

### Terms

- Training data set

### Types of supervised learning

- Predicting final exam score (0-100) based on time spent
  - regression
- Pass/none-pass based on time spent
  - binary classification
- Letter grade (A, B, C, D, and E) based on time spent
  - multi-label classification

## 3. ML lab 01: TensorFlow install

### install tensorflow

```sh
poetry self update
# poetry add tensorflow tensorflow-gpu
# tensorflow can't be installed on mac m1
poetry add tensorflow-macos tensorflow-metal  # for mac m1

poetry shell
python3
import tensorflow as tf
tf.__version__
'2.11.0'
```

### Check GPU availability

```py
gpu = len(tf.config.list_physical_devices('GPU'))>0
print("GPU is", "available" if gpu else "NOT AVAILABLE")
```

### [Tensor Ranks, Shapes, and Types](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/resources/dims_types.md#rank)

| Rank | Math entity                      | Python example                                               |
| ---- | -------------------------------- | ------------------------------------------------------------ |
| 0    | Scalar (magnitude only)          | s = 483                                                      |
| 1    | Vector (magnitude and direction) | v = [1.1, 2.2, 3.3]                                          |
| 2    | Matrix (table of numbers)        | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]                        |
| 3    | 3-Tensor (cube of numbers)       | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n    | n-Tensor (you get the idea)      | ....                                                         |

## 6. ML lec 03: Linear Regression - cost minimum algorithm

### Gradient descent algorithm

- Minimize cost function
- Gradient descent is used many jinimization problems

### Derivative calculator

link: https://www.derivative-calculator.net/

### Convex function

https://holehouse.org/mlclass/01_02_Introduction_regression_analysis_and_gr.html

## Formal definition

</details>
