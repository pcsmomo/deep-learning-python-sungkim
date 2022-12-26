import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hellow, TensorFlow!")

tf.print(hello)
