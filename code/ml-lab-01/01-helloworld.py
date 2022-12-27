import tensorflow as tf

# print("TensorFlow version:", tf.__version__)

# # Create a constant op
# # This op is added as a node to the default graph
# hello = tf.constant("Hellow, TensorFlow!")

# tf.print(hello)

# Using session to follow the lecture
# tf.compat.v1.disable_eager_execution()
# tf.executing_eagerly()

# https://github.com/OlafenwaMoses/ImageAI/issues/400
# start a TF session
with tf.compat.v1.Session() as sess:
    hello = tf.constant("Hello, TensorFlow!")
    # run the op and get result
    print(sess.run(hello))
