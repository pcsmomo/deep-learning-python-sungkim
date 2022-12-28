# Lab 3 Minimizing Cost
import tensorflow

tf = tensorflow.compat.v1
tf.disable_v2_behavior()
tf.disable_eager_execution()

tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data
# We know that W should be 1
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name="weight")

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer())

    for step in range(21):
        _, cost_val, W_val = sess.run(
            [update, cost, W], feed_dict={X: x_data, Y: y_data}
        )
        print(step, cost_val, W_val)

"""
0 0.79993474 [0.77918804]
1 0.22753702 [0.8822336]
2 0.06472163 [0.93719125]
3 0.018409718 [0.966502]
4 0.005236534 [0.9821344]
5 0.0014895027 [0.99047166]
6 0.000423682 [0.9949182]
7 0.00012051514 [0.9972897]
8 3.42793e-05 [0.9985545]
9 9.750314e-06 [0.9992291]
10 2.7733454e-06 [0.99958885]
11 7.8898245e-07 [0.9997807]
12 2.2437578e-07 [0.99988306]
13 6.384909e-08 [0.99993765]
14 1.8124851e-08 [0.99996674]
15 5.154282e-09 [0.99998224]
16 1.4765504e-09 [0.9999905]
17 4.1801348e-10 [0.99999493]
18 1.2039081e-10 [0.9999973]
19 3.3894075e-11 [0.99999857]
20 9.549694e-12 [0.9999992]
"""
