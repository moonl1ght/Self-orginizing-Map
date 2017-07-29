import numpy as np
import tensorflow as tf

som_dim = 10

class SOMNetwork():
	def __init__(self, input_dim, som_dim=10):
		self.som_dim = som_dim
		self.input_dim = input_dim
		self.x = tf.placeholder(shape=[input_dim], dtype=tf.float32, name='input')
		self.w = tf.Variable(tf.random_uniform([som_dim*som_dim, input_dim], -1, 1),
			name='weights')

	def __competition__(self):
		with tf.name_scope('competition') as scope:
			distance = tf.square(self.x - self.w)
			argmin = tf.argmin(tf.reduce_sum(distance, axis=1))
		return argmin


	def __cooperation__(self):
		pass

	def __adaptation__(self):
		pass

	def train(self, train_data=None):
		x = np.array([1,2,3])
		distance = tf.square(self.x - self.w)
		rsum = tf.reduce_sum(distance, axis=1)
		argmin = self.__competition__()
		index = 0
		win_neuron = tf.Variable([0]*self.input_dim, dtype=tf.float32, name='win_neuron')
		print(win_neuron.get_shape())

		d1 = tf.square(self.w - win_neuron)
		rsum1 = tf.reduce_sum(d1, axis=1)
		rsum1_2 = tf.square(rsum1)
		sigma = tf.constant(3.0, dtype=tf.float32)
		h = tf.exp(-rsum1_2 / (2 * sigma))

		lr = tf.constant(0.01, dtype=tf.float32)
		delta = tf.transpose(lr * h * tf.transpose(self.x - self.w))
		training_op = tf.assign(self.w, self.w + delta)

		init = tf.global_variables_initializer()
		sess = tf.Session()
		sess.run(init)
		print(sess.run(self.w))
		print(sess.run(distance, feed_dict={self.x:x}))
		print(sess.run(rsum, feed_dict={self.x:x}))
		index = sess.run(argmin, feed_dict={self.x:x})
		print(index)
		print(sess.run(tf.assign(win_neuron, tf.gather(self.w, index))))
		print('-------')
		print(sess.run(d1))
		print(sess.run(rsum1))
		print(sess.run(rsum1_2))
		print(sess.run(h))
		print('delta')
		print(sess.run(delta, feed_dict={self.x:x}))
		print('=========')
		print(sess.run(training_op, feed_dict={self.x:x}))
		print('=-=-=-=')
		print(sess.run(self.w))




		

som = SOMNetwork(input_dim=3, som_dim=2)
som.train()


a = tf.Variable([[1,1,1],[2,2,2],[3,3,3],[4,4,4]])
b = tf.Variable([1,2,3,4])
# c = a * b
init = tf.global_variables_initializer()
s = tf.Session()
t = tf.transpose(a)
c = t * b
tb = tf.transpose(c)
s.run(init)
print(s.run(a))
print(s.run(b))
print(s.run(t))
print(s.run(tb))
print(s.run(c))