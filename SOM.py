import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

som_dim = 10

class SOMNetwork():
	def __init__(self, input_dim, dim=10, sigma=10, learning_rate=0.01, dtype=tf.float32):
		self.dim = tf.constant(dim, dtype=tf.int64)
		self.input_dim = input_dim
		self.dtype = dtype
		self.x = tf.placeholder(shape=[input_dim], dtype=dtype, name='input')
		# self.__win_neuron = tf.Variable(tf.zeros([self.input_dim]), dtype=tf.float32, name='win_neuron')
		self.w = tf.Variable(tf.random_uniform([dim*dim, input_dim], -1, 1),
			dtype=dtype, name='weights')
		# self.w = tf.Variable([[1,1,1],[0,0,0],[3,3,3],[2,2,2]],
		# 	dtype=dtype, name='weights')
		self.positions = tf.where(tf.fill([dim, dim], True))
		self.win_index = tf.Variable(tf.zeros([2], dtype=tf.int64), name='win_index')
		self.sigma = tf.Variable(sigma, dtype=dtype, name='sigma')
		self.learning_rate = tf.Variable(learning_rate, dtype=dtype, name='learning_rate')

	def feed(self, input):
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			init.run()
			win_index = sess.run(self.__competition(), feed_dict={self.x: input})
			win_index_2d = np.array([win_index//self.dim.eval(), win_index-win_index//self.dim.eval()*self.dim.eval()])
		return win_index_2d


	def training_op(self):
		# x = np.array([1,2,3])
		win_index = self.__competition('train_')
		with tf.name_scope('cooperation') as scope:
			assign = tf.assign(self.win_index, [win_index//self.dim, win_index-win_index//self.dim*self.dim])
			coop_dist = tf.reduce_sum(tf.square(tf.cast(self.positions - [win_index//self.dim, win_index-win_index//self.dim*self.dim], dtype=self.dtype)), axis=1)
			tnh = tf.exp(-tf.square(coop_dist) / (2 * tf.square(self.sigma))) # topological neighbourhood
		with tf.name_scope('adaptation') as scope:
			delta = tf.transpose(self.learning_rate * tnh * tf.transpose(self.x - self.w))
			# delta1 = self.x - self.w
			# delta2 = tf.transpose(self.x - self.w)
			# delta3 = tnh * tf.transpose(self.x - self.w)
			# delta4 = tf.transpose(tnh * tf.transpose(self.x - self.w))
			training_op = tf.assign(self.w, self.w + delta)
		return training_op

		# distance = tf.square(self.x - self.w)
		# rsum = tf.reduce_sum(distance, axis=1)
		# argmin = self.__competition()
		# index = tf.Variable([0, 0], dtype=tf.int64, name='win_index')
		# # win_neuron = tf.Variable([0]*self.input_dim, dtype=tf.float32, name='win_neuron')
		# # print(win_neuron.get_shape())

		# positions = tf.where(tf.fill([self.__som_dim, self.__som_dim], True))

		# d1 = tf.square(tf.cast(positions - index, tf.float32))
		# rsum1 = tf.reduce_sum(d1, axis=1)
		# rsum1_2 = tf.square(rsum1)
		# sigma = tf.constant(3.0, dtype=tf.float32)
		# h = tf.exp(-rsum1_2 / (2 * sigma))

		# lr = tf.constant(0.01, dtype=tf.float32)
		# delta = tf.transpose(lr * h * tf.transpose(self.x - self.w))
		# training_op = tf.assign(self.w, self.w + delta)

		# init = tf.global_variables_initializer()
		# sess = tf.Session()
		# sess.run(init)
		# print(sess.run(self.w))
		# print(sess.run(win_index, feed_dict={self.x:x}))
		# print(sess.run(tf.assign(self.win_index, [win_index//self.dim, win_index-win_index//self.dim*self.dim]), feed_dict={self.x:x}))
		# print(sess.run(coop_dist))
		# print(sess.run(tnh))
		# print('delta')
		# print(sess.run(delta1, feed_dict={self.x:x}))
		# print(sess.run(delta2, feed_dict={self.x:x}))
		# print(sess.run(delta3, feed_dict={self.x:x}))
		# print(sess.run(delta4, feed_dict={self.x:x}))
		# print(sess.run(training_op, feed_dict={self.x:x}))
		# print(sess.run(positions))
		# print(sess.run(distance, feed_dict={self.x:x}))
		# print(sess.run(rsum, feed_dict={self.x:x}))
		# windex = sess.run(argmin, feed_dict={self.x:x})
		# print(windex)
		# print(sess.run(tf.assign(index, [windex//self.__som_dim, windex-windex//self.__som_dim*self.__som_dim])))
		# print(sess.run(index))
		# # print(sess.run(tf.assign(win_neuron, tf.gather(self.w, index))))
		# print('-------')
		# print(sess.run(d1))
		# print(sess.run(rsum1))
		# print(sess.run(rsum1_2))
		# print(sess.run(h))
		# print('delta')
		# print(sess.run(delta, feed_dict={self.x:x}))
		# print('=========')
		# print(sess.run(training_op, feed_dict={self.x:x}))
		# print('=-=-=-=')
		# print(sess.run(self.w))


	def __competition(self, info=''):
		with tf.name_scope(info+'competition') as scope:
			distance = tf.reduce_sum(tf.square(self.x - self.w), axis=1)
		return tf.argmin(distance, axis=0)


#== Test SOM Network ==
# som = SOMNetwork(input_dim=3, dim=2, sigma=2)
# som.training_op()

def test_som_with_color_data():
	som_dim = 100
	som = SOMNetwork(input_dim=3, dim=som_dim, sigma = 100)
	print(dir(som))
	test_data = np.random.uniform(0, 1, (10000, 3))
	training_op = som.training_op()
	init = tf.global_variables_initializer()
	print('start')
	with tf.Session() as sess:
		init.run()
		print(sess.run(som.w))
		img1 = tf.reshape(som.w,[som_dim,som_dim,-1]).eval()
		plt.figure(1)
		plt.subplot(121)
		# print(img1)
		plt.imshow(img1)
		# plt.show()
		for color_data in test_data:
			# plt.imshow([color_data])
			# plt.show()
			sess.run(training_op, feed_dict={som.x: color_data})
		print(sess.run(som.w))
		img = tf.reshape(som.w,[som_dim,som_dim,-1]).eval()
		plt.subplot(122)
		plt.imshow(img)
	print('complete')
	# print(som.feed([1,2,3]))
	plt.show()		

test_som_with_color_data()

# a = tf.constant(2, dtype=tf.int64)
# b = tf.constant(3, dtype=tf.int64)

# c = b//a
# sess = tf.Session()
# print(sess.run(c))