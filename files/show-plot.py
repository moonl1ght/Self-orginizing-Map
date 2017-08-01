import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# def lr_func(x):
# 	return np.maximum(0.01, 0.1 * np.exp(-x/1000))

# x = np.linspace(0, 4000, 3000)
# y = lr_func(x)

# plt.yticks(np.arange(0., 0.11, 0.01))
# plt.xticks(np.arange(0, 11000, 500))
# plt.title('Learning Rate, $\eta(n)$')
# plt.xlabel('step, n')
# plt.ylabel('$\eta(n)$')
# plt.grid(True)
# plt.plot(x, y)
# plt.show()


# def sigma_func(x):
# 	tay = 1000/np.log(10)
# 	print(10 * np.exp(-x/tay))
# 	return np.maximum(1, 10 * np.exp(-x/tay))

# print(sigma_func(1000))

# x = np.linspace(0, 4000, 5000)
# y = sigma_func(x)

# # plt.yticks(np.arange(0., 20.5, 2))
# # plt.xticks(np.arange(0, 11000, 1000))
# props = dict(facecolor='black', shrink=0.1)
# plt.annotate('min $\sigma(n) = 1$', xytext=(2500, 4), xy=(1500, 1), arrowprops=props, fontsize=14, ha="center")
# plt.title('Effective width, $\sigma(n)$')
# plt.xlabel('step, n')
# plt.ylabel('$\sigma(n)$')
# plt.grid(True)
# plt.plot(x, y)
# plt.show()

# def tnh(x, sigma):
# 	return np.exp(-(x**2)/(2*sigma**2))

# x = np.linspace(-30, 30, 100)
# y = tnh(x, 10)
# y1 = tnh(x, 9)
# y2 = tnh(x, 5)
# y3 = tnh(x, 3)
# y4 = tnh(x, 2)
# y5 = tnh(x, 1)
# # y6 = tnh(x, 10)

# # plt.yticks(np.arange(0., 20.5, 2))
# # plt.xticks(np.arange(0, 11000, 1000))
# # props = dict(facecolor='black', shrink=0.1)
# # plt.annotate('min $\sigma(n) = 1$', xytext=(2500, 4), xy=(1500, 1), arrowprops=props, fontsize=14, ha="center")
# plt.title('Topological neighbourhood, $h(d, \sigma(n))$')
# plt.xlabel('lateral distance, d')
# plt.ylabel('$h(d, \sigma(n))$')
# plt.grid(True)
# g1 = plt.plot(x, y, label="$\sigma = 10$")
# # plt.legend(handles=[g1])
# plt.plot(x, y1, label="$\sigma = 9$")
# plt.plot(x, y2, label="$\sigma = 5$")
# plt.plot(x, y3, label="$\sigma = 3$")
# plt.plot(x, y5, label="$\sigma = 1$")
# plt.legend(loc="upper left", fontsize=16)
# # plt.plot(x, y, label="$\sigma = 10$", x, y1, label="$\sigma = 9$",
# # 		x, y2, label="$\sigma = 5$", x, y3, label="$\sigma = 3$", x, y5, label="$\sigma = 1$")
# plt.show()

def tnh3d(x, y, sigma):
	return np.exp(-(((x**2 + y**2))**2)/(2*sigma**2))

# x = np.linspace(-30, 30, 100)
# y = np.linspace(-30, 30, 100)
x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
print(X)

z = tnh3d(X, Y, 10)

# R = np.sqrt(X**2 + Y**2)
# z = np.sin(R)

figure = plt.figure(1, figsize = (12, 4))
subplot3d = plt.subplot(111, projection='3d')
surface = subplot3d.plot_surface(X, Y, z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0.1)
plt.show()

# x = np.linspace(-5, 5, 50)
# y = np.linspace(-5, 5, 50)
# X, Y = np.meshgrid(x, y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)

# figure = plt.figure(1, figsize = (12, 4))
# subplot3d = plt.subplot(111, projection='3d')
# surface = subplot3d.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=matplotlib.cm.coolwarm, linewidth=0.1)
# plt.show()