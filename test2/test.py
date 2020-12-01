import numpy as np

a = np.arange(400)
a = np.reshape(a,(10,40))
a_x, a_y = a.shape
x = np.linspace(0,a_x -1, a_x)
y = np.linspace(0,a_y -1, a_y)
yy, xx = np.meshgrid(x,y)
print(a)
c = np.arange(1200)
world = np.reshape(c,(a_x, a_y, 3))
world[:, :, 0] = np.swapaxes(yy, 0, 1)
world[:, :, 1] = np.swapaxes(xx, 0, 1)
world[:, :, 2] = a

print(world.shape)
print(world)
