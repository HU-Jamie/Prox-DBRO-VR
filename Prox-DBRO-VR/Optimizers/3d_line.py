import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

mpl.rcParams['legend.fontsize'] = 10
font = FontProperties()
font.set_size(12)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(0, 2, 100)
r = z ** 2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax.plot(x, y, z, label='parametric curve')
plt.xlabel('time', fontproperties=font)
plt.ylabel('Area (A)', fontproperties=font)
ax.set_zlabel('velocity (v)', fontproperties=font)
ax.legend()
plt.show()