import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set(xlim=[-10,35], ylim=[0,5], title='total dos for Ru', ylabel='tdos', xlabel='Energy')
# xlim\ylim设置两个轴的取值范围

# data = np.fromfile('tdos.dat', dtype=np.float32, sep=' ')
# print(np.shape(data))

with open('tdos.dat','r',encoding='utf-8') as f:
    data = f.readlines()
    energy = []
    tdos = []
    for line in data[1:]:
        line = line.split()
        energy.append(float(line[0]))
        tdos.append(float(line[1]))

x = np.array(energy)
y = np.array(tdos)

plt.plot(x,y)

plt.show()