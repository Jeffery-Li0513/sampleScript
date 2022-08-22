import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set(xlim=[-10,35], ylim=[0,100], title='pdos for Ru', xlabel='Energy(eV)', ylabel='pdos')

with open('0_IPDOS_Ru.dat', 'r', encoding='utf-8') as f:
    data = f.readlines()
    orbitrat_name = data[0].split()
    new_data = []
    for line in data[1:]:
        line = line.split()
        for i in range(len(line)):
            line[i] = float(line[i])
        new_data.append(line)

new_data = np.array(new_data)
x = new_data[:,0]
for i in range(1,len(orbitrat_name)):
    plt.plot(x, new_data[:,i], label=orbitrat_name[i])

plt.legend(loc="upper left")
plt.show()