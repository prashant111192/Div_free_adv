import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('nohup_V1_2.out', skiprows=10, usecols=(1,3), delimiter=':')

temp = data[:,1]-data[0,1]
temp = temp/data[0,1]
plt.plot(data[:,0], temp)
# plt.plot(data[:,0], data[:,1]-data[0,1])
plt.ylabel('Relative change in divergence')
plt.xlabel('Iternation')
# plt.ylim(1, 3)
plt.savefig('div_ker_temp.png')