import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('nohup.out', skiprows=9, usecols=(1,3,5), delimiter=':')

temp = data[:,1]-data[0,1]
temp = temp*100/data[0,1]
slope = (temp[-1]-temp[0])/(data[-1,0]-data[0,0])
print(f"Average rate of change in divergence: {slope}")
plt.plot(data[:,0], temp)
# plt.plot(data[:,0], data[:,1]-data[0,1])
plt.ylabel('Relative % change in divergence')
plt.xlabel('Iternation')
# plt.ylim(1, 3)
plt.savefig('change_div.png')
plt.clf()
temp = data[:,2]-data[0,2]
temp = temp*100/data[0,2]
plt.plot(data[:,0], temp)
plt.xlabel('Iternation')
plt.ylabel('% change in total divergence')
plt.savefig('Total_divergence.png')
print(f"change in total divergence: {data[-1,2]-data[0,2]}")
print(f"% change in total div: {temp[-1]}")
slope = (temp[-1]-temp[0])/(data[-1,0]-data[0,0])
print(f"Average rate of change in total divergence: {slope}")