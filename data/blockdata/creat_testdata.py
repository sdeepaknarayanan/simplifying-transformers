import numpy as np

l = []

for i in range(1000):

    x = np.linspace(i,(768/2*np.pi+i),768)
    sinx = np.sin(x)
    siny = np.sin(x+np.pi)
    l.append([sinx,siny])
    
l = np.asarray(l)


np.save('block_testdata.npy', l)
    



