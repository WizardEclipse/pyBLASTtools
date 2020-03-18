from pyBLASTtools import timing as tm
import numpy as np
import matplotlib.pyplot as plt
import pygetdata as gd

t = tm.timing('/media/gabriele/mac')

master = t.ctime_master()
ctime_master = t.time_master

roach_number = [1, 2, 3, 4, 5]

kind = ['Packet', 'Clock']
### TEST DIFFERENT METHODS and COMPARE THEM ###

roach_comparison = {}

for i in range(len(kind)):

    print('Kind', kind[i])
    
    roach = t.ctime_roach(roach_number, kind[i])

    roach_comparison[kind[i]] = t.time_roach

chunk = 10000

number = 10 

for j in range(len(roach_number)):
    d = gd.dirfile(t.roach_path[roach_number[j]-1])
    
    ctime_roach_name = 'ctime_roach'+str(int(roach_number[j]))
    ctime_roach = (d.getdata(ctime_roach_name)).astype(np.float64)
    
    pps_roach_name = 'pps_count_roach'+str(int(roach_number[j]))
    pps_roach = (d.getdata(pps_roach_name)).astype(np.float64)
    
    length_chunk = np.floor(len(pps_roach)/number)
    
    roach_str = 'roach'+str(int(roach_number[j]))
    path = '/home/gabriele/Documents/pyBLASTtools/plots/'
    
    for k in range(number):
        min_val = j*length_chunk
        max_val = (j+1)*length_chunk-2*chunk
        
        idx_start = np.random.randint(min_val, max_val)
        idx_end = idx_start+chunk
        
        x_val = np.arange(idx_start,idx_end, 1)
        
        ctime = ctime_roach[idx_start:idx_end]*1e-2
        ctime += 1570000000.
 
        plt.plot(x_val, pps_roach[idx_start:idx_end]+ctime, label='PPS')
        plt.plot(x_val, roach_comparison['Packet'][roach_str][idx_start:idx_end], label='Packet')
        plt.plot(x_val, roach_comparison['Clock'][roach_str][idx_start:idx_end], label='Clock')
        
        plt.xlabel('Index')
        plt.ylabel('Ctime')
        
        name = path+roach_str+'_'+str(int(idx_start))+'_'+str(int(idx_end))+'.png'
        
        plt.savefig(name)
        plt.close()



