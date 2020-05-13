from pyBLASTtools import timing as tm

t = tm.timing('/media/gabriele/mac')

# master = t.ctime_master(write=True)
# ctime_master = t.time_master

roach_number = [1]#, 2, 3, 4, 5]

kind = ['Packet', 'Clock']

roach_comparison = {}

roach = t.ctime_roach(roach_number, kind, mode='average', write=True)

    





