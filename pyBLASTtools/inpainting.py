import numpy as np
import itertools
from numba import jit
import time 

class NearestNeighbours:
    """
    Nearest-neighbours inpainting for filling a gap in a map.
    """


    def __init__ (self, maps, tol=1e-8):

        self.maps = maps.copy()
        self.idx_cell = np.where(self.maps==0)
        self.Y = np.shape(self.maps)[0]
        self.X = np.shape(self.maps)[1]

        self.tol=tol

    def find_cell_neighbours(self, x, y):

        idx_temp = list(itertools.product(range(y-1, y+2), range(x-1, x+2)))

        idx = []

        for i in idx_temp:
            if (-1 < i[1] <= self.X-1 and -1 < i[0] <= self.Y-1 and (i[1] != x or i[0] != y)):
                idx.append(i)

        return idx

    def find_all_neighbours(self):

        idx_neighbours_x = []
        idx_neighbours_y = []

        for i in range(len(self.idx_cell[0])):
            temp = np.array(self.find_cell_neighbours(self.idx_cell[1][i],self.idx_cell[0][i]))

            idx_neighbours_x.append(temp[:,1])
            idx_neighbours_y.append(temp[:,0])

        return idx_neighbours_x, idx_neighbours_y

    def mean_ignore(self, array):

        zero_count = np.count_nonzero(array)

        if zero_count != 0:
            return np.sum(array)/zero_count
        else:
            return 0

    @staticmethod
    @jit(nopython=True)  
    def mean_loop(mp, idx_cell, idx_x, idx_y):

        val = mp.copy()

        for i in range(len(idx_x)):

            ix = idx_x[i]
            iy = idx_y[i]

            array = mp[iy, ix]
            zero_count = np.count_nonzero(array)

            ixc = idx_cell[1][i]
            iyc = idx_cell[0][i]

            if zero_count != 0:
                val[iyc, ixc] = np.sum(array)/zero_count
            else:
                val[iyc, ixc] = 0

        return val

    def predict_numba(self, niter=1000):
        idx_neighbours_x, idx_neighbours_y = self.find_all_neighbours()

        final_map = self.maps.copy()

        count=0

        while True: 
            temp_map = final_map.copy()
            t1 = time.time()
            final_map = self.mean_loop(temp_map, self.idx_cell, idx_neighbours_x, idx_neighbours_y)
            print('TIME', time.time()-t1)
            if np.allclose(final_map[self.idx_cell], temp_map[self.idx_cell], atol=self.tol) or count>niter:
                break
            
            count += 1

        return final_map



    def predict(self, niter=1000):
        idx_neighbours_x, idx_neighbours_y = self.find_all_neighbours()

        final_map = self.maps.copy()

        count=0

        while True: 
            temp_map = final_map.copy()
            t1 = time.time()
            final_map[self.idx_cell] = np.array([self.mean_ignore(temp_map[y,x]) for x,y in zip(idx_neighbours_x, idx_neighbours_y)])
            
            if np.allclose(final_map[self.idx_cell], temp_map[self.idx_cell], atol=self.tol) or count>niter:
                break
            
            count += 1

        return final_map
