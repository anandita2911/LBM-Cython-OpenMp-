import math, time
import numpy as np
cimport numpy as cnp
from cython cimport floating
from cython.parallel cimport parallel, prange
from libc.stdio cimport printf

# Defining Global Variables
cdef float Reynolds_Number = 50
cdef int Timestep = 2000
cdef int Nodes_X = 30
cdef int Nodes_Y = 30
cdef int NDir = 9 
cdef float U_Wall = 0.1
cdef float Initial_Density = 2.7
cdef float Kinematic_Viscocity = (Nodes_X*U_Wall)/Reynolds_Number
cdef float Relaxation_Time = (6*Kinematic_Viscocity+1)/2

cdef Stream(cnp.ndarray[cnp.double_t, ndim = 3] f):
    cdef int i = 0, j = 0

    with nogil:
        # Direction 1 (1,0)
        for i in range(0,Nodes_X,1):
            for j in range(Nodes_Y-1,0,-1):
                f[i,j,1] = f[i,j-1,1]
        
        # Direction 2 (0,1)
        for i in range(Nodes_X-1,0,-1):
            for j in range(0,Nodes_Y,1):
                f[i,j,2] = f[i-1,j,2]

        # Direction 3 (-1,0)
        for i in range(0,Nodes_X,1):
            for j in range(0,Nodes_Y-1,1):
                f[i,j,3] = f[i,j+1,3]
        
        # Direction 4 (0,-1)
        for i in range(0,Nodes_X-1,1):
            for j in range(0,Nodes_Y,1):
                f[i,j,4] = f[i+1,j,4]
        
        # Direction 5 (1,1)
        for i in range(Nodes_X-1,0,-1):
            for j in range(Nodes_Y-1,0,-1):
                f[i,j,5] = f[i-1,j-1,5]
        
        # Direction 6 (-1,1)
        for i in range(Nodes_X-1,0,-1):
            for j in range(0,Nodes_Y-1,1):
                f[i,j,6] = f[i-1,j+1,6]
        
        # Direction 7 (-1,-1)
        for i in range(0,Nodes_X-1,1):
            for j in range(0,Nodes_Y-1,1):
                f[i,j,7] = f[i+1,j+1,7]
        
        # Direction 8 (1,-1)
        for i in range(0,Nodes_X-1,1):
            for j in range(Nodes_Y-1,0,-1):
                f[i,j,8] = f[i+1,j-1,8]

cdef Bounceback(cnp.ndarray[cnp.double_t, ndim = 3] f, cnp.ndarray[cnp.int_t, ndim = 2] Walls):
    cdef int i = 0, j = 0
    cdef double temp = 0

    with nogil, parallel(num_threads = 2):
        for i in prange(0,Nodes_X,schedule = 'guided'):
            for j in range(0,Nodes_Y):
                if (Walls[i,j] == 1):
                    temp = f[i,j,1]
                    f[i,j,1] = f[i,j,3]
                    f[i,j,3] = temp

                    temp = f[i,j,5]
                    f[i,j,5] = f[i,j,7]
                    f[i,j,7] = temp

                    temp = f[i,j,2]
                    f[i,j,2] = f[i,j,4]
                    f[i,j,4] = temp

                    temp = f[i,j,6]
                    f[i,j,6] = f[i,j,8]
                    f[i,j,8] = temp

cdef Compute_Fields(cnp.ndarray[cnp.double_t, ndim = 3] f, cnp.ndarray[cnp.double_t, ndim = 2] U, cnp.ndarray[cnp.double_t, ndim = 2] V, cnp.ndarray[cnp.double_t, ndim = 2] Density, cnp.ndarray[cnp.int_t, ndim = 2] C):
    cdef int i = 0, j = 0, k = 0

    with nogil, parallel(num_threads = 2):
        for i in prange(0, Nodes_X, schedule = 'guided'):
            for j in range(0,Nodes_Y):
                Density[i,j] = 0
                U[i,j] = 0
                V[i,j] = 0
                for k in range(0,NDir):
                    Density[i,j] = Density[i,j] + f[i,j,k]
                    U[i,j] = U[i,j] + f[i,j,k]*C[k,0]
                    V[i,j] = V[i,j] + f[i,j,k]*C[k,1]
                U[i,j] = U[i,j]/Density[i,j]
                V[i,j] = V[i,j]/Density[i,j]
    
cdef Collision(cnp.ndarray[cnp.double_t, ndim = 3] f, cnp.ndarray[cnp.double_t, ndim = 3] feq):
    cdef int i = 0, j = 0, k = 0

    with nogil, parallel(num_threads = 2):
        for i in prange(0,Nodes_X, schedule = 'guided'):
            for j in range(0,Nodes_Y):
                for k in range(0,NDir):
                    f[i,j,k] = f[i,j,k] - (f[i,j,k] - feq[i,j,k])/Relaxation_Time

cdef Compute_feq(cnp.ndarray[cnp.double_t, ndim = 3] feq, cnp.ndarray[cnp.double_t, ndim = 2] U, cnp.ndarray[cnp.double_t, ndim = 2] V, cnp.ndarray[cnp.double_t, ndim = 2] Density, cnp.ndarray[cnp.int_t, ndim = 2] C, cnp.ndarray[cnp.double_t, ndim = 1] Weight):
    cdef int i = 0, j = 0, k = 0
    cdef double CdotU = 0

    with nogil, parallel(num_threads = 2):
        for i in prange(0,Nodes_X,schedule = 'guided'):
            for j in range(0,Nodes_Y):
                for k in range(0,NDir):
                    CdotU = C[k,0] * U[i,j] +  C[k,1]* V[i,j]
                    feq[i,j,k] = Density[i,j]*Weight[k]*(1 + 3*(CdotU) + 4.5*(CdotU*CdotU) - 1.5*(U[i,j]*U[i,j] + V[i,j]*V[i,j]))

cdef LBM():
    start = time.time()
    # Allocate Memory
    cdef cnp.ndarray[cnp.double_t, ndim = 3] f = np.zeros((Nodes_X, Nodes_Y, NDir), dtype = "double")
    cdef cnp.ndarray[cnp.double_t, ndim = 3] feq = np.zeros((Nodes_X, Nodes_Y, NDir), dtype = "double")
    cdef cnp.ndarray[cnp.double_t, ndim = 2] U = np.zeros((Nodes_X, Nodes_Y), dtype = "double")
    cdef cnp.ndarray[cnp.double_t, ndim = 2] V = np.zeros((Nodes_X, Nodes_Y), dtype = "double")
    cdef cnp.ndarray[cnp.double_t, ndim = 2] Density = np.zeros((Nodes_X, Nodes_Y), dtype = "double")
    cdef cnp.ndarray[cnp.int_t, ndim = 2] Walls = np.zeros((Nodes_X, Nodes_Y), dtype = "int")
    cdef cnp.ndarray[cnp.int_t, ndim = 2] C = np.zeros((NDir, 2), dtype = "int")
    cdef cnp.ndarray[cnp.double_t, ndim = 1] Weight = np.ones((NDir), dtype = "double")

    # Initialisation
    Weight = np.array([0.444445, 0.111112, 0.111112, 0.111112, 0.111112, 0.027778, 0.027778, 0.027778, 0.027778])
    C = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]])
    Density.fill(Initial_Density)
    Walls[0,:] = Walls[Nodes_Y-1,:] = Walls[:,0] = Walls[:,Nodes_X-1] = 1
    Compute_feq(feq, U, V, Density, C, Weight)
    f = feq.copy()

    for t in range(0,Timestep):
        print(t)
        Stream(f)  
        Bounceback(f, Walls)
        Compute_Fields(f, U, V, Density, C)

        # Moving Wall Boundary Condition
        U[Nodes_X-1,:] = U_Wall
        V[Nodes_X-1,:] = 0

        Compute_feq(feq, U, V, Density, C, Weight)
        Collision(f, feq)

        # Putting f = feq For Moving Wall
        for j in range(0,Nodes_Y):
            for k in range(0,NDir):
                f[Nodes_X-1][j][k] = feq[Nodes_X-1][j][k].copy()

    with open('UV_Cython.plt', 'w') as file:
        for i in range(0,Nodes_X):
            for j in range(0,Nodes_Y):
                file.write('\n'+str(round(1*i/Nodes_X,6))+'\t'+str(round(1*j/Nodes_Y,6))+'\t'+str(round(U[j][i]/U_Wall,6))+'\t'+str(round(V[j][i]/U_Wall,6))+'\t'+str(round(Density[j][i],6)))
    file.close()
    end = time.time()
    print(end-start)

LBM()
