# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 12:15:13 2019

@author: Charlie
"""
import numpy as np
import time 
import scipy as sci 
import matplotlib.pyplot as plt
def frange(start, stop, step): # function that allows decimal steps
     i = start
     while i < stop - step:
         yield i
         i += step
def Submatrix(M):
    n = len(M)
    Subarr =  [ [0] * n for k in range(n)] # 2d List to store cofactor matrices
    
    
    for i in range(n):
        M_copy = M 
        M_del = np.delete(M_copy,i, axis=0) # deletes row i
        for j in range(n):
            M_del_2 = np.delete(M_del,j, axis=1) #deletes column j
            Subarr[i][j] = M_del_2
    return Subarr


def Determinant(M):
    n = len(M)
    det = 0 
    if n == 1:
        det = M[0]
    elif n == 2:
        det = M[0][0]*M[1][1] - M[0][1]*M[1][0]
        
    elif n == 3:
        M = np.concatenate((M,M),axis=1)
        M = np.concatenate((M,M),axis=0)
        # This section coule be done without np.concatenate but would require writing the list to a large list
        # 4 times

        for i in range(n):
 
            det += M[i][0]*M[i+1][1]*M[i+2][2] # iterates through list looping back to beginning
            det -= M[0][i]*M[1][i-1]*M[2][i-2] 
            
    elif n >= 4:  # Calcualtes laplace expansions
        
        M2 = Submatrix(M)
        for i in range(n): 
            det += (-1)**(i)*M[i][0]*Determinant(M2[i][0])
        
    return det
    
    
def C_arr(M):
    n = len(M)
    C = np.zeros((len(M),len(M)))
    for i in range(n):
        for j in range(n):
            Cofactor = Determinant(Submatrix(M)[i][j])
            C[j][i] = (-1)**(i+j)*Cofactor              #Cofactor equation for matrix elements
    return np.array(C)
def Inv(M):
# =============================================================================
#     if Determinant(M) == 0:
#         print('Determinant was zero so could not compute for', M)
# =============================================================================

    return 1/(Determinant(M))*C_arr(M)
     
def NxNError(N): # Graph of error vs size of matrix N, N = 7 is a decent value
    Err = []
    Error = 0
    NVAL = []
    for i in range(2,N,1):
        for k in range(0,10):
            arr = np.random.randint(5,size=(i,i))
            while Determinant(arr) == 0:   # checks to make sure determinant is not zero
                arr = np.random.randint(5,size=(i,i))
            Error += np.max(np.absolute(np.matmul(Inv(arr),arr) - np.identity(i))) #A^-1A-I   = 0
            
        Err.append(Error/10)
        NVAL.append(i)
    plt.plot(NVAL,Err)
    plt.xlabel('N')
    plt.ylabel('Max Difference Of Inverse function')
    plt.show()
def NxNTimeInv(N):   
    T = []
    NVAL = []
    
    for i in range(1,N,1):
        print(i)
        arr = np.random.randint(5,size=(i,i))
        begin = time.perf_counter()              # starts a clock
        Inv(arr)
        NVAL.append(i)
        T.append(time.perf_counter() - begin)   # time taken
    plt.plot(NVAL,T)
    plt.xlabel('N')
    plt.ylabel('Time(s)')
    plt.show()
def NxNTime_Compare(N):   # plots a graph of all methods against each other vs NxN

    TI = []
    TL = []
    TS = []
    NVAL = []
    for i in range(1,N,1):   # increasing size of random matrix to solve
        print(i)
        arr = np.random.randint(5,size=(i,i)) 
        b = np.random.randint(5,size=(i,1))
        begin = time.perf_counter()   

        np.matmul(Inv(arr),b)
        TI.append(time.perf_counter() - begin)
        beginL = time.perf_counter()        
        LUD(arr,b)
        TL.append(time.perf_counter() - beginL)
        beginS = time.perf_counter()        
        np.matmul(SVD(arr),b)
        TS.append(time.perf_counter() - beginS)
        

        NVAL.append(i)

    

    plt.plot(NVAL,TI, label='My function')
    plt.plot(NVAL,TL, label='LU Decomp')
    plt.plot(NVAL,TS, label='SV Decomp')
    plt.legend()
    plt.xlabel('N')
    plt.ylabel('Time(s)')
    plt.show()
    
def SVD(M):
    U,s,Vh = sci.linalg.svd(M)   # decomposes
    p = np.reciprocal(s, where=s!=0)    # stops singular value error
    S = sci.linalg.diagsvd(p,len(p),len(p))
    M_SVD = np.matmul(S,np.transpose(U))
    M_SVD = np.matmul(np.transpose(Vh),M_SVD)
    return M_SVD    # returns the inverse matrix from SVD

def Trapeze2d(wire):     # wire = 0,1 for the 2 wires
    Graph = []
    Max = 0 
    Max_Pos = []
    dx = 1E-8     # stops division by zero problems
    for z in frange(1,7,0.1):
        #print('progress =',z)
        TVAL = []
        for x in frange(dx,16,0.1):
            
            theta1 = np.arctan((8-z)/x)     # trig functions
            theta2 = np.arctan((8-z)/(15 - x))
            M = np.array([[np.cos(theta1), -np.cos(theta2)],[np.sin(theta1),np.sin(theta2)]])   # Anglular matrix 
            TVAL.append(float(np.matmul(sci.linalg.inv(M),np.array([[0],[70*9.81]]))[wire]))
            if TVAL[-1] > Max:    # checks if Tension value is a maximum 
                Max = TVAL[-1]
                Max_Pos = [x,z]
        Graph.append(TVAL)
    plt.imshow(Graph,origin='lower',extent =[0,15,0,7],cmap='tab20c',)   
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.colorbar(label='Tension(N)')
    plt.show()
    print('The Maximum Tension reached in String %s'%(str(wire + 1)),'=', round(Max,5),'and the position [x,z] =',Max_Pos)
def LUD(M,B):
    return (sci.linalg.lu_solve(sci.linalg.lu_factor(M),B))
def Singularity():   # Returns graph of errors for Singularity k 
    ErrSVD = []
    ErrLUD = []
    ErrINV = []
    K = []
    dx = 1E-18
    for k in frange(dx,1E-14,1E-18):     #use frange(1E-3,0.2,1E-3) for other graph in report
        
        A = [[1,1,1],[1,2,-1],[2,3,k]]
        
        b = [[5],[10],[15]]
        I = np.matmul(Inv(A),b)
        
        S = np.matmul(SVD(A),b)
        Sv = np.max(np.absolute(np.matmul(A,S) - b))
        L = LUD(A,b)
        Lu = np.max(np.absolute(np.matmul(A,L) - b))
        In = np.max(np.absolute(np.matmul(A,I) - b))
        ErrSVD.append(Sv)
        ErrLUD.append(Lu)
        ErrINV.append(In)
        K.append(k)
    plt.plot(K,ErrSVD, label = 'SV Decomposition')
    plt.plot(K,ErrLUD, label = 'LU Decomposition')
    plt.plot(K,ErrINV, label = 'My function')
    plt.xlabel('log(k)')
    plt.ylabel('Error')
    plt.xscale('log')    # comment out to remove log scale
    plt.legend()

    plt.show()
def alpha(x,y):
    return np.arctan(y/x) 
def beta(x,y):
    return np.arctan(y/(15 - x)) 
def gamma(x,y):
    return np.arctan(abs(7.5-x)/(8-y)) 
def theta1(x,y,z):
    return np.arctan((8-z)/np.sqrt(x**2+y**2)) 
def theta2(x,y,z):
    return np.arctan((8-z)/np.sqrt(y**2 +(15-x)**2))
def phi(x,y,z):
    return np.arctan((8-z)/np.sqrt((7.5-x)**2 +(8 - z)**2))
def Matrix3dNeg(x,y,z):    # 3d anglular array
    return np.array([[-(np.cos(theta1(x,y,z)))*np.cos(alpha(x,y)), np.cos(theta2(x,y,z))*np.cos(beta(x,y)),-(np.cos(phi(x,y,z)))*np.sin(gamma(x,y))],[-(np.cos(theta1(x,y,z)))*np.sin(alpha(x,y)),-np.cos(theta2(x,y,z))*np.sin(beta(x,y)), np.cos(phi(x,y,z))*np.cos(gamma(x,y))],[np.sin(theta1(x,y,z)),np.sin(theta2(x,y,z)),np.sin(phi(x,y,z))]])
    
def Matrix3dPos(x,y,z):   # Changes the top right entry to a negative for use in Trapeze3d when crosses the central line
    MPos = Matrix3dNeg(x,y,z)
    MPos[0,2] = -(Matrix3dNeg(x,y,z))[0,2]
    return MPos
def Trapeze3d(wire):   
    dx = 1E-10
    for z in frange(0,9,1):
           
        print('z =', z)   #
        Graph = []
        Max_Pos = []
        Max = 0
        for y in frange(dx,8.2,0.1):
            TVAL = []
            for x in frange(dx,15,0.1):

                if x <= 7.5:
                    if y >= 8/7.5*x:     # makes values outside the desired triangle zero
                        TVAL.append(0)
                    else:
                        TVAL.append(float(np.matmul(sci.linalg.inv(Matrix3dNeg(x,y,z)),np.array([[0],[0],[70*9.81]]))[wire]))
                if x > 7.5:
                    if y >= (-8/7.5)*x + 16: # same as above
                        TVAL.append(0)
                    else:
                        TVAL.append(float(np.matmul(sci.linalg.inv(Matrix3dPos(x,y,z)),np.array([[0],[0],[70*9.81]]))[wire]))
                if TVAL[-1] > Max:  # Finds max as in 2d function
                    Max = TVAL[-1]
                    Max_Pos = [x,y]
            Graph.append(TVAL)
        plt.imshow(Graph,origin = 'lower',extent =[0,15,0,7],cmap ='jet') # extent scales the imshow axes
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar(label='Tension(N)')
        plt.show()
        print('The Maximum Tension reached in String %s'%(str(wire + 1)),'=', round(Max,5),'and the position [x,y] =',[Max_Pos[0],Max_Pos[1] - dx])



