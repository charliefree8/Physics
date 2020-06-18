
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:37:29 2019

@author: Charlie
"""
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')

def Grid(Size):
    # creates a meshgrid to plot with 
    x,y = np.meshgrid(np.arange(0,Size),np.arange(0,Size))
    return x,y
def ValueGrid(Size,Guess, Top = 0,  Bottom = 0, Left = 0, Right = 0): 
    # Creates a 2D array with the guessed value 
    Vals = np.empty((Size,Size))
    Vals.fill(Guess)
    #Vals = np.random.randint(0,50,(Size,Size)) # sets guesses to individually random guesses
    
    # Setting the boundary conditions as specified in function definition # default is zero if not defined
    Vals[:,0] = Left
    Vals[:,Size - 1] = Right
    Vals[Size - 1,:] = Top
    Vals[0,:] = Bottom
    
    return Vals

def Laplace(Size,Guess, Rho, Top = 0,  Bottom = 0, Left = 0, Right = 0):
    Vals = ValueGrid(Size,Guess,Top,Bottom,Left,Right)
    Vold = 1
    Vnew = 2
    Iterations = 0 
    while abs(Vnew - Vold) > 1E-6:        # Will terminate when convergence limit is met
        Vold = np.mean(Vals)
        for i in range(1,Size - 1):
            for j in range(1,Size - 1):
                Vals[i,j] = 1/4*(Vals[i-1,j]+Vals[i+1,j]+Vals[i,j-1]+Vals[i,j+1]) + Rho/4
        Vnew = np.mean(Vals)
        Iterations += 1 
    print('Number of Iterations =', Iterations)
    return Vals
                    
def Plot(Size,Guess, Rho, Top = 0,  Bottom = 0, Left = 0, Right = 0):   # Plots the Graph for Task 1
    colour = plt.cm.hot
    x,y = Grid(Size)
    Vals = Laplace(Size,Guess,Rho,Top,Bottom,Left,Right)
    plt.contourf(x,y,Vals,100,cmap=colour)
    plt.colorbar()
    plt.show()

# Show example for task 1 Boundary conditions can be varied, with undefined conditions set to zero
print('========== Graph showing solutions to Laplaces equation with some initial boundary conditions ==========')
Plot(50,50,0,Top = 100)
def ItvN():             # Produces Graph with values found from running the code at differrent sizes, A loop to generate this data would work but the runtime is quite long..
     xp = np.linspace(0,100,20)
     y = [114,467,1024,1766,2683,3764,5003,6392,7927,9601]
     x = [10,20,30,40,50,60,70,80,90,100]    
     z = np.polyfit(x,y,2)                  # Fits a quadratic to the data
     p = np.poly1d(z)                       # Generates Numbers to plot from the fit     
     plt.plot(x,y,'.',xp,p(xp),'--')        
     plt.xlabel('N')
     plt.ylabel('Iterations')
     plt.legend(('No. Iterations','Polynomial Fit'))
     plt.show()

# Uncomment to show plot and fit of Iteration vs Grid size graph
print('========== Graph of Number of Iterations vs Grid Size showing quadratic relationship ==========' )
ItvN()
def ItvCL():                                            # As above but for Convergence limit
     x1 = np.linspace(1,11,10)
     y1 = [39,446,1004,1564,2123,2683,3243,3803,4362,4922]
     xp1 = np.linspace(1,11,10)
     z1 = np.polyfit(x1,y1,1)
     p1 = np.poly1d(z1)
     plt.plot(x1,y1,'.',xp1,p1(xp1),'--')
     plt.xlabel('- Log(Convergence Limit)')
     plt.ylabel('Iterations')
     plt.legend(('No. Iterations','1st order Polynomial Fit'))
     plt.show()
print('========== Graph of Number of Iterations vs -Log(Convergence Limit) showing linear relationship ==========' )

ItvCL()

def ValueGridCap(Size,Guess, CapLength, Separation): 
    # Creates a 2D array with the guessed value 
    #Vals = np.empty((Size,Size))                               # Uncomment the top 2 lines and comment out the Vals = np.random.randint(0,50,(Size,Size)) to change to Set guesses   

    #Vals.fill(Guess)                                
    Vals = np.random.randint(0,50,(Size,Size))       
    
    # Setting the boundary conditions as specified in function definition 
# =============================================================================
    k = int((Size + Separation)/2)
    l = int((Size - CapLength )/2)
    m = int((Size + CapLength)/2)
    n = int((Size - Separation)/2)
    Vals[k,l:m] = 100
    Vals[n,l:m] = -100
# =============================================================================

    
    return Vals

def LaplaceCap(Size,Guess, Rho, CapLength = int(10), Separation = int(10)):
    Vals = ValueGridCap(Size,Guess,CapLength,Separation)
    Vold = 1
    Vnew = 2
    Iterations = 0                                        # Counter to check how many iterations occur
    while abs(Vnew - Vold) > 1E-6:                        # Convergence limit
        Vold = np.mean(Vals)
        for i in range(0,Size-2):                         
            for j in range(0,Size):
                if Vals[i,j] == 100:                      # Doesn't update values at on capacitor surface
                    pass                                  # i.e initial conditions aren't updated
                elif Vals[i,j] == -100:
                    pass
                elif j == Size - 1:                       # Sets up periodic boundary conditions in the j direction to simulate an infinte plane
                    Vals[i,j] = 1/4*(Vals[i-1,0]+Vals[i+1,0]+Vals[i,j-1]+Vals[i,1]) + Rho/4
 



                
                else:
                    Vals[i,j] = 1/4*(Vals[i-1,j]+Vals[i+1,j]+Vals[i,j-1]+Vals[i,j+1]) + Rho/4
        Vnew = np.mean(Vals)
        Iterations += 1
    print('Number of Iterations =', Iterations)
    return Vals
                    
def PlotCap(Size,Guess, Rho, CapLength = int(10), Separation = int(10)):
    colour = plt.cm.seismic
    x,y = Grid(Size)
    Vals = LaplaceCap(Size,Guess,Rho,CapLength,Separation)
    E = np.gradient(Vals)                    # Generate vector values From Electric Potential array
    plt.contourf(x,y,Vals,100,cmap=colour)
    plt.colorbar()
    plt.quiver(x[::2,::2],y[::2,::2],E[1][::2,::2],E[0][::2,::2])        # Plots E-field vectors from above values
    plt.show()
print('========== Graph of Finite Capacitor Solution, arrows represent E field, Colour represents Potential ==========' )

PlotCap(50,0,0,CapLength = 20,Separation = 12)    

print('========== Graph of Infinite capacitor solution ==========' )

PlotCap(50,0,0,CapLength = 50,Separation = 12) 

def PokerTemp(N_PosNodes,N_Iter,Total_Time,T_C=0,T_H = 1000,NoHeatLoss=True):
    
    dx = 0.5/N_PosNodes 
    dt = Total_Time/N_Iter 
    alpha = 59/(450*7900) # Constants given in Notes
    gamma = alpha*dt/(dx)**2
    
    M=np.zeros((N_PosNodes,N_PosNodes))         
    Pos = np.linspace(0,0.5,N_PosNodes)
    Temp = np.empty((N_PosNodes,1))             #Defines an array of Temp values, starting at 20 C      
    Temp.fill(20)
    
    
    for i in range(1,N_PosNodes-1):             #Defines the matrix of coefficients to solve, calculated through derivation
          M[i,i] =1+2*gamma 
          M[i,i+1] = M[i,i-1]=-gamma 
        
    M[0,0] = 1+3*gamma                          # Adjustments to matrix due to "ghost cells"
    M[0,1] = M[-1,-2]=-gamma
    
    if NoHeatLoss == True:    
        M[-1,-1] = 1+gamma
    
    elif NoHeatLoss == False:
        M[-1,-1] = 1+3*gamma
        
    
    for k in range(N_Iter):
        Temp[0] += 2*gamma*T_H                 # Defines the first and last (if Neumann is false)
        
        if NoHeatLoss == False:
            Temp[-1] += 2*gamma*T_C
            
        Temp = np.linalg.solve(M,Temp)
        
    return Temp,Pos

def PlotMultiTemp(N_PosNodes,N_Iter,Timestep, NoHeatLoss = True,MaxTime = 4000):
    plt.clf()
    for i in range(30,MaxTime,Timestep):
        Temp,Pos = PokerTemp(N_PosNodes,N_Iter,i,NoHeatLoss = NoHeatLoss)
        plt.plot(Pos,Temp, label='T='+str(i) +'s')
    plt.xlabel('Position (m)')
    plt.ylabel('Temperature (Celsius)')
    plt.legend(loc = 'center left', bbox_to_anchor = (1,0.5))
    plt.show()
    
print('========== Graph of Temperature of hot poker vs position along poker at differrent times ==========' )
print('========== No Heat Loss ==========')
PlotMultiTemp(100,100,5000,NoHeatLoss = True,MaxTime = 40000)
print('========== If one end of the poker is in an ice bath ==========')
PlotMultiTemp(100,100,500,NoHeatLoss = False, MaxTime = 4000)           
def PlotSingTemp(N_PosNodes,N_Iter, NoHeatLoss = True, Time = 2000,Maxtime = 4000,Time_interval = 50):
    
    plt.switch_backend('Qt5Agg')
    for i in range(0,Maxtime,Time_interval):
        plt.clf()
        Temp,Pos = PokerTemp(N_PosNodes,N_Iter, Total_Time = i, NoHeatLoss = NoHeatLoss)
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.imshow(Temp,cmap='hot', vmin = 0 ,vmax = 1000 ,extent = [0,19,50,0])
        plt.colorbar()
        plt.title('Time =' +str(i) +'s')
        plt.ylabel('Postion (cm)')
        plt.show()
        plt.pause(1/(i**2 +1))        

# A window will open which will show an animation of the heat distribution in 1D through time
print('========== Final State of Poker ===========')

PlotSingTemp(100,100,Time = 2000,NoHeatLoss = False) 
# If visualisation of No heat loss condition change above NoHeatLoss to True
