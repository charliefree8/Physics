# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:28:49 2019

@author: Charlie
"""
import numpy as np 
import matplotlib.pyplot as plt
from IPython import get_ipython
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import sys
import time
from scipy.stats import chisquare
import statistics

get_ipython().run_line_magic('matplotlib', 'inline')

def Analytical_Formula(bins,Iterations): # Analytical Formula function, used for task 1

    
    Yvals = []
    
    XVALS = np.linspace(0,np.pi,100)    # Used to generate sin curve
    SinVals = 0.5*np.sin(XVALS)         # Used to generate sin curve
    start = time.clock()                # Times the iterations
    x = np.random.rand(int(Iterations)) # NumPy array of random values
    Yvals = np.arccos(1-2*x)            # operation on array using the derived analytical formula 
    end = time.clock()      
    
    
   
    
    plt.hist(Yvals,bins, normed=True)       #plots the histogram with the sin curve 
    plt.plot(XVALS,SinVals)     
    plt.xlabel('Random Angle')
    plt.ylabel('Normalised aFrequency')
    plt.show()
    a,b = np.histogram(Yvals,bins,(0,np.pi),normed = True)   # Below section finds the statistics to be printed
    q = 0.5*np.sin(b[0:bins])
    k,l = (chisquare(a[1:-1],q[1:-1]))      # chi square values checking at each bin compared to the sin curve
    print('-----Analytical -----')
    print('Chisquared value:', k)
    print('Time taken:', end - start)    
    print('Mean Value:', np.mean(Yvals))    # Checks the mean value (should be pi/2)
    return (end - start)

def Accept_Reject(bins,Iterations): 

    
    Xvals = []
    
    XVALS = np.linspace(0,np.pi,100)    # Generates sin curve
    SinVals = 0.5*np.sin(XVALS)
    
    start = time.clock()
    for i in range(Iterations):         # Implements accept/reject method
        x = np.random.uniform(0,np.pi)  #random number between desired range
        y = np.random.random()          # random number between 0 and 1
        if y < np.sin(x):               # checks if value is in the required distriution 
            Xvals.append(x)
    end = time.clock()
    
    plt.hist(Xvals,bins=bins,normed= True)  #plots graph with overlayed sin curve
    plt.plot(XVALS,SinVals)
    plt.xlabel('Random Angle')
    plt.ylabel('Normalised Frequency')
    plt.show()
    a,b = np.histogram(Xvals,bins,(0,np.pi),normed = True)  #calculates chisquared 
    q = 0.5*np.sin(b[0:bins])
    k,l = (chisquare(a[1:-1],q[1:-1]))
    print('----- Accept/Reject -----')
    print('Chisquared value = ', k)
    print('Time taken was', end - start)      
    print('Mean Value:', np.mean(Xvals))
    return end - start

def ComparisoninT():
    Time1 = []
    Time2 = []  
    Time_Diff = []
    Iteration = []
    print('Per cent complete = \n')
    for Iterations in range(0,100000,1000):     # smaller range than for report due to runtime
        #Iterations = int(10**k)
        if Iterations % 10000:                     # prints out percentage completed if value is multiple of 100000
            sys.stdout.write('\r'+  str(Iterations/1000000*100))
        Xvals = []
        start = time.clock()        # implements Accept reject method same as above
        for i in range(Iterations):
            x = np.random.uniform(0,np.pi)
            y = np.random.random()
            if y < np.sin(x):
                Xvals.append(x)
        end = time.clock()
        
        time1 = end - start
        
        
        Xvals = []
        Yvals = []
        
        start = time.clock()            #implements analytical solution same as above
        for i in range(Iterations):
            x = np.random.uniform(0,2)
            Yvals.append(np.arccos(1-x))
            Xvals.append(x)
        end = time.clock()
        
        time2 = end - start
        diff = time1 - time2
        Time1.append(time1)
        Time2.append(time2)
        Time_Diff.append(diff)
        Iteration.append(Iterations)
    
    print('Plot of Time taken for both methods' )           #   plots the times against number of iterations against each other
    plt.plot(Iteration,Time1,'.',label = 'Accept Reject')
    plt.plot(Iteration,Time2,'.',label = 'Analytical')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()
    
    print('Plot of time differrence between both methods')  # plots the time differrence between the two methods
    plt.plot(Iteration,Time_Diff,'.')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Time Differrence (s)')

    plt.show()
    
def Decay1D(Iterations):        

    Yvals = []
    v = 2000
    tau = 550E-6
    for i in range(0,Iterations):   # uses accept reject method to model radioactive decay 
        x = np.random.random()
        d = np.random.uniform(0,2)
        N = np.exp(-d/(v*tau))      # exponential radioactive decay function 
        if x < N:                   #accept reject
            Yvals.append(d)
    print('----- 1D dist of decay positions -----' )
    plt.hist(Yvals,bins = 'auto')   #plots exponential distribution, with the x axis being the distance from source
    plt.xlabel('Distance from source')
    plt.ylabel('Frequency')
    plt.show()
def Decay3D():
    
    #plt.switch_backend('Qt5Agg') should make plot appear in seperate window but having bugs recently so have to comment out
    Xvals = []
    Yvals = []
    Zvals = []
    Iterations = 5000
    v = 2000
    tau = 550E-6
    print('Non modified graph')     
    for i in range(0,Iterations):       # accept reject method showing bunching at poles for spherical distribution
        X = np.random.random()
        r = 1
        N = np.exp(-r/(v*tau))

        
        theta = np.random.uniform(0,2*np.pi)
        phi = np.random.uniform(0,np.pi)
        

        if X < N:
            x = r*np.cos(theta)*np.sin(phi) # conversion to spherical coordinates
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(phi)
            Xvals.append(x)
            Yvals.append(y)
            Zvals.append(z)
    fig = plt.figure()                  #plots 3D figure, with graphics on Automatic will be interactive as backend switches not working 
    ax = Axes3D(fig)
    ax.scatter(Xvals,Yvals,Zvals,s=2)
    ax.set_aspect('equal')
    plt.show()
    
    

    Xvals = []
    Yvals = []
    Zvals = []
    print('Modified decay direction (Isotropic)')
    for i in range(0,Iterations):           # Plots isotropic distribution
        X = np.random.random()
        r = 1
        N = np.exp(-r/(v*tau))

        
        theta = np.random.uniform(0,2*np.pi)
        phi = np.arccos(1-2*np.random.random()) # Phi distribution needed for isotropic solution
        

        if X < N:
            x = r*np.cos(theta)*np.sin(phi) # conversion to spherical coordinates
            y = r*np.sin(theta)*np.sin(phi)
            z = r*np.cos(phi)
            Xvals.append(x)
            Yvals.append(y)
            Zvals.append(z)
    fig = plt.figure()                      # 3d scatter plot as above 
    ax = Axes3D(fig)
    ax.scatter(Xvals,Yvals,Zvals,s=2)
    ax.set_aspect('equal')
    plt.show()
    

def GammaDetector(Iterations):
        


    v = 2000
    tau = 550E-6
    Xvals = []
    Yvals = []
    gauss_x, gauss_y = [], []
    counter = 0
    print('----- Gamma Ray Detection -----')
    print('Percentage complete = ')
    for i in range(0,Iterations):
        counter+=1
        progress = (counter/Iterations)*100
        sys.stdout.write("\r"+ str(round(progress)))    # writes the percentage completed
        X = np.random.random()
        r = np.random.uniform(0,2)
        N = np.exp(-r/(v*tau))
    
        
        theta = np.random.uniform(0,2*np.pi)
        phi = np.arccos(1-2*np.random.random()) # implements the 3d isotropic distribution from above
        d = 2 - r
    
        if X < N and theta < np.pi and abs(d*np.cos(theta)/np.tan(phi)) <= 10 and abs(d*np.sin(theta)/np.tan(phi)) <= 10:
    
            x = d*np.cos(theta)/np.tan(phi)     # Trig equations to find position on screen (x,y)
            y = d*np.sin(theta)/np.tan(phi)
            Xvals.append(x)
            Yvals.append(y)
            
            gauss_x.append(np.random.normal(x, 0.1))    # adds gausssian term to position to simulate resolution
            gauss_y.append(np.random.normal(y, 0.3))
            
            
    fig, ((ax1,ax2)) = plt.subplots(1,2,figsize = (12,5))           # plots 2d histogram, one of non smeared and one gaussian smeared plot
    h = ax1.hist2d(Xvals,Yvals,bins=100,cmap='jet',norm=LogNorm())  # LogNorm() to allow the distribution to be more clearly visualised
    ax1.set_aspect('equal')
    plt.colorbar(h[3],ax =ax1)
    ax2.hist2d(gauss_x,gauss_y,bins=100,cmap='jet',norm=LogNorm())
    plt.colorbar(h[3],ax =ax2)
    ax2.set_aspect('equal')


    plt.show()
def Confidence(sigma,Error,n):   # Checks standard deviation in confidence for a  number of repeats around the limit 

    Sig_event = 0
    for Iteration in range(0,100000):
        
        if n == 0:
            L = 12
        if n == 1:
            L = np.random.normal(12,Error)
            while L <= 0:                   
                L = np.random.normal(12,Error)
        
        N = np.random.normal(5.7,0.4)           #Number of events (Gauss)
        B = np.random.poisson(N)                # Number of background events (poisson)
        Events = np.random.poisson(L*sigma)     # number of experimental events (poisson)
        Total = Events + B                      # Total number of events recorded
        if Total > 5:                           # checks for significnce criteria of more than 5 events
            Sig_event += 1
    return Sig_event*100/Iteration

def Collider():
    print('Checking for sigma =')
    Sigma = []
    ConfidenceLIST = []
    for sigma in np.arange(0.1,0.6,0.01):      # iterates through range of sigma's0.001 used as step in report but 0.01 here for faster runtime 

        sys.stdout.write("\r"+ str(sigma)) 
        X = Confidence(sigma,0,0)               # generates confidence value for specific sigma 
        ConfidenceLIST.append(X)
        Sigma.append(sigma)
    for i in range(len(ConfidenceLIST)):                # searches for first instance in list of >95% confidence, to put set the limit 
        if ConfidenceLIST[i] >= 95:
            print(' \n The limit on the cross section is', Sigma[i],'+/- 0.001 at 95% confidence level ')
            break

    plt.plot(Sigma,ConfidenceLIST,'.')              # plots the graph of confidence vs sigma 
    plt.xlabel('Sigma (nb)' )
    plt.ylabel('Confidence / percentage of significant events')
    plt.title('Luminosity = 12')
    plt.show()
def ColliderLumError(Error):
    Sigma = []
    ConfidenceLIST = []
    print('Checking for sigma =')       
    for sigma in np.arange(0.1,0.6,0.01):      # Same as above function 0.001 used as step in report but 0.01 here for faster runtime 

        sys.stdout.write("\r"+ str(sigma))
        X = Confidence(sigma,Error,1)           # 1 in 3rd entry causes the error to be used
        ConfidenceLIST.append(X)                # to generate the luminosity from a normal dist.
        Sigma.append(sigma)
    for i in range(len(ConfidenceLIST)):
        if ConfidenceLIST[i] >= 95:
            print(' \n The limit on the cross section is', Sigma[i],' +/- 0.08 at 95% confidence level ')
            break
        
    plt.plot(Sigma,ConfidenceLIST,'.')
    plt.xlabel('Sigma (nb)' )
    plt.ylabel('Confidence / percentage of significant events')
    plt.title('Luminosity = 12 +/-' + str(Error))
    plt.show()  


def ColliderError():
    ConfidenceLIST1 = []
    ConfidenceLIST2 = []
    for i in range(0,100):
        X = Confidence(0.4,0,0)             # For the value at which sigma is found, does 100 repeats to find the std dev in the confidence value
        ConfidenceLIST1.append(X)
        
    print('Error in cross section confidence is for constant L = ', statistics.stdev(ConfidenceLIST1)) # returns value of 0.001
    
    
    for i in range(0,100):                  # Same as above but for vsalue from L with error +/- 5
        X = Confidence(0.5,5,1)
        ConfidenceLIST2.append(X)
    print('Error in cross section confidence for  L = 12 +/- 5 ', statistics.stdev(ConfidenceLIST2))
def MeanConversion():       # Function that looks at how the two methods from task 1 converge to the expected mean value of pi/2

    Mean_Mean_Analytical = []   
    Mean_Mean_Accept = []
    Iteration = []
    Mean_Diff_Accept = []
    Mean_Diff_Analytical = []
    for Iterations in np.arange(100,10000,1000):# do range(0,int(1000),1) for quick graph that shows
        #Iterations = int(10**k)   #actual graph in report used much larger range with small steps but takes ~ 5 hours to run so short version implemented to see that it works.
        
        Mean_Analytical = []
        Mean_Accept = []
        Iteration.append(Iterations)

        for i in range(0,50):        # goes through 0 repeats at each iteration total, to find a mean due to high variability 

            Xvals = []
            for i in range(Iterations):     # accept reject methid same as task 1 
                x = np.random.uniform(0,np.pi)
                y = np.random.random()
                if y < np.sin(x):
                    Xvals.append(x)
    
            Mean_Accept.append(np.mean(Xvals))   # Takes the mean value from the distribution
            
            
            Xvals = []
            Yvals = []
            
    
            x = np.random.rand(int(Iterations)) # Analytical method same as task 1 
            Yvals = np.arccos(1-2*x)

            
            Mean_Analytical.append(np.mean(Yvals))
            
        
        Mean_Mean_Accept.append(np.mean(Mean_Accept)) # finds the mean of the 50 repeats for both methods
        Mean_Mean_Analytical.append(np.mean(Mean_Analytical))
        Mean_Diff_Accept.append(abs(np.pi/2 - Mean_Mean_Accept[-1]))   # finds the distance from the mean for all values
        Mean_Diff_Analytical.append(abs(np.pi/2 - Mean_Mean_Analytical[-1]))
    line = np.empty(len(Iteration))         # plots line at pi/2
    line.fill(np.pi/2)
    plt.plot(Iteration,Mean_Mean_Analytical,'.',label = 'Analytical',markersize = 4,alpha = 0.25) # plots log of the distributions for both methods
    plt.plot(Iteration,Mean_Mean_Accept,'.',label = 'Accept reject',markersize =4,alpha = 0.25)
    plt.plot(Iteration,line,label = 'Real value pi/2')
    plt.xlabel('Iterations')
    plt.ylabel('mean value')  
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    plt.plot(Iteration,Mean_Mean_Analytical,'.',label = 'Analytical',markersize = 4,alpha = 0.25)   # plots distributions of both values
    plt.plot(Iteration,Mean_Mean_Accept,'.',label = 'Accept reject',markersize =4,alpha = 0.25)
    plt.plot(Iteration,line,label = 'Real value pi/2')
    plt.xlabel('Iterations')
    plt.ylabel('mean value')  
    plt.legend()
    plt.show()
    
    plt.plot(Iteration,Mean_Diff_Accept,label = 'Accept' )         # plots log of the distance from the mean for both methods
    plt.plot(Iteration,Mean_Diff_Analytical, label = 'Analytical')
    plt.xlabel('Iterations')
    plt.ylabel('Distance from Expected value')  
    plt.xscale('log')
    plt.legend()
    plt.show()
    
    plt.plot(Iteration,Mean_Diff_Accept,'.',label = 'Accept' )          # same as above but with dots instea dof lines to see which was easier to see result 
    plt.plot(Iteration,Mean_Diff_Analytical,'.',label = 'Analytical')
    plt.xlabel('Iterations')
    plt.ylabel('Distance from Expected value')  
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(Iteration,Mean_Diff_Accept,'.',label = 'Accept' )              
    plt.plot(Iteration,Mean_Diff_Analytical,'.',label = 'Analytical')
    plt.xlabel('Iterations')
    plt.ylabel('Distance from Expected value')  
    plt.legend()
    plt.show()
    a = np.array(Mean_Diff_Accept)
    b = np.array(Mean_Diff_Analytical)
    
    c = a - b
    plt.plot(Iteration,c)               # plots differrence between the distances from the mean, to see if one dominates over the other 
    plt.xlabel('Iterations')
    plt.ylabel('Differrence Between Distance from expected value')    
    plt.xscale('log')
    plt.show()
    print('Mean differrence', np.mean(c))
MyInput ='0'
while MyInput != 'q':           #Basic menu system 
    MyInput = input('Enter a choice, "a --- Comparison of methods", "b --- Decay Position and Isotropic Gamma distribution", "c - Gamma ray detection from beam decay", "d -------- Cross section calculation from Collider","e" ------ mean conversion, "f" -------- To run all with set inputs(RECOMMENDED), or  "q" to quit: ')
    print('You entered the choice: ', MyInput)
    if MyInput == 'a':
        print('You have chosen part (a)')
        parsed = False
        while not parsed:               # Makes sure input is only in useable form
            bins_input = input('enter a value for number of bins (100 is a good choice)')
            try:
                bins = int(bins_input)
                parsed = True               # if input is correct form, parsed becomes true and infinite loop is ended
            except ValueError:  
                print('Enter a valid number')
            
        parsed = False
        while not parsed:
            iter_input = input('enter a value for number of iterations')
            try:
                Iterations = int(iter_input)
                parsed = True
            except ValueError:
                print('Enter a valid number')  
            
        a = Analytical_Formula(bins,Iterations)
        
        b = Accept_Reject(bins,Iterations)
        
        print('\n Analytical Method was', b - a,'s faster than Accept reject method')
        
        ComparisoninT()
      
    elif MyInput == 'b':
        print('You have chosen part b')
        parsed = False
        
        while not parsed:
            Iter_input = input('enter a value for number of Iterations e.g 100000')
            try:
                Iterations = int(Iter_input)
                parsed = True
            except ValueError:
                print('Enter a valid number')
       
        Decay1D(Iterations)
        
        
        print('For an interactive plot of Isotropic decay Tools >> Preferences >> IPython console >> Graphics --> Automatic and then restart the kernel, as the old method of switching backens does not work on new update')
  
        Decay3D()
        


       
        
    elif MyInput == 'c':
        print('You have chosen part d')
        parsed = False
        while not parsed:
            dt_input = input('enter a value for number of iterations e.g 1000000')
            try:
                Iterations = int(dt_input)
                parsed = True
            except ValueError:
                print('Enter a valid number')
        GammaDetector(Iterations)
        
      
    
    elif MyInput == 'd':
        print('You have chosen part d')

        Collider()
        parsed = False
        while not parsed:
            Error_input = input('enter a value for error on Luminosity, to be modelled as a gaussian')
            try:
                Error = float(Error_input)
                parsed = True
            except ValueError:
                print('Enter a valid number')

        ColliderLumError(Error)
        
    elif MyInput == 'e':
        print('You have chosen part e')
        print('------------ Differrent from graph in report as actual graph takes ~ 5 hours to produce, this is a quick version  ----------')
        
        
        MeanConversion()
            
    elif MyInput == 'f':
        print('You have chosen to run all parts')
        Analytical_Formula(100,10000)
        Accept_Reject(100,100000)
        ComparisoninT()
        Decay1D(100000)
        Decay3D()
        GammaDetector(100000)
        Collider()
        ColliderLumError(5)
        ColliderError()
        MeanConversion()
        print('The last function only shows good results with lots of data, but takes a long time to run so not shown, see report for example (~5 hours')
    elif MyInput != 'q':
        print('This is not a valid choice')
             
print('You have chosen to finish - goodbye.')
