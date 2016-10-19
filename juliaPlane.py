#!/usr/bin/env python3

import cplane_np as cpnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba as nb

class JuliaPlane(cpnp.ComplexPlaneNP):
    """
    This module allows the user to create a plane of
    complex numbers, which can then be transformed 
    under the function julia(c). The show function then 
    provides the resulting Julia set based on a color
    map.

    Attributes:
        xmax (float) : maximum horizontal axis value
        xmin (float) : minimum horizontal axis value
        xlen (int)   : number of horizontal points
        ymax (float) : maximum vertical axis value
        ymin (float) : minimum vertical axis value
        ylen (int)   : number of vertical points
        plane        : stored complex plane implementation
        f    (func)  : function displayed in the plane
    """
    
    def __init__(self, xmin, xmax, xlen, ymin, ymax, ylen):
        """
        Args:
            xmax (float) : maximum horizontal axis value
            xmin (float) : minimum horizontal axis value
            xlen (int)   : number of horizontal points
            ymax (float) : maximum vertical axis value
            ymin (float) : minimum vertical axis value
            ylen (int)   : number of vertical point
        """
        self.xmin = xmin
        self.xmax = xmax
        self.xlen = xlen 
        self.ymin = ymin
        self.ymax = ymax
        self.ylen = ylen
        self.c = -1.037 + 0.17j
        self.f = np.vectorize(cpnp.julia(self.c)) #function must be vectorized to be applied to dataframe
        
        self.refresh()
        
    def refresh(self):
        """Regenerate complex plane.
        For every point (x + y*1j) in self.plane, replace
        the point with the value self.f(x + y*1j). 
        """
        
        real = np.linspace(self.xmin,self.xmax,self.xlen)
        imaginary = np.linspace(self.ymin,self.ymax,self.ylen)
        x, y= np.meshgrid(real,imaginary)
        z = x+ y*1j
        rl = np.linspace(self.xmin,self.xmax,self.xlen)
        imag = np.linspace(self.ymin,self.ymax,self.ylen)
        self.plane = pd.DataFrame(self.f(z), index=imag, columns=rl)
 
        
    def set_f(self, complexNum):
        """Reset the transformation function f.
        Refreshes the plane after setting attribute 
        f to the function julia(complexNum).
        Args:
            ComplexNum (complex) : complex number to use
            as argument in cplane_np.julia(c).
        """
        
        try:
            complexNum.imag
        except AttributeError:
            print("Invalid argument; argument must be a complex number")
        else:
            self.c = complexNum
            self.f = np.vectorize(cpnp.julia(complexNum))
            self.refresh() #calls refresh to have function change take effect

    def show(self):
        """Plots an image of the complex plane
        using the imshow plot in matplotlib. Includes
        a label indicating the value of c input into 
        the function julia(c) used to transform the
        complex plane.
        """
        
        plt.imshow(self.plane, cmap = 'hot', interpolation = 'bicubic', extent = (self.xmin, self.xmax, self.ymin, self.ymax))
        cValue = str(self.c)
        plt.text(-1.75, 1.75, '$c=$'+cValue, bbox={'facecolor':'white', 'alpha':0.75, 'pad':2})
            
    #reading and writing with files info referenced from https://docs.python.org/3.5/tutorial/inputoutput.html
 
    def toCSV(self, filename):
        """Saves the arguments for the complex plane
        to a csv file called filename, then appends 
        self.plane.
        
        Args:
            filename (string) : name of the file to 
                write complex plane to
        """
        
        df = self.plane
        rl = [float(i) for i in df.columns[0:]] #horizontal labels of dataframe
        tempImag = np.array(df.index) #vertical labels of dataframe
        imag = np.reshape(tempImag, (len(tempImag)))
        with open(filename, 'w') as f:
            f.write(str(rl[0]) + '\n' + str(rl[len(rl)-1]) + '\n' + str(len(rl)) + '\n'
                    + str(imag[0]) + '\n' + str(imag[len(imag)-1]) + '\n' + str(len(imag)) + '\n') #write min,max,len attributes
            f.write(str(self.c) + '\n')     #write c attribute
            
    def fromCSV(self, filename):
        """Reads the arguments for the complex plane
        from a csv file called filename, then regenerates
        the pane using those parameters.
        
        Args:
            filename (string) : name of the file to 
                read complex plane from
        """
        
        with open(filename, 'a') as f:
            df.to_csv(filename, mode='a')  #append dataframe content to end of file
 
        with open(filename, 'r') as f:
            self.xmin = float(f.readline()) #read in each of the attributes
            self.xmax = float(f.readline())
            self.xlen = int(f.readline())
            self.ymin = float(f.readline())
            self.ymax = float(f.readline())
            self.ylen = int(f.readline())
            self.c = complex(f.readline())
        self.set_f(self.c)   # sets f and refreshes the plane to the values in the .csv file
    
    def toJSON(self, filename):
        """Saves the complex plane, self.plane
        as a JSON file named filename.
        
        Args:
            filename (string) : name of the file to 
                write complex plane to
        """
        
        self.plane.to_json(filename)
    
    def fromJSON(self, filename):
        """Sets self.plane as the array of values from
        a JSON file named filename
        
        Args:
            filename (string) : name of the file to 
                read complex plane from
        """
        
        df = pd.read_json(filename)
        self.plane = df
        
        
@nb.vectorize([nb.int32(nb.complex128)])
def julia(c, max=100):
    def f(z):
        n = 1
        mag = abs(z)
        if mag > 2:
            return 1  #return 1 if |z|>2 before transformation
        while n <= max: #continue transformation until max is exceeded
            z = z**2 + c
            mag = abs(z)
            if mag > 2:
                return n #return number of transformations before |z|>2
            else:
                n += 1
        return 0  #return 0 if max is reached before |z|>2
    return f
    pass
