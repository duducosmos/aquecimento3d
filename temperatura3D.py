#!/usr/bin/env python
#-*- coding: UTF-8 -*-

"""
Calculo da temperatura superficial no imageador do sensor de estrelas
autonomos ao ser exposto ao sol

"""

from numpy.numarray import zeros,ones,Float64
from numpy import array,arange,log10,meshgrid
import numpy
import multiprocessing as mp 

import  scipy.io as io

from math import sqrt 
#from __future__ import division

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, LogFormatter


from matplotlib.patches import Patch, Ellipse
from pylab import savefig,pcolor

import random 

import sys,os


def progress_bar(value, max, barsize):
    """
    Modulo que implementa barra de progressos em modo texto:
Exemplo de uso:

>>print "processing..."
>>for i in xrange(11):
   progress_bar(i, 10, 40)
   time.sleep(1)
>>print "ok"
>>raw_input()
    """
    chars = int(value * barsize / float(max))
    percent = int((value / float(max)) * 100)
    sys.stdout.write("#" * chars)
    sys.stdout.write(" " * (barsize - chars + 2))
    if value >= max:
        sys.stdout.write("Feito \n\n")
    else:
        sys.stdout.write("[%3i%%]\r" % (percent))
        sys.stdout.flush()

class SensorTemp:
    
    def __init__(self,x0,xf,y0,yf,z0,zf,tf,tau,alpha):
        
        self.x0 = x0
        self.xf = xf
        self.y0 = y0
        self.yf = yf
        self.z0 = z0
        self.zf = zf
        self.tf = tf
        self.amb = 25.0
        self.RSolI = 0.31/2.0 #mm
        
        self.nt = int(5e5)
        self.deltaT = tf/float(self.nt)
        self.rhoSi = 2.3e-3 #g/mm^{3}
        self.Csi = 0.7 # J /(g C) Calor especifico
        self.ksi = 130e-3 # W/ (mm - C) Condutividade Termica
        self.C0 = self.ksi/self.rhoSi*self.Csi
        

        self.Delta1 = 8.0*self.C0*self.deltaT

        self.ny =  30
        self.nx = int(self.ny*(xf-x0)/(yf-y0))
        self.nz = int(self.ny*(zf-z0)/(yf-y0))
        self.Delta = (yf-y0)/float(self.ny)
        DeltaXYZ2 = self.Delta**2.0
        print numpy.pi*(self.RSolI)**2.0,DeltaXYZ2,self.Delta,self.RSolI*2.0
        raw_input('Press Enter to Start ...')
        if(self.Delta1 <= DeltaXYZ2):
            pass
        else:
            print """
            4.0*k*Delta t = %3.5e, Delta**2 = %3.5e 
            """ %(self.Delta1,DeltaXYZ2)
            print " 4.0*k*Delta t tem que ser menor que o Delta**2"
            sys.exit()

        #self.alpha =  self.C0*self.deltaT/(self.Delta**2.0)
        #self.nx = int((xf-x0)/self.Delta)
        #self.ny = int((yf-y0)/self.Delta)
        #print self.nx,self.ny
        #raw_input('enter')

        self.T = 25.0*ones((self.nx,self.ny,self.nz),type=Float64)
        #self.T[4:7,4:7] = 60.0 
        self.T0 = 25.0*ones((self.nx,self.ny,self.nz),type=Float64) 
        
        self.dTdx_B = 1.0
        self.NConst = 1.0
        

        ic = int((xf-x0)/(2.0*self.Delta))
        jc = int((yf-y0)/(2.0*self.Delta))

        self.CSol = [ic,jc]
        
        self.tau = tau
        self.alpha = alpha
        self.P =  0.74 #W
        self.E = 9.81 #W/mm^{2}
        self.crand = None
        self.trand = 0.0




    def Dx(self,i,j,k):
        return self.T[i+1,j,k]-2.0*self.T[i,j,k]+self.T[i-1,j,k] 

    def Dy(self,i,j,k):
        return self.T[i,j+1,k]-2.0*self.T[i,j,k]+self.T[i,j-1,k]

    def Dz(self,i,j,k):
        return self.T[i,j,k+1]-2.0*self.T[i,j,k]+self.T[i,j,k-1] 
    
    def icenter(self,j):
        '''
        Define a posicao i a partir do centro da imagem formada pelo sol no
        sensor
        '''	
        if( j == self.CSol[1]):
            if(self.Delta >= self.RSolI):
                i1 = i2 = self.CSol[0]
            else:
                i1 = self.CSol[0] - (self.RSolI/self.Delta)
                i2 = self.CSol[0] + (self.RSolI/self.Delta)           
            return (int(i1),int(i2))
        j=float(j)

        cx = float(self.CSol[0])
        cy = float(self.CSol[1])
        
        C0 = cx**2.0 + j**2.0 - 2.0*j*cy + cy**2.0 - (self.RSolI/self.Delta)**2.0
        DeltaB = 4.0*cx**2.0-4.0*C0
        if(DeltaB < 0.0): return (0,0)
        i2 = (2.0*cx+sqrt(DeltaB))/2.0
        i1 = (2.0*cx-sqrt(DeltaB))/2.0
        
        return (int(i1),int(i2))

    def Q(self,t,i,j):

        ic,jc = self.CSol
        idif = i - ic
        jdif = j - jc
        if( 0 <= idif <= 1):
            if( 0 <= jdif <= 1):
                qsol = self.tau*self.alpha*(self.P/4.0)*self.C0/self.ksi
                return qsol
            else:
                return 0.0
        else:
            return 0.0
            
        #i1,i2 = self.icenter(j)    
        #if( i >= i1 and i<= i2):
            #if(i1 == i2):
                #qsol = self.tau*self.alpha*self.P*self.C0/self.ksi
            #else:
                #qsol = self.tau*self.alpha*self.E\
                       #*(self.Delta**2.0)*self.C0/self.ksi

            #return qsol

        #else:
            #return 0.0

    def Temperatura(self,i,j,k,t):
        gamma = self.C0*(self.deltaT/self.Delta**2.0)
        
        if(k == self.nz-1):
            deltasxyz = self.Dx(i,j,k) + self.Dy(i,j,k)                        
            myT = self.T[i,j,k]+gamma*deltasxyz\
                   + self.deltaT*self.Q(t,i,j) \
                   + gamma*(2.0*self.T[i,j,k-1]-2.0*self.T[i,j,k])\
                  

        elif(k == 0):
            deltasxyz = self.Dx(i,j,k) + self.Dy(i,j,k)
            myT = self.T[i,j,k] + gamma*deltasxyz \
                   + gamma*(2.0*self.T[i,j,k+1]-2.0*self.T[i,j,k])
        else:
            deltasxyz = self.Dx(i,j,k) + self.Dy(i,j,k) +\
                        self.Dz(i,j,k)
            #if( deltasxy != 0.0):print deltasxy,gamma,self.deltaT,self.Delta**2.0
            myT = self.T[i,j,k]+gamma*deltasxyz

        return myT    
    

    def bordas(self):
        gamma = self.C0*(self.deltaT/self.Delta**2.0)

        T0 = self.T[0,:,:]+gamma*(2.0*self.T[1,:,:]-2.0*self.T[0,:,:])
        self.T[0,:,:]=T0
        
        Tnx = self.T[self.nx-1,:,:]+gamma*(2.0*self.T[self.nx-2,:,:]-2.0*self.T[self.nx-1,:,:])
        self.T[self.nx-1,:,:]=Tnx
        
        T1 = self.T[:,0,:]+gamma*(2.0*self.T[:,1,:]-2.0*self.T[:,0,:])
        self.T[:,0,:] = T1
        
        Tny = self.T[:,self.ny-1,:]+gamma*(2.0*self.T[:,self.ny-2,:]-2.0*self.T[:,self.ny-1,:])
        self.T[:,self.ny-1,:]=Tny
        
        
    def TNew(self,t):
        self.bordas()
        

        Ttemp = array([\
                 [\
                 [self.Temperatura(i,j,k,t) for k in range(0,self.nz)]\
                                            for j in range(1,self.ny-1)] \
                                            for i in range(1,self.nx-1)]\
                                            )
        self.T[1:self.nx-1,1:self.ny-1,:] = Ttemp
        
        return self.T

    def myPlot3D(self,T,cont2,tmax,time):
        import mayavi.mlab as mlab
        import mayavi.scripts as script
        from mayavi.core.api import Engine
        from mayavi import api    

        mlab.options.backend = 'simple'
        X,Y,Z = numpy.mgrid[self.x0:self.xf:self.Delta,\
                            self.y0:self.yf:self.Delta,\
                            self.z0:self.zf:self.Delta]

        fig = mlab.figure(fgcolor=(0,0,0),\
                                  bgcolor=(1,1,1),\
                                  size=(1300, 700))

        fig.scene.disable_render = True
        mlab.view(azimuth=130.0,
                     elevation=70.0,
                     distance=10.0,\
                     focalpoint=(45,45,30))
        e = mlab.get_engine()                
        v = e.get_viewer(fig)
        p = mlab.points3d(X,Y,Z,T,\
                      opacity=1.0,\
                      scale_mode='vector',\
                      scale_factor=0.6,\
                      transparent=True,\
                      #mode = 'cube',\
                      #colormap="cool",\
                      figure=fig
                      #vmin=25.0,\
                      #vmax=50.0
                      )
        b = mlab.colorbar(title=r'Temperature Variation', \
                          orientation='vertical',\
                          label_fmt="%1.5f")
        mlab.title('Time = %3.5e seg.  \nT max =%3.5e Cel.' %(time,tmax),\
                   line_width=0.5,\
                    size=0.1)
        
        #mlab.show()

        if(cont2 < 10):
            mlab.savefig('./figuras3d/MT_0'+str(cont2)+'.png')
        else:
            mlab.savefig('./figuras3d/MT_'+str(cont2)+'.png')

        mlab.close()
        sys.exit(1)

    def myQueue(self,q,MT,cont2,tmax,T):
        q.put(self.myPlot3D(MT,cont2,tmax,T))
        


    def TTime(self, crand = 0):
        import time
        print """
        Iniciando o calculo do processo de difusao termica

        """
        
        
        cont = 0
        cont2 = 0
        TMax = open('./Dados3d/TmaxTime.dat','w')
        self.crand = crand
        p=None
        job = []
        #X = arange(self.x0,self.xf,self.Delta)
        #Y = arange(self.y0,self.yf,self.Delta)
        #Z = arange(self.z0,self.zf,self.Delta)
        #X,Y,Z=meshgrid(X,Y,Z)

                    
        #X,Y = meshgrid(X,Y)
        t = arange(0.0,self.tf+self.deltaT,self.deltaT)
        nt = len(t)

        for T in t:
            progress_bar(cont, nt, 40)
            
            if(self.crand == 0):
                pass
            else:
                if(not(cont % 60)):
                    ic = random.randint(1,self.nx-1)
                    jc = random.randint(1,self.nx-1)                
                    self.CSol = [ic,jc]
                    self.trand = T
                    
                                
            MT = self.TNew(T) #- self.T0
            tmax = numpy.max(MT)
            
            TMax.write('  %3.9e  %3.9e ' %(tmax,T))

            
            
            if( not(cont % 5000)): 
                if(cont2 < 10):
                    arq = open('./Dados3d/temperatura_0'+str(cont2)+'.mtx','w')
                else:
                    arq = open('./Dados3d/temperatura_'+str(cont2)+'.mtx','w')
                    
                for k in range(self.nz):
                    comentario='''
                Matrix contem o campo de variacao de temperatura do imageador do sensor
                de estrelas autonomo. O campo e gerado a partir da potencia de uma 
                imagem solar formada no sensor. Os dados sao para o instante de tempo 
                t = %3.9e e Camada k = %3.9e
                ''' %(T,k)
                    io.mmwrite(arq,MT[:,:,k],comment=comentario,field='real')
                
                arq.close()
                q = mp.Queue()
                p = mp.Process(target=self.myQueue, args=(q,MT,cont2,tmax,T))
                p.start()
                job.append(p)
                cont2 += 1
            cont+=1
                    
            

        while job:
            job.pop().join()
            
        #for tjob in job:
        #    tjob.start()
        #for tjob in job:
        #    tjob.join()
        #
        print "Gerando gif animado da simulacao"        
        os.system(\
         'convert -delay 60 -loop 0 ./figuras3d/MT_*.png ./figuras3d/animacao.gif')
        os.system(\
         'convert -delay 60 -loop 1 ./figuras3d/MT_*.png ./figuras3d/animacao2.gif')
        print "Trabalho finalizado"
        



if( __name__ == "__main__"):

    obj = SensorTemp(0.0,15.0,0.0,15.0,0.0,1.5,300,1.0,1.0)
    obj.TTime(crand = 0)
