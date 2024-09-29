import numpy as np


def rotacao(self):
    self.Te = []
    self.Le = []
    self.Lb = []   
    self.Xe = []
    self.Dx = []
    for i in self.incidencia:
        xi = self.coord[i[0]][0]    #coordenada X do ponto inicial
        yi = self.coord[i[0]][1]    #coordenada Y do ponto inicial
        xf = self.coord[i[1]][0]    #coordenada X do ponto final
        yf = self.coord[i[1]][1]    #coordenada Y do ponto final
        dx = xf - xi 
        dy = yf - yi
        L = np.sqrt(dx**2 + dy**2)
        Lbx = dx/L   
        Lby = dy/L
#-----------------------Matriz de rotação ----------------------        
        T = np.matrix([[Lbx, Lby, 0., 0., 0., 0.],
                      [-Lby, Lbx, 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., Lbx, Lby, 0.],
                      [0., 0., 0., -Lby, Lbx, 0.],
                      [0., 0., 0, 0., 0., 1.]])
        self.Te.append(T) 
        self.Le.append(L) 
        self.Lb.append([Lbx, Lby]) 
        self.Xe.append([xi, yi, xf, yf]) 
        self.Dx.append([dx, dy]) 
        