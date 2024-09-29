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

def Id_gl(self):
    self.Idgl = np.zeros(self.no_gl).reshape(self.no_no, 3)  #Matriz de zeros
    cont = 0
    for i in range(self.no_no):    
        if self.restr[i][0] == 0:
            self.idgl[i][0] = cont
            cont += 1
        if self.restr[i][1] == 0:
            self.idgl[i][1] = cont
            cont += 1
        if self.restr[i][2] == 0:
            self.idgl[i][2] = cont
            cont += 1
    self.gl_livre = cont
    for i in range(self.no_no):    
        if self.restr[i][0] == 0:
            self.idgl[i][0] = cont
            cont += 1
        if self.restr[i][1] == 0:
            self.idgl[i][1] = cont
            cont += 1
        if self.restr[i][2] == 0:
            self.idgl[i][2] = cont
            cont += 1
    self.Idgl = self.Idgl.astype(int)

def GeomSecao(self):
    self.Sec = []
    self.secoes = []
    
    for i in range(self.no_Sec):
        if self.g_sec[i][0] == 'generico':
            A = self.g_sec[i][2]
            I = self.g_sec[i][3]
       
        if self.g_sec[i][0] == 'retangular':
            A = self.g_sec[i][2]*self.g_sec[i][3]
            I = self.g_sec[i][2]*self.g_sec[i][3]**3/12
            
        if self.g_sec[i][0] == 'circular':
            A = np.pi*self.g_sec[i][2]**2/4
            I = np.pi*self.g_sec[i][2]**4/64
        self.secoes.append([A, I])
    for i in range(self.no_el):
        sec_pivo = None
        for j in range(self.no_sec):
            if self.sec_el[i] == self.g_sec[j][1]:
                sec_pivo = self.secoes[j]
                break
            #else: continue
        if sec_pivo is not None: 
            self.Sec.append(sec_pivo)
        #self.Sec.append(sec_pivo)
                
        
