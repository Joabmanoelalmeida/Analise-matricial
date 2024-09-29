import numpy as np

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