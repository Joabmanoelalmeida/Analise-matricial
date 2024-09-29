import numpy as np

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
        
        
    
    