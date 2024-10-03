import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import sympy as sp

class Portico:
    def __init__(self, incidencia, coord, restr, no_no, no_sec, g_sec, sec_el, no_mat, mater, mat_el, vinc, no_el, i_rigido, a_rigido, distv, disth, distgb, pont, dloc):
        # Inicializando variáveis importantes
        self.incidencia = incidencia
        self.coord = coord
        self.restr = restr
        self.no_no = no_no
        self.no_sec = no_sec
        self.g_sec = g_sec
        self.sec_el = sec_el
        self.no_mat = no_mat
        self.mater = mater
        self.mat_el = mat_el
        self.vinc = vinc
        self.no_el = no_el
        self.i_rigido = i_rigido
        self.a_rigido = a_rigido
        self.distv = distv
        self.disth = disth
        self.distgb = distgb
        self.pont = pont
        self.dloc = dloc
        # Inicializar listas para armazenar dados intermediários
        self.Te = []
        self.Le = []
        self.Lb = []
        self.Xe = []
        self.Dx = []
        self.secoes = []
        self.Sec = []
        self.Mat = []
        self.Idgl = None
        self.gl_fr = 0
        self.no_gl = 0
        self.Ke = []
        self.Ke_nr = []
        self.Kg = None
        self.pontual = None
        self.distnod = None
        self.desloc = None
        self.dk = None
        self.Du = None
        self.Qu = None
        self.l_De = []
        self.l_Qi = []
        self.l_dist_nod = []
        
    def rotacao(self):
        for i in self.incidencia:
            xi = self.coord[i[0]][0]
            yi = self.coord[i[0]][1]
            xf = self.coord[i[1]][0]
            yf = self.coord[i[1]][1]
            
            dx = xf - xi
            dy = yf - yi
            L = np.sqrt(dx**2 + dy**2)
            Lbx = dx / L
            Lby = dy / L
            
            # Matriz de rotação ou incidência cinemática
            T = np.matrix([
                [Lbx, Lby, 0., 0., 0., 0.],
                [-Lby, Lbx, 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., Lbx, Lby, 0.],
                [0., 0., 0., -Lby, Lbx, 0.],
                [0., 0., 0., 0., 0., 1.]
            ])
            self.Te.append(T)
            self.Le.append(L)
            self.Lb.append([Lbx, Lby])
            self.Xe.append([xi, yi, xf, yf])
            self.Dx.append([dx, dy])

    def Id_gl(self):
        self.Idgl = np.zeros((self.no_no, 3), dtype=int)
        cont = 0
        
        for i in range(self.no_no):
            for j in range(3):
                if self.restr[i][j] == 0:
                    self.Idgl[i][j] = cont
                    cont += 1
        
        self.gl_fr = cont

    def GeomSecao(self):
        for i in range(self.no_sec):
            if self.g_sec[i][0] == 'generico':
                A = self.g_sec[i][2]
                I = self.g_sec[i][3]
            elif self.g_sec[i][0] == 'retangular':
                A = self.g_sec[i][2] * self.g_sec[i][3]
                I = self.g_sec[i][2] * self.g_sec[i][3]**3 / 12
            elif self.g_sec[i][0] == 'circular':
                A = np.pi * self.g_sec[i][2]**2 / 4
                I = np.pi * self.g_sec[i][2]**4 / 64
            
            self.secoes.append([A, I])
        
        for i in range(len(self.sec_el)):
            sec_pivo = next((self.secoes[j] for j in range(self.no_sec) if self.sec_el[i] == self.g_sec[j][1]), None)
            if sec_pivo is not None:
                self.Sec.append(sec_pivo)

    def material(self):
        self.Mat = []
        material = []
        
        for i in range(self.no_mat):
            E = self.mater[i][1]
            material.append([E])
        
        for i in range(self.no_el):
            mat_pivo = None  
            for j in range(self.no_mat):
                if self.mat_el[i] == self.mater[j][0]:
                    mat_pivo = material[j]
                    break 
            if mat_pivo is not None:
                self.Mat.append(mat_pivo)
           
    # Método para calcular o número de graus de liberdade
    def calcular_gl(self):
        self.no_gl = sum(1 for i in range(self.no_no) for j in range(3) if self.restr[i][j] == 0)

    # Matriz de Rigidez
    def Rigidez(self):
        self.Ke = []
        self.Ke_nr = []
        for i in range(self.no_el):
            V1 = self.vinc[i][0]
            V2 = self.vinc[i][1]
            A = self.Sec[i][0]
            I = self.Sec[i][1]
            E = self.Mat[i][0]
            L = self.Le[i]
            Ir = 1
            Ar = 1
            
            if self.i_rigido[i] == 'ir':
                Ir = 1e10
            if self.a_rigido[i] == 'ar':
                Ar = 1e10
            # Elemento que tem ambos os extremos como elemento
            if V1 == 'e' and V2 == 'e':
                ke1 = np.matrix([
                    [ E*A/L*Ar,              0.,             0., -E*A/L*Ar,              0.,            0.],
                    [       0.,  (12.*E*I/L**3),  (6.*E*I/L**2),        0., (-12.*E*I/L**3), (6.*E*I/L**2)],
                    [       0.,   (6.*E*I/L**2),     (4.*E*I/L),        0.,  (-6.*E*I/L**2),    (2.*E*I/L)],
                    [-E*A/L*Ar,              0.,             0.,  E*A/L*Ar,              0.,            0.],
                    [       0., (-12.*E*I/L**3), (-6.*E*I/L**2),        0.,  (12.*E*I/L**3),(-6.*E*I/L**2)],
                    [       0.,   (6.*E*I/L**2),     (2.*E*I/L),        0.,  (-6.*E*I/L**2),    (4.*E*I/L)],
                ]) * Ir
            # Elemento onde um extremo é um elemento e o outro extremo é rígido
            if V1 == 'e' and V2 == 'r':
                ke1 = np.matrix([
                    [ E*A/L*Ar,              0.,             0., -E*A/L*Ar,              0.,            0.],
                    [       0.,   (3.*E*I/L**3),  (3.*E*I/L**2),        0.,  (-3.*E*I/L**3),            0.],
                    [       0.,   (3.*E*I/L**2),     (3.*E*I/L),        0.,  (-3.*E*I/L**2),            0.],
                    [-E*A/L*Ar,              0.,             0.,  E*A/L*Ar,              0.,            0.],
                    [       0.,  (-3.*E*I/L**3), (-3.*E*I/L**2),        0.,   (3.*E*I/L**3),            0.],
                    [       0.,              0.,             0.,        0.,              0.,        1.e-30],
                ]) * Ir
            # Elemento onde um extremo é rígido e o outro extremo é elemento
            if V1 == 'r' and V2 == 'e':
                ke1 = np.matrix([
                    [ E*A/L*Ar,              0.,             0., -E*A/L*Ar,              0.,            0.],
                    [       0.,   (3.*E*I/L**3),             0.,        0.,  (-3.*E*I/L**3), (3.*E*I/L**2)],
                    [       0.,   (3.*E*I/L**2),     (3.*E*I/L),        0.,  (-3.*E*I/L**2),            0.],
                    [-E*A/L*Ar,              0.,             0.,  E*A/L*Ar,              0.,            0.],
                    [       0.,  (-3.*E*I/L**3),             0.,        0.,   (3.*E*I/L**3),(-3.*E*I/L**3)],
                    [       0.,   (3.*E*I/L**3),             0.,        0.,  (-3.*E*I/L**3), (3.*E*I/L**2)],
                ]) * Ir
                
            # Elemento que tem ambos os extremos como rígido
            if V1 == 'r' and V2 == 'r':
                ke1 = np.matrix([
                    [ E*A/L*Ar,              0.,             0.,      -E*A/L*Ar,              0.,            0.],
                    [       0.,          1.e-30,             0.,             0.,              0.,            0.],
                    [       0.,              0.,         1.e-30,             0.,              0.,            0.],
                    [-E*A/L*Ar,              0.,             0.,       E*A/L*Ar,              0.,            0.],
                    [       0.,              0.,             0.,             0.,          1.e-30,            0.],
                    [       0.,              0.,             0.,             0.,              0.,        1.e-30],
                ]) * Ir
                
            # Matriz de rigidez elementar rotacional
            self.Ke.append(self.Te[i].T * ke1 * self.Te[i])
            self.Ke_nr.append(ke1) 
                
        # Matriz de rigidez global
        self.Kg = np.zeros((self.no_gl, self.no_gl))
        cont = 0
        for i in self.incidencia:
            pivo = [self.Idgl[i[0]][0],
                    self.Idgl[i[0]][1],
                    self.Idgl[i[0]][2],
                    self.Idgl[i[1]][0],
                    self.Idgl[i[1]][1],
                    self.Idgl[i[1]][2]]
            # Acumular os valores elementares para matriz global
            lin_e = 0
            for lin in pivo:
                col_e = 0
                for col in pivo:
                    self.Kg[lin, col] += self.Ke[cont][lin_e, col_e]
                    col_e += 1
                lin_e += 1
            cont += 1
        
#-------------------Carregamento nodal-------------------
    def Nodal(self):
        '''# Inicializa a lista para armazenar cargas nos nós
        self.pontual = [0] * self.no_no  # Inicializa com zero
        cont = 0

        # Itera sobre os índices dos nós
        for i in range(len(self.pont)):
            # Verifica se o índice cont está dentro do alcance da lista self.pont
            if cont < len(self.pont):
                self.pontual[self.pont[cont][0]] = self.pont[cont][1]  # Atualiza a carga no nó correto
                cont += 1  # Incrementa cont após a atribuição'''
    
    def Nodal(self):
        self.pontual = np.zeros(self.no_gl)
        cont = 0
        for i in self.Idgl:
            self.pontual[i[0]] = self.pont[cont][0]
            self.pontual[i[1]] = self.pont[cont][1]
            self.pontual[i[2]] = self.pont[cont][2]
            cont += 1
            
    def Distribuido(self):
        self.l.dist_nod = []
        for i in range(self.no_el):
            cosseno = self.Lb[i][0]
            seno = self.Lb[i][1]
            V1 = self.vin[i][0]
            V2 = self.vin[i][1]
            I = self.Sec[i][1]
            E = self.Mat[i][0]
            L = self.Le[i]
            if self.distgb[i] == 'global':
                q1 = self.distv[i][0]*cosseno - self.disth[i][0]*seno
                q1 = self.distv[i][1]*cosseno - self.disth[i][1]*seno  
                n1 = self.disth[i][0]*cosseno + self.distv[i][0]*seno 
                n2 = self.disth[i][1]*cosseno + self.distv[i][1]*seno
            else:
                q1 = self.distv[i][0]
                q2 = self.distv[i][1]
                n1 = self.disth[i][0]
                n2 = self.disth[i][1]
            Dq = q2 - q1
            Dn = n2 - n1
            F1 =L*(Dn/6 + n1/2)
            F4 =L*(Dn/3 + n1/2)

            if V1 == 'e' and V2 == 'e':
                d10m = -L**4*(Dq/30 + q1/8)/(E*I)
                d20m = L**3*(Dq + 4*q1)/(24*E*I)
                f11m = L**3/(3*E*I)
                f22m = L/(E*I)
                f12m = -L**2/(2*E*I)
                f11 = f11m
                f22 = f22m
                f12 = f12m
                d10 = d10m
                d20 = d20m
                K = np.array([[f11,f12], [f12,f22]])
                F = np.array([-d10, -d20])
                R = la.solve(K, F)
                F2 = R[0]
                F3 = R[1]
                F5 = -(-Dq*L/2 - L*q1 +F2)  #convenção troca sinal F5
                F6 = -Dq*L**2/6 - L**2*q1/2 + F2*L - F3
                
            if V1 =='r' and V2 == 'e':
                d10m = -L**4*(Dq/30 + q1/8)/(E*I)
                f11m = L**3/(3*E*I)
                f11 = f11m
                d10 = d10m
                F2 = -d10/f11
                F3 = 0
                F5 = -(-Dq*L/2 - L*q1 +F2)
                F6 = -Dq*L**2/6 - L**2*q1/2 + F2*L
                
            if V1 =='e' and V2 == 'r':
                d10m = L**4*(Dq/30 + q2/8)/(E*I)
                f11m = L**3/(3*E*I)
                f11 = f11m
                d10 = d10m
                F5 = -d10/f11
                F6 = 0
                F2 = -Dq*L/2 + L*q2 - F5
                F3 = -(Dq*L**2/6 - L**2*q2/2 + F5*L) #convenção troca sinal F3
                
            if V1 =='r' and V2 == 'r':
                F2 = q1*L/2 + Dq*L/6
                F3 = 0
                F5 = q1*L/2 + Dq*L/3
                F5 = 0
                
#-----------------cargas nodais devido o car, distribuido na sequenecia da Idgl------
            dist_el = np.array([F1, F2, F3, F4, F5, F6])
            dist_nod = la.solve(self.Te[i], dist_el)
            self.l_dist_nod.append(dist_nod)
        
        self.distnod = np.zeros(self.no_gl)
        cont = 0
        for i in self.incidencia:
            self.distnod[self.Idgl[[0][0]]] += self.l_dist_nod[cont][0]
            self.distnod[self.Idgl[[0][1]]] += self.l_dist_nod[cont][1]
            self.distnod[self.Idgl[[0][2]]] += self.l_dist_nod[cont][2]
            self.distnod[self.Idgl[[1][0]]] += self.l_dist_nod[cont][3]
            self.distnod[self.Idgl[[1][1]]] += self.l_dist_nod[cont][4]
            self.distnod[self.Idgl[[1][2]]] += self.l_dist_nod[cont][5]
            cont += 1
    
    def CalcCarg(self):
        self.Nodal()
        self.Distribuido()
        self.Carregamento()
    
    def Carregamento(self):
        self.carga = np.zeros((self.no_gl, 1))
        for i in range(self.no_gl):
            self.carga[i] += self.distnod[i] + self.pontual[i]
        self.DeslocGlobal()
        
        
#-------------------Deslocamento prescitos nos graus de liberdade restringido -----
    def DlocPrescrito(self):
        self.desloc = np.zeros((self.no_gl, 1))
        cont = 0
        for i in range(self.no_no):
            if self.restr[i][0]==1:
                self.desloc[self.Idgl[i][0]] = self.dloc[i][0]
                cont += 1
            if self.restr[i][1]==1:
                self.desloc[self.Idgl[i][1]] = self.dloc[i][1]
                cont += 1   
            if self.restr[i][2]==1:
                self.desloc[self.Idgl[i][2]] = self.dloc[i][2]
                cont += 1 

#-------------------Deslocamentos globais nos graus de liberdade livre ----------

    def DeslocGlobal(self):
        K12 = self.Kg[0:self.gl_fr, self.gl_fr:self.no_gl]  
        K11 = self.Kg[0:self.gl_fr, 0:self.no_gl]
        qk = self.carga[self.gl_fr]
        self.dk = self.desloc[self.gl_fr: self.no_gl]
        if K11 != []:
            self.Du = la.solve(K11, qk - np.dot(K12,self.dk))
        else:
            self.Du = np.dot(K12, self.dk)
        cont = 0
        for i in range(self.no_no):
            if self.restr[i][0] == 0:
                self.desloc[self.Idgl[i][0]] = self.Du[cont]
                cont += 1  
            if self.restr[i][1] == 0:
                self.desloc[self.Idgl[i][1]] = self.Du[cont]
                cont += 1 
            if self.restr[i][2] == 0:
                self.desloc[self.Idgl[i][2]] = self.Du[cont]
                cont += 1    
           
        self.RecApoio()
        self.EsfInterno()
        
        
#------------------Reação de apoio-------------

    def RecApoio(self):
        K21 = self.Kg[self.gl_fr: self.no_gl, 0: self.gl_fr]
        K22 = self.Kg[self.gl_fr: self.no_gl, self.gl_fr: self.no_gl]
        self.Qu = np.dot(K21, self.Du) + np.dot(K22, self.dk)
        
        pont_apoio = self.carga[self.gl_fr: self.no_gl]
        for i in range((self.no_gl-self.gl_fr)):
            self.Qu[i] += - pont_apoio[i]
            
    
#---------------------Esforço interno-------------------

    def EsfInter(self):
        cont = 0
        self.l_De = []
        self.l_Qi = []
        for i in self.incidencia:
            Dix = self.desloc[self.Idgl[i[0][0]]]
            Diy = self.desloc[self.Idgl[i[0][1]]]
            Dim = self.desloc[self.Idgl[i[0][2]]]
            Dfx = self.desloc[self.Idgl[i[1][0]]]
            Dfy = self.desloc[self.Idgl[i[1][1]]]
            Dfm = self.desloc[self.Idgl[i[1][2]]]
            
            De = np.matrix([Dix, Diy, Dim, Dfx, Dfy, Dfm])
            self.l_De.append(De)
            Fi = -1*np.matrix(self.l_dis_nod[cont]).T
            Qi = self.Te[cont]*self.Ke[cont]*De +self.Te[cont]*Fi
            self.l_Qi.append(Qi)
            cont += 1
        self.Equacoes()
        
#-----------------Equações para plotagem dos diagramas --------
    def Equacoes(self):
        cont = 0
        self.Eq = []
        for i in self.incidencia:
            A = self.Sec[cont][0]
            E = self.Mat[cont][0]
            Ir = 1.
            if self.i_rigido[cont] == 'ir':
                ir = 1e10
            Ar = 1
            if self.a_rigido[cont] == 'Ar':
                Ar = 1e10
            cosseno = self.Lb[cont][0]
            seno = self.Lb[cont][1]
            if self.distgb[cont] == 'global':
                q1 = self.distv[cont][0]*cosseno - self.disth[cont][0]*seno
                q2 = self.distv[cont][1]*cosseno - self.disth[cont][1]*seno
                n1 = self.distv[cont][0]*cosseno - self.disth[cont][0]*seno
                n2 = self.distv[cont][1]*cosseno - self.disth[cont][1]*seno
            else:
                q1 = self.distv[cont][0]
                q2 = self.distv[cont][1]
                n1 = self.distv[cont][0]
                n2 = self.distv[cont][1]
            Dn = (-n2)-(-n1)
            Dq = (-q2)-(-q1)
            Ni = self.l_Qi[cont][0]
            Vi = self.l_Qi[cont][1]
            Mi = self.l_Qi[cont][2]
            Di = self.desloc[self.Id_gl[i[0][0]]]*self.Lb[cont][0]-self.desloc[self.Idgl[i[0][0]]]*self.Lb[cont][1]
            Dix = self.desloc[self.Id_gl[i[0][0]]]*self.Lb[cont][1]+self.desloc[self.Idgl[i[0][0]]]*self.Lb[cont][0]
            Df = self.desloc[self.Id_gl[i[1][1]]]*self.Lb[cont][0]-self.desloc[self.Idgl[i[1][0]]]*self.Lb[cont][1]
            C = (((Df/self.Le[cont])-(Di/self.Le[cont]))+((Dq*(self.Le[cont])**3)/120 +((-q1)*(self.Le[cont])**3)/24 - (Vi*(self.Le[cont])**2)/6 + (Mi*(self.Le[cont]))/2)/(self.Mat[cont][0]*self.Sec[cont][1]*Ir))
            x = sp.symbols('x')
            q = (Dq/self.Le[cont])*x+(-q1)
            n = (Dn/self.Le[cont])*x+(-n1)
            N = sp.integrate(n, x) - Ni
            # Equação dos deslocamentos horizontal no sistema local
            Dx = sp.integrate(N,x)/(self.Mat[cont][0]*self.Sec[cont][0]*Ir*Ar) + Dix
            V = -sp.integrate(q,x) + Vi
            M = sp.integrate(V,x) - Mi 
            # Equação do deslocamento local no sistema local
            Dy = sp.integrate(sp.integrate(M,x),x)/(self.Mat[cont][0]*self.Sec[cont][1]*Ir) + C*x + Di
            R = sp.diff(Dy, x)
            Dgx = Dx.self.Lb[cont][0] - Dy*self.Lb[cont][1]
            Dgy = Dx.self.Lb[cont][1] + Dy*self.Lb[cont][0]
            self.Eq.append([q, n , N, V, M, R, Dx, Dy, Dgx, Dgy])
            cont += 1
        self.Outpout()
        








# Definindo os parâmetros do pórtico
incidencia = [(0, 1), (1, 2), (1, 3)]
coord = [(0, 0), (4, 0), (4, 3), (0, 3)]  # Coordenadas dos nós
restr = [[1, 1, 1],  # Nó 0 restrito (fixo)
         [0, 0, 0],  # Nó 1 livre
         [0, 0, 0],  # Nó 2 livre
         [1, 1, 1]]  # Nó 3 restrito (fixo)

no_no = len(coord)  # Número de nós
no_sec = 1  # Número de seções
g_sec = [['retangular', 'Viga', 0.2, 0.4]]  # Seção retangular
sec_el = ['Viga']  # Tipo da seção para cada elemento
no_mat = 1  # Número de materiais
mater = [['Aço', 200e9]]  # Nome e módulo de elasticidade do material (Aço)
mat_el = ['Aço']  # Material associado a cada elemento
vinc = [['e', 'e'], ['e', 'e'], ['e', 'e']]  # Tipo de conexão (elemento a elemento)
no_el = len(incidencia)  # Número de elementos
i_rigido = ['r', 'r', 'r']  # Tipo de rigidez (rigido)
a_rigido = ['r', 'r', 'r']  # Tipo de rigidez (rigido)
distv = [[0, 0], [0, 0], [0, 0]]  # Deslocamento vertical
disth = [[0, 0], [0, 0], [0, 0]]  # Deslocamento horizontal
distgb = ['global', 'global', 'global']  # Sistema de coordenadas para carregamento
pont = [[1, -10], [2, -15]]  # Cargas pontuais nos nós
dloc = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Deslocamentos prescritos

# Instanciando o pórtico
portico = Portico(incidencia, coord, restr, no_no, no_sec, g_sec, sec_el, no_mat, mater, mat_el, vinc, no_el, i_rigido, a_rigido, distv, disth, distgb, pont, dloc)

# Executando os métodos principais
portico.rotacao()  # Calcula a matriz de rotação
portico.Id_gl()  # Calcula os graus de liberdade
portico.GeomSecao()  # Calcula a geometria das seções
portico.material()  # Define os materiais
portico.calcular_gl()  # Calcula o número de graus de liberdade
portico.Rigidez()  # Monta a matriz de rigidez global
portico.CalcCarg()  # Calcula as cargas nodais

# Exibindo os resultados
print("Matriz de Rigidez Global (Kg):")
print(portico.Kg)
print("\nDeslocamentos nos nós (desloc):")
print(portico.desloc)