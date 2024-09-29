import numpy as np
import matplotlib.pyplot as plt

class Portico:
    def __init__(self, incidencia, coord, no_no, restr, no_sec, g_sec, sec_el, no_mat, mater, no_el, mat_el, vinc, i_rigido, a_rigido):
        self.incidencia = incidencia
        self.coord = coord
        self.no_no = no_no
        self.restr = restr
        self.no_sec = no_sec
        self.g_sec = g_sec
        self.sec_el = sec_el
        self.no_mat = no_mat
        self.mater = mater
        self.no_el = no_el
        self.mat_el = mat_el
        self.vinc = vinc
        self.i_rigido = i_rigido
        self.a_rigido = a_rigido
        
        # Inicialização das variáveis
        self.Te = []
        self.Le = []
        self.Lb = []
        self.Xe = []
        self.Dx = []
        self.Idgl = None
        self.gl_fr = 0
        self.Sec = []
        self.secoes = []
        self.Mat = []
        self.Ke = []
        self.Ke_nr = []
        self.no_gl = 0  # Inicializa o número de graus de liberdade

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




if __name__ == "__main__":
    incidencia = [[0, 1], [1, 2], [2, 3]]
    coord = [[0, 0], [0, 5], [5, 5], [5, 0]]
    no_no = 4
    restr = [[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0]]  
    no_sec = 1
    g_sec = [['retangular', 0, 0.1, 0.2]]
    sec_el = [0, 0, 0]
    no_mat = 1
    mater = [['A', 210e6]]
    no_el = 3
    mat_el = ['A', 'A', 'A']
    vinc = [['r', 'e'], ['e', 'e'], ['e', 'r']]
    i_rigido = ['ir', 'ir', 'ir']
    a_rigido = ['ar', 'ar', 'ar']

    portico = Portico(incidencia, coord, no_no, restr, no_sec, g_sec, sec_el, no_mat, mater, no_el, mat_el, vinc, i_rigido, a_rigido)
    portico.rotacao()
    portico.Id_gl()
    portico.GeomSecao()
    portico.material()
    portico.calcular_gl()  # Chama o novo método
    portico.Rigidez()
    np.set_printoptions(precision=3)
    print("Matriz de Rigidez Global:", portico.Kg)


#--------------visualização do pórtico-----------------
# Criar figura
plt.figure()

# Desenhar nós
for i, (x, y) in enumerate(coord):
    plt.plot(x, y, 'ko')  # nó em preto
    plt.text(x, y, f'N{i}', fontsize=12, ha='right')

# Desenhar elementos
for el in incidencia:
    x_values = [coord[el[0]][0], coord[el[1]][0]]
    y_values = [coord[el[0]][1], coord[el[1]][1]]
    plt.plot(x_values, y_values, 'b-')  # elemento em azul

# Configurações do gráfico
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Portico')
plt.grid(True)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.gca().set_aspect('equal', adjustable='box')

# Mostrar o desenho
plt.show()