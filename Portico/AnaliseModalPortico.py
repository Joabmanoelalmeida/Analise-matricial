''' Autor _ JOAB MANOEL ALMEIDA SANTOS
    Disciplina: Mecânica Computacional das estruturas
    Professor: Dr. Eduardo Toledo
    Universidade Federal de Alagoas
    Mestrando em Estruturas e Materiais 
    Implementação para pórtico plano
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

class No:
    def __init__(self, id, x, y, restricoes, deslocamentos_prescritos):
        self.id = id
        self.x = x
        self.y = y
        self.restricoes = restricoes 
        self.deslocamentos_prescritos = deslocamentos_prescritos  

class Elemento:
    def __init__(self, id, no_inicial, no_final, E, A, I, qx, qy, poisson, densidade):
        self.id = id
        self.no_inicial = no_inicial
        self.no_final = no_final
        self.E = E
        self.A = A
        self.I = I
        self.qx = qx
        self.qy = qy
        self.poisson = poisson
        self.densidade = densidade  
        self.G = self.calcula_modulo_cisalhamento()  
        self.L = self.calcula_comprimento()
        self.cos, self.sen = self.calcula_inclinacao()

    def calcula_modulo_cisalhamento(self):
        return self.E / (2 * (1 + self.poisson))

    def calcula_comprimento(self):
        dx = self.no_final.x - self.no_inicial.x
        dy = self.no_final.y - self.no_inicial.y
        return (dx**2 + dy**2) ** 0.5

    def calcula_inclinacao(self):
        dx = self.no_final.x - self.no_inicial.x
        dy = self.no_final.y - self.no_inicial.y
        return dx / self.L, dy / self.L

    def matriz_rigidez_local(self):
        L = self.L
        E = self.E
        A = self.A
        I = self.I

        return np.array([
            [ E*A/L,              0.,             0., -E*A/L,              0.,            0.],
            [       0.,  (12.*E*I/L**3),  (6.*E*I/L**2),        0., (-12.*E*I/L**3), (6.*E*I/L**2)],
            [       0.,   (6.*E*I/L**2),     (4.*E*I/L),        0.,  (-6.*E*I/L**2),    (2.*E*I/L)],
            [-E*A/L,              0.,             0.,  E*A/L,              0.,            0.],
            [       0., (-12.*E*I/L**3), (-6.*E*I/L**2),        0.,  (12.*E*I/L**3),(-6.*E*I/L**2)],
            [       0.,   (6.*E*I/L**2),     (2.*E*I/L),        0.,  (-6.*E*I/L**2),    (4.*E*I/L)],
        ])

    def matriz_massa_local(self):
        L = self.L
        A = self.A
        densidade = self.densidade
        '''I = self.I 
        j1 = 312
        j2 = -44 * self.L
        j3 = 108
        j4 = 26 * self.L
        j5 = 8 * self.L ** 2
        j6 = -6 * self.L ** 2
        j7 = 36
        j8 = -3 * self.L
        j9 = 4 * self.L ** 2
        j10 = -self.L ** 2
        Tb = densidade*A*L/840
        Rb = densidade*I/(30*L)

        return np.array([
            [densidade * A * L / 3,    0,                     0,                     densidade * A * L / 6,    0,                     0],
            [0,                        (j1 * Tb + Rb * j7), (Tb * j2 + Rb * j8),    0,                        (Tb * j3 + Rb * j7), (Tb * j4 + Rb * j8)],
            [0,                        (Tb * j2 + Rb * j8), (Tb * j5 + Rb * j9),    0,                       -(Tb * j4 + Rb * j8), (Tb * j6 + Rb * j10)],
            [densidade * A * L / 6,    0,                     0,                     densidade * A * L / 3,    0,                     0],
            [0,                        (Tb * j3 + Rb * j7), -(Tb * j4 + Rb * j8),  0,                        (Tb * j1 + Rb * j7), -(Tb * j2 + Rb * j8)],
            [0,                        (Tb * j4 + Rb * j8), (Tb * j6 + Rb * j10),  0,                       -(Tb * j2 + Rb * j8), (Tb * j5 + Rb * j9)],
        ])'''
        return (densidade * A * L / 420)*np.array([
            [140,          0,             0,                70,              0,              0],
            [0,           150,           22*L,               0,              54,         -13*L],
            [0,           22*L,         4*L**2,              0,              13*L,     -3*L**2],
            [70,           0,              0,               140,              0,             0],
            [0,           54,             13*L,              0,              156,        -22*L],
            [0,          -13*L,         -3*L**2,             0,              -22*L,     4*L**2],
        ])

    def matriz_cinematica(self):
        Lbx, Lby = self.cos, self.sen

        return np.array([
            [ Lbx, Lby, 0., 0., 0., 0.],
            [-Lby, Lbx, 0., 0., 0., 0.],
            [  0.,  0., 1., 0., 0., 0.],
            [  0.,  0., 0., Lbx, Lby, 0.],
            [  0.,  0., 0., -Lby, Lbx, 0.],
            [  0.,  0., 0., 0., 0., 1.],
        ])

class Estrutura:
    def __init__(self, nos, elementos):
        self.nos = nos
        self.elementos = elementos
        self.Nnos = len(nos)
        self.Nelem = len(elementos)
        self.betaT = np.zeros((self.Nelem, 6, 6))
        self.rgi = np.zeros((self.Nelem, 6, 6))
        self.Mgi = np.zeros((self.Nelem, 6, 6))
        self.R = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.M = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.Rp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.Mp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.qxe = np.zeros((self.Nelem))
        self.qye = np.zeros((self.Nelem))
        self.Pne_ei = np.zeros((self.Nelem, 6))
        self.Pne_gi = np.zeros((self.Nelem, 6))
        self.Fne = np.zeros((self.Nnos * 3))
        self.F = np.zeros((self.Nnos * 3))

    def calcular_matriz_rigidez_global(self):
        for i, elemento in enumerate(self.elementos):
            beta_local = elemento.matriz_cinematica()
            rei_local = elemento.matriz_rigidez_local()
            self.betaT[i] = np.transpose(beta_local)
            self.rgi[i] = np.matmul(np.matmul(self.betaT[i], rei_local), beta_local)

    def montar_matriz_rigidez_global(self, con):
        for i in range(self.Nelem):
            for j in range(2):
                for k in range(3):
                    glib1 = 3 * con[i, j] - 3 + k
                    for l in range(2):
                        for m in range(3):
                            glib2 = 3 * con[i, l] - 3 + m
                            self.R[glib1, glib2] = self.R[glib1, glib2] + self.rgi[i, 3 * j + k, 3 * l + m]
    def imprimir_matriz_rigidez_global(self):
        print('\n\nMatriz de Rigidez Global:')
        for row in self.R:
            print(row.tolist())

    def metodo_penalty_rigidez(self, cod):
        self.Rp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        nmg = 10**20
        for i in range(3 * self.Nnos):
            for j in range(3 * self.Nnos):
                self.Rp[i, j] = self.R[i, j]
                if cod[i] == 1:  
                    self.Rp[i, i] = nmg
        np.set_printoptions(precision=2, suppress=True, linewidth=100)

        print('\n\nMatriz de rigidez global da estrutura após o método de penalty:')
        print(self.Rp)

    def calcular_matriz_massa_global(self):
        for i, elemento in enumerate(self.elementos):
            beta_local = elemento.matriz_cinematica()
            Mei_local = elemento.matriz_massa_local()
            self.betaT[i] = np.transpose(beta_local)
            self.Mgi[i] = np.matmul(np.matmul(self.betaT[i], Mei_local), beta_local)

    def montar_matriz_massa_global(self, con):
        for i in range(self.Nelem):
            for j in range(2):
                for k in range(3):
                    glib1 = 3 * con[i, j] - 3 + k
                    for l in range(2):
                        for m in range(3):
                            glib2 = 3 * con[i, l] - 3 + m
                            self.M[glib1, glib2] = self.M[glib1, glib2] + self.Mgi[i, 3 * j + k, 3 * l + m]
    def imprimir_matriz_massa_global(self):
        print('\n\nMatriz de Massa Global:')
        for row in self.M:
            print(row.tolist())

    def metodo_penalty_massa(self, cod):
        self.Mp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        nmg = 10**20
        for i in range(3 * self.Nnos):
            for j in range(3 * self.Nnos):
                self.Mp[i, j] = self.M[i, j]
                if cod[i] == 1:  # Verifica se a restrição é ativa
                    self.Mp[i, i] = nmg
        np.set_printoptions(precision=2, suppress=True, linewidth=100)

        print('\n\nMatriz de Massa global da estrutura após o método de penalty:')
        print(self.Mp)

    def calcular_carregamentos_distribuidos(self, qx, qy):
        for i, elemento in enumerate(self.elementos):
            cos = elemento.cos
            sen = elemento.sen
            self.qxe[i] = qx[i] * cos + qy[i] * sen
            self.qye[i] = qy[i] * cos - qx[i] * sen

    def calcular_reacoes_nodais(self):
        for i, elemento in enumerate(self.elementos):
            L = elemento.L
            self.Pne_ei[i, 0] = self.qxe[i] * L / 2
            self.Pne_ei[i, 1] = self.qye[i] * L / 2
            self.Pne_ei[i, 2] = -self.qye[i] * L**2 / 12
            self.Pne_ei[i, 3] = -self.qxe[i] * L / 2
            self.Pne_ei[i, 4] = -self.qye[i] * L / 2
            self.Pne_ei[i, 5] = -self.qye[i] * L**2 / 12

    def calcular_acoes_equivalentes(self):
        for i in range(self.Nelem):
            self.Pne_gi[i] = np.matmul(self.betaT[i], self.Pne_ei[i])

    def calcular_forcas_equivalentes(self, con):
        for i in range(self.Nelem):
            glib = 3 * (con[i, 0]) - 3
            self.Fne[glib] = self.Fne[glib] + self.Pne_gi[i, 0]
            self.Fne[glib + 1] = self.Fne[glib + 1] + self.Pne_gi[i, 1]
            self.Fne[glib + 2] = self.Fne[glib + 2] + self.Pne_gi[i, 2]
            glib = 3 * (con[i, 1]) - 3
            self.Fne[glib] = self.Fne[glib] + self.Pne_gi[i, 3]
            self.Fne[glib + 1] = self.Fne[glib + 1] + self.Pne_gi[i, 4]
            self.Fne[glib + 2] = self.Fne[glib + 2] + self.Pne_gi[i, 5]

    def unir_forcas(self, Fn):
        self.F = Fn + self.Fne

    def calcular_deslocamentos_globais(self):
        invRp = np.linalg.inv(self.Rp)
        self.D = np.matmul(invRp, self.F)

    def extrair_deslocamentos_elementos(self, con):
        self.Dgi = np.zeros((self.Nelem, 6))
        for i in range(self.Nelem):
            self.Dgi[i, 0] = self.D[3 * (con[i, 0] - 1)]
            self.Dgi[i, 1] = self.D[3 * (con[i, 0] - 1) + 1]
            self.Dgi[i, 2] = self.D[3 * (con[i, 0] - 1) + 2]
            self.Dgi[i, 3] = self.D[3 * (con[i, 1] - 1)]
            self.Dgi[i, 4] = self.D[3 * (con[i, 1] - 1) + 1]
            self.Dgi[i, 5] = self.D[3 * (con[i, 1] - 1) + 2]

    def calcular_deslocamentos_locais(self):
        self.Dei = np.zeros((self.Nelem, 6))
        for i in range(self.Nelem):
            self.Dei[i] = np.matmul(self.elementos[i].matriz_cinematica(), self.Dgi[i])

    def calcular_reacoes(self):
        self.Reac = np.matmul(self.R, self.D) - self.Fne
        print("\n\n")
        for i in range(self.Nnos):
            print(f'Reação horizontal (positivo para a direita) no nó {i + 1}:\n')
            print(self.Reac[3 * i])
            print(f'\nReação vertical (positivo para cima) no nó {i + 1}:\n')
            print(self.Reac[3 * i + 1])
            print(f'\nReação ao giro (positivo no sentido anti-horário) no nó {i + 1}:\n')
            print(self.Reac[3 * i + 2])

    def calcular_esforcos_solicitantes(self):
        self.Pei = np.zeros((self.Nelem, 6))
        for i in range(self.Nelem):
            rei_local = self.elementos[i].matriz_rigidez_local()
            self.Pei[i] = np.matmul(rei_local, self.Dei[i]) - self.Pne_ei[i]
        print('\n\n')
        print('Esforços solicitantes nodais:')
        for i in range(self.Nelem):
            print(f'\nEsforço normal no extremo esquerdo (positivo para a esquerda) da barra {i + 1}:\n')
            print(self.Pei[i, 0])
            print(f'\nEsforço cortante no extremo esquerdo (positivo para cima) da barra {i + 1}:\n')
            print(self.Pei[i, 1])
            print(f'\nMomento fletor no extremo esquerdo (positivo no sentido horário) da barra {i + 1}:\n')
            print(self.Pei[i, 2])
            print(f'\nEsforço normal no extremo direito (positivo para a direita) da barra {i + 1}:\n')
            print(self.Pei[i, 3])
            print(f'\nEsforço cortante no extremo direito (positivo para baixo) da barra {i + 1}:\n')
            print(self.Pei[i, 4])
            print(f'\nMomento fletor no extremo direito (positivo no sentido anti-horário) da barra {i + 1}:\n')
            print(self.Pei[i, 5])

    def analise_modal(self):

        w2, v = eigh(self.Rp, self.Mp)
        frequencias = np.sqrt(np.real(w2))
        indices = np.argsort(frequencias)
        frequencias_natural = frequencias[indices]
        modos_vibracao = v[:, indices]/np.linalg.norm(v, axis=0)

        print("Frequências naturais (rad/s):")
        for freq in frequencias_natural:
            print(freq)

        modos_vibracao_transposta = modos_vibracao.T
        print("\nModos naturais (formato de matriz, cada coluna representa um modo):")
        for i in range(modos_vibracao_transposta.shape[1]):
            print(f"Modo {i+1}:\n{modos_vibracao_transposta[:, i]}")
        return frequencias_natural, modos_vibracao

#------------------------------- Definição dos nós-----------------------------------------------------
n1 = No(id=1, x=0, y=0, restricoes=[1, 1, 1], deslocamentos_prescritos=[0, 0, 0])
n2 = No(id=2, x=0, y=1, restricoes=[0, 0, 0], deslocamentos_prescritos=[0, 0, 0]) 
n3 = No(id=3, x=0, y=3, restricoes=[0, 0, 0], deslocamentos_prescritos=[0, 0, 0])  
n4 = No(id=4, x=1.5, y=3, restricoes=[1, 1, 1], deslocamentos_prescritos=[0, 0, 0])
nos = [n1, n2, n3, n4]

#-----------------------------Definição dos elementos---------------------------------------------------
elemento1 = Elemento(id=1, no_inicial=n1, no_final=n2, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0, poisson = 0.3,densidade= 7850)
elemento2 = Elemento(id=2, no_inicial=n2, no_final=n3, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0, poisson = 0.3, densidade= 7850)
elemento3 = Elemento(id=3, no_inicial=n3, no_final=n4, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0, poisson = 0.3,densidade= 7850)
elementos = [elemento1, elemento2, elemento3]

conectividade = np.array([
    [1, 2],  
    [2, 3],  
    [3, 4]   
])

#--------------------------Aplicação dos carregamentos---------------------------------------------------
F = np.zeros(12) 
F[3] = 5000  

#-------------------------Instância da projeto------------------------------------------
estrutura = Estrutura(nos, elementos)
estrutura.calcular_matriz_rigidez_global()
estrutura.montar_matriz_rigidez_global(conectividade)
estrutura.imprimir_matriz_rigidez_global()
estrutura.calcular_matriz_massa_global()
estrutura.montar_matriz_massa_global(conectividade)
estrutura.imprimir_matriz_massa_global()
estrutura.calcular_carregamentos_distribuidos(qx=np.array([0, 0, 0]), qy=np.array([0, 0, 0]))
estrutura.calcular_reacoes_nodais()
estrutura.calcular_acoes_equivalentes()
estrutura.calcular_forcas_equivalentes(conectividade)
estrutura.unir_forcas(F)
cod_restricoes = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
estrutura.metodo_penalty_rigidez(cod_restricoes)
estrutura.metodo_penalty_massa(cod_restricoes)
estrutura.calcular_deslocamentos_globais()
estrutura.extrair_deslocamentos_elementos(conectividade)
estrutura.calcular_deslocamentos_locais()
estrutura.calcular_reacoes()
estrutura.calcular_esforcos_solicitantes()
estrutura.analise_modal()
