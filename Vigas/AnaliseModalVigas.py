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
        self.restricoes = restricoes  # [horizontal, vertical, rotacional]
        self.deslocamentos_prescritos = deslocamentos_prescritos  # [u, v, θ]

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
        G = self.calcula_modulo_cisalhamento()

        k = E * I / L**3

        return np.array([[12 * k, 6 * L * k, -12 * k, 6 * L * k, 0, 0],
                     [6 * L * k, 4 * L**2 * k, -6 * L * k, 2 * L**2 * k, 0, 0],
                     [-12 * k, -6 * L * k, 12 * k, -6 * L * k, 0, 0],
                     [6 * L * k, 2 * L**2 * k, -6 * L * k, 4 * L**2 * k, 0, 0],
                     [0, 0, 0, 0, A * E / L, 0],
                     [0, 0, 0, 0, 0, G * I / L]])

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
        
    def matriz_massa_local(self):
        L = self.L
        A = self.A
        densidade = self.densidade
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
        
        M = np.array([
            [(j1 * j7 + j1 * j8), (j1 * j2 + j1 * j8),    0,                        (j1 * j3 + j1 * j7), (j1 * j4 + j1 * j8)],
            [(j1 * j2 + j1 * j8), (j1 * j5 + j1 * j9),    0,                       -(j1 * j4 + j1 * j8), (j1 * j6 + j1 * j10)],
            [0,                     0,                     densidade * A * L / 3,    0,                     0],
            [(j1 * j3 + j1 * j7), -(j1 * j4 + j1 * j8),  0,                        (j1 * j1 + j1 * j7), -(j1 * j2 + j1 * j8)],
            [(j1 * j4 + j1 * j8), (j1 * j6 + j1 * j10),  0,                       -(j1 * j2 + j1 * j8), (j1 * j5 + j1 * j9)],
        ])
        
        return M
    
class Estrutura:
    def __init__(self, nos, elementos):
        self.nos = nos
        self.elementos = elementos
        self.Nnos = len(nos)
        self.Nelem = len(elementos)
        self.betaT = np.zeros((self.Nelem, 6, 6))
        self.rgi = np.zeros((self.Nelem, 6, 6))
        self.R = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.Rp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        self.qxe = np.zeros((self.Nelem))
        self.qye = np.zeros((self.Nelem))
        self.Pne_ei = np.zeros((self.Nelem, 6))
        self.Pne_gi = np.zeros((self.Nelem, 6))
        self.Fne = np.zeros((self.Nnos * 3))
        self.F = np.zeros((self.Nnos * 3))

    def calcular_matrizes_rigidez_globais(self):
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
        print('Matriz de Rigidez Global:')
        for row in self.R:
            print(row.tolist())
        
    def metodo_penalty(self, cod):
        self.Rp = np.zeros((3 * self.Nnos, 3 * self.Nnos))
        nmg = 10**20
        for i in range(3 * self.Nnos):
            for j in range(3 * self.Nnos):
                self.Rp[i, j] = self.R[i, j]
                if cod[i] == 1:  # Verifica se a restrição é ativa
                    self.Rp[i, i] = nmg
        print('\n\nMatriz de rigidez global da estrutura após o método de penalty:')
        print(self.Rp)
    
    def montar_matriz_massa_global(self, con):
        self.Mg = np.zeros((3 * self.Nnos, 3 * self.Nnos))  # Matriz de massa global
        for i in range(self.Nelem):
            M_local = self.elementos[i].matriz_massa_local()  # Obtém a matriz de massa local
            for j in range(2):  # Para os dois nós do elemento
                for k in range(3):  # Para as 3 direções (u, v, θ)
                    glib1 = 3 * con[i, j] - 3 + k  # Índice global do nó j
                    for l in range(2):  # Para os dois nós do elemento
                        for m in range(3):  # Para as 3 direções (u, v, θ)
                            glib2 = 3 * con[i, l] - 3 + m  # Índice global do nó l
                            self.Mg[glib1, glib2] += M_local[3 * j + k, 3 * l + m]
        print('Matriz de Massa Global:')
        for row in self.Mg:
            print(row.tolist())
    

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
        #print('Reações:')
        #print(self.Reac)
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
        #print(self.Pei)
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
            
    def analisar_modal(self, restricoes):
    # Chama o método para aplicar as condições de contorno
        self.aplicar_condicoes_contorno(restricoes)

    # Agora você pode acessar R_reduzida e Mg_reduzida
        K = self.R_reduzida  # Matriz de rigidez global
        M = self.Mg_reduzida  # Matriz de massa global

    # Calcula os valores próprios e vetores próprios
        eigenvalues, eigenvectors = eigh(K, M)

    # Frequências naturais são a raiz quadrada dos valores próprios
        frequencias_naturais = np.sqrt(eigenvalues)

        print("Frequências Naturais (Hz):")
        print(frequencias_naturais)

        return frequencias_naturais, eigenvectors
    
    
    
#------------------------------- Definição dos nós-----------------------------------------------------
n1 = No(id=1, x=0, y=0, restricoes=[1, 1, 0], deslocamentos_prescritos=[0, 0, 0])  
n2 = No(id=2, x=5000, y=0, restricoes=[1, 0, 0], deslocamentos_prescritos=[0, 0, 0])
nos = [n1, n2]

#-----------------------------Definição dos elementos---------------------------------------------------
elemento1 = Elemento(id=1, no_inicial=n1, no_final=n2, E=210000, A=700, I=40000, qx=0, qy=10, poisson=0.3, densidade=7850)
elementos = [elemento1]

conectividade = np.array([[1, 2]])

#--------------------------Aplicação dos carregamentos---------------------------------------------------
F = np.zeros(6) 
F[1] = 5000  # Carregamento horizontal no nó 2




#-------------------------Instância da projeto------------------------------------------

estrutura = Estrutura(nos, elementos)
estrutura.calcular_matrizes_rigidez_globais()
estrutura.montar_matriz_rigidez_global(conectividade)
estrutura.calcular_carregamentos_distribuidos(qx=np.array([0, 0, 0]), qy=np.array([0, 0, 0]))
estrutura.calcular_reacoes_nodais()
estrutura.calcular_acoes_equivalentes()
estrutura.calcular_forcas_equivalentes(conectividade)
estrutura.unir_forcas(F)
estrutura.imprimir_matriz_rigidez_global()
cod_restricoes = np.array([1, 1, 0, 0, 1, 0])
estrutura.metodo_penalty(cod_restricoes)
estrutura.calcular_deslocamentos_globais()
estrutura.extrair_deslocamentos_elementos(conectividade)
estrutura.calcular_deslocamentos_locais()
estrutura.calcular_reacoes()
estrutura.calcular_esforcos_solicitantes()
#estrutura.montar_matriz_massa_global(conectividade)
#estrutura.analisar_modal(cod_restricoes)



    