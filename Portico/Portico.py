''' Autor _ JOAB MANOEL ALMEIDA SANTOS
    Disciplina: Mecânica Computacional das estruturas
    Professor: Dr. Eduardo Toledo
    Universidade Federal de Alagoas
    Mestrando em Estruturas e Materiais 
    Implementação para pórtico plano
'''

import numpy as np
import matplotlib.pyplot as plt
import json

class No:
    def __init__(self, id, x, y, restricoes, deslocamentos_prescritos):
        self.id = id
        self.x = x
        self.y = y
        self.restricoes = restricoes 
        self.deslocamentos_prescritos = deslocamentos_prescritos  

class Elemento:
    def __init__(self, id, no_inicial, no_final, E, A, I, qx, qy):
        self.id = id
        self.no_inicial = no_inicial
        self.no_final = no_final
        self.E = E
        self.A = A
        self.I = I
        self.qx = qx
        self.qy = qy
        self.L = self.calcula_comprimento()
        self.cos, self.sen = self.calcula_inclinacao()

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
        print("Deslocamentos:")
        print(self.D)

        deslocamentos = np.zeros((self.Nnos, 2))  
        for i in range(self.Nnos):
            deslocamentos[i, 0] = self.D[3 * i]     
            deslocamentos[i, 1] = self.D[3 * i + 1] 

        return deslocamentos

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

    def analisar(self, F, conectividade, cod_restricoes):
        self.calcular_matrizes_rigidez_globais()
        self.montar_matriz_rigidez_global(conectividade)
        self.calcular_carregamentos_distribuidos(qx=np.array([0, 0, 0]), qy=np.array([0, 0, 0]))
        self.calcular_reacoes_nodais()
        self.calcular_acoes_equivalentes()
        self.calcular_forcas_equivalentes(conectividade)
        self.unir_forcas(F)
        self.imprimir_matriz_rigidez_global()
        self.metodo_penalty(cod_restricoes)
        deslocamentos = self.calcular_deslocamentos_globais()
        self.extrair_deslocamentos_elementos(conectividade)
        self.calcular_deslocamentos_locais()
        self.calcular_reacoes()
        self.calcular_esforcos_solicitantes()
        return deslocamentos

def visualizar_estrutura_com_deslocamentos(estrutura, esforcos_normais, esforcos_cortantes, momentos_fletores, deslocamentos, escala = 300):
    plt.figure(figsize=(10, 8))
    for elemento in estrutura.elementos:
        x_values = [elemento.no_inicial.x, elemento.no_final.x]
        y_values = [elemento.no_inicial.y, elemento.no_final.y]
        plt.plot(x_values, y_values, 'b-', linewidth=2)  

    for i, no in enumerate(estrutura.nos):
        plt.scatter(no.x, no.y, color='r', s=100)  
        plt.text(no.x+0.5, no.y-0.1,  
                 f'N{no.id}\nV: {esforcos_normais[i]:.2f} N\nH: {esforcos_cortantes[i]:.2f} N\nM: {momentos_fletores[i]:.2f} Nm', 
                 fontsize=10, ha='center', color='black')

        if no.id == 2:
            deslocamento_x = no.x + deslocamentos[i][0]*escala
            deslocamento_y = no.y + deslocamentos[i][1]*escala
            plt.scatter(deslocamento_x, deslocamento_y, color='green', s=100)  
            plt.text(deslocamento_x, deslocamento_y, f'D: {deslocamentos[i][0]*1000:.7f}mm, {deslocamentos[i][1]*1000:.7f}mm', 
                     fontsize=10, ha='center', color='black')  

    for no in estrutura.nos:
        if no.restricoes[0] == 1:  
            plt.plot(no.x, no.y, '^', color='black', markersize=12)  
        if no.restricoes[1] == 1:  
            plt.plot(no.x, no.y, 's', color='black', markersize=12) 

    plt.title("Esforços Normais, Cortantes e Momentos Fletores Nodais", pad=20)
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

def carregar_dados_json(arquivo):
    with open(arquivo, 'r') as f:
        dados = json.load(f)
    nos = []
    for no in dados['nos']:
        nos.append(No(id=no['id'], x=no['x'], y=no['y'], restricoes=no['restricoes'], deslocamentos_prescritos=no['deslocamentos_prescritos']))

    elementos = []
    for elem in dados['elementos']:
        no_inicial = next(no for no in nos if no.id == elem['no_inicial'])
        no_final = next(no for no in nos if no.id == elem['no_final'])
        elementos.append(Elemento(id=elem['id'], no_inicial=no_inicial, no_final=no_final, E=elem['E'], A=elem['A'], I=elem['I'], qx=elem['qx'], qy=elem['qy']))

    conectividade = np.array(dados['conectividade'])
    F = np.array(dados['carregamentos']['F'])

    return nos, elementos, conectividade, F

nos, elementos, conectividade, F = carregar_dados_json('dados_Portico.json')
estrutura = Estrutura(nos, elementos)
cod_restricoes = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1])
deslocamentos = estrutura.analisar(F, conectividade, cod_restricoes)
esforcos_normais = estrutura.Pei[:, [0, 3]].flatten()  
esforcos_cortantes = estrutura.Pei[:, [1, 4]].flatten()  
momentos_fletores = estrutura.Pei[:, [2, 5]].flatten()   
visualizar_estrutura_com_deslocamentos(estrutura, esforcos_normais, esforcos_cortantes, momentos_fletores, deslocamentos)

    #Exemplo de entrada de dados sem uso do json
'''n1 = No(id=1, x=0, y=0, restricoes=[1, 1, 1], deslocamentos_prescritos=[0, 0, 0])
n2 = No(id=2, x=0, y=1, restricoes=[0, 0, 0], deslocamentos_prescritos=[0, 0, 0]) 
n3 = No(id=3, x=0, y=3, restricoes=[0, 0, 0], deslocamentos_prescritos=[0, 0, 0])  
n4 = No(id=4, x=1.5, y=3, restricoes=[1, 1, 1], deslocamentos_prescritos=[0, 0, 0])
nos = [n1, n2, n3, n4]
elemento1 = Elemento(id=1, no_inicial=n1, no_final=n2, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0)
elemento2 = Elemento(id=2, no_inicial=n2, no_final=n3, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0)
elemento3 = Elemento(id=3, no_inicial=n3, no_final=n4, E=210000000000, A=0.0025, I=0.00000052083, qx=0, qy=0)
elementos = [elemento1, elemento2, elemento3]

conectividade = np.array([
    [1, 2],  
    [2, 3],  
    [3, 4]   
])
F = np.zeros(12) 
F[3] = 5000  # Carregamento horizontal no nó 2'''
