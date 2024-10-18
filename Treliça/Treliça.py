''' Autor _ JOAB MANOEL ALMEIDA SANTOS
    Disciplina: Mecânica Computacional das estruturas
    Professor: Dr. Eduardo Toledo
    Universidade Federal de Alagoas
    Mestrando em Estruturas e Materiais 
    Implementação para treliça plana
'''

import numpy as np
import matplotlib.pyplot as plt
import json

class Material:
    def __init__(self, E, A):
        self.E = E 
        self.A = A  

class Barra:
    def __init__(self, n1, n2, material, coord):
        self.n1 = n1
        self.n2 = n2
        self.material = material
        self.coord = coord
        self.L = self.calcular_comprimento()
        self.cos_theta = (self.coord[self.n2, 0] - self.coord[self.n1, 0]) / self.L
        self.sin_theta = (self.coord[self.n2, 1] - self.coord[self.n1, 1]) / self.L

    def calcular_comprimento(self):
        x1, y1 = self.coord[self.n1]
        x2, y2 = self.coord[self.n2]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def matriz_rigidez(self):
        E = self.material.E
        A = self.material.A
        L = self.L
        k = (E * A / L) * np.array([[self.cos_theta**2, self.cos_theta * self.sin_theta, -self.cos_theta**2, -self.cos_theta * self.sin_theta],
                                     [self.cos_theta * self.sin_theta, self.sin_theta**2, -self.cos_theta * self.sin_theta, -self.sin_theta**2],
                                     [-self.cos_theta**2, -self.cos_theta * self.sin_theta, self.cos_theta**2, self.cos_theta * self.sin_theta],
                                     [-self.cos_theta * self.sin_theta, -self.sin_theta**2, self.cos_theta * self.sin_theta, self.sin_theta**2]])
        return k

    def matriz_massa_consistente(self):
        A = self.material.A
        L = self.L
        rho = 1000  
        m_barra = rho * A * L / 6

        m_local = m_barra * np.array([[2, 0, 1, 0],
                                       [0, 2, 0, 1],
                                       [1, 0, 2, 0],
                                       [0, 1, 0, 2]])
        return m_local

class Trelica:
    def __init__(self, coord, conect, material, restrs, cargasNos):
        self.coord = coord
        self.conect = conect
        self.material = material
        self.restrs = restrs
        self.cargasNos = cargasNos
        self.numNos = coord.shape[0]
        self.numGDL = 2 * self.numNos  
        self.K_global = np.zeros((self.numGDL, self.numGDL))
        self.M_global = np.zeros((self.numGDL, self.numGDL))  
        self.P = self.montar_vetor_forcas()

    def montar_matriz_rigidez(self):
        for i in range(self.conect.shape[0]):
            barra = Barra(self.conect[i, 0], self.conect[i, 1], self.material, self.coord)
            k = barra.matriz_rigidez()
            numero_matriz = np.array([2 * barra.n1, 2 * barra.n1 + 1, 2 * barra.n2, 2 * barra.n2 + 1])  
            for a in range(4):
                for b in range(4):
                    self.K_global[numero_matriz[a], numero_matriz[b]] += k[a, b]

    def montar_vetor_forcas(self):
        total_gdl = self.numGDL
        P = np.zeros(total_gdl)

        N = 0
        for i in range(self.numNos):
            for j in range(2):  
                if i < self.cargasNos.shape[0] and j < self.cargasNos.shape[1]:
                    P[N] = self.cargasNos[i, j]
                else:
                    P[N] = 0  
                N += 1
        return P

    def aplicar_condicoes_contorno(self):
        numGDL = self.K_global.shape[0]
        gdl_livres = np.ones(numGDL, dtype=bool)

        for restr in self.restrs:
            gdl_livres[2 * restr] = False  
            gdl_livres[2 * restr + 1] = False  

        K_reduzida = self.K_global[gdl_livres, :][:, gdl_livres]
        P_reduzido = self.P[gdl_livres]

        return K_reduzida, P_reduzido, gdl_livres

    def calcular_deslocamentos(self):
        self.montar_matriz_rigidez()  
        print("\nMatriz de Rigidez Global:\n", np.array2string(self.K_global, precision=2, suppress_small=True, floatmode='fixed', max_line_width=120))
        K_reduzida, P_reduzido, gdl_livres = self.aplicar_condicoes_contorno()
        U_reduzida = np.linalg.solve(K_reduzida, P_reduzido)
        print("\nMatriz de Rigidez Reduzida:\n", np.array2string(K_reduzida, precision=2, suppress_small=True, floatmode='fixed', max_line_width=120))
        U = np.zeros(self.K_global.shape[0])
        U[gdl_livres] = U_reduzida
        return U

    def calcular_esforcos_normais(self, U):
        esforcos_normais = np.zeros(self.conect.shape[0])

        for i in range(self.conect.shape[0]):
            barra = Barra(self.conect[i, 0], self.conect[i, 1], self.material, self.coord)
            U_n1 = U[2 * barra.n1:2 * barra.n1 + 2]
            U_n2 = U[2 * barra.n2:2 * barra.n2 + 2]

            dU = U_n2 - U_n1
            deformacao = (barra.cos_theta * dU[0] + barra.sin_theta * dU[1]) / barra.L
            esforcos_normais[i] = self.material.E * self.material.A * deformacao

        return esforcos_normais

    def calcular_e_plotar(self):
        U = self.calcular_deslocamentos()
        esforcos_normais = self.calcular_esforcos_normais(U)
        self.plotar_trelica(U, esforcos_normais)
        return U, esforcos_normais 

    def plotar_trelica(self, U, esforcos_normais):
        plt.figure(figsize=(10, 8))

        for i in range(self.conect.shape[0]):
            n1 = self.conect[i, 0]
            n2 = self.conect[i, 1]
            x_values = [self.coord[n1, 0], self.coord[n2, 0]]
            y_values = [self.coord[n1, 1], self.coord[n2, 1]]
            plt.plot(x_values, y_values, 'b-', linewidth=2)  

            centro_x = (self.coord[n1, 0] + self.coord[n2, 0]) / 2
            centro_y = (self.coord[n1, 1] + self.coord[n2, 1]) / 2
            plt.text(centro_x, centro_y, f'{esforcos_normais[i]:.2f} N', fontsize=10, ha='center', color='purple')

        plt.scatter(self.coord[:, 0], self.coord[:, 1], color='r', s=100)  
        for idx, (x, y) in enumerate(self.coord):
            plt.text(x, y, f'N{idx}', fontsize=12, ha='right', color='black')

        deslocamentos = self.coord + U.reshape(-1, 2)  
        plt.scatter(deslocamentos[:, 0], deslocamentos[:, 1], color='g', s=100, label='Deslocamentos')  

        for restr in self.restrs:
            plt.scatter(self.coord[restr, 0], self.coord[restr, 1], color='black', s=100) 
            plt.plot(self.coord[restr, 0], self.coord[restr, 1], '^', color='black', markersize=30)  

        plt.title("Treliça 2D com Deslocamentos, Restrições e Esforços Normais")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()

def carregar_dados_json(arquivo):
    with open(arquivo, 'r') as f:
        dados = json.load(f)
    coord = np.array(dados['coord'])
    conect = np.array(dados['conect'])
    restricao = np.array(dados['restricao'])
    dadosElem = np.array(dados['material'])
    cargasNos = np.array(dados['cargasNos'])

    return coord, conect, restricao, dadosElem, cargasNos

arquivo_json = 'dados_trelica.json' 
coord, conect, restricao, dadosElem, cargasNos = carregar_dados_json(arquivo_json)

material = Material(dadosElem[0], dadosElem[1])
trelica = Trelica(coord, conect, material, restricao, cargasNos)
U, esforcos_normais = trelica.calcular_e_plotar()

print("Deslocamentos (em metros):")
for i in range(trelica.numNos):
    print(f"Nó {i}: Deslocamento em X: {U[2*i]:.6f}, Deslocamento em Y: {U[2*i + 1]:.6f}")

print("\nEsforços Normais (em N):")
for i in range(trelica.conect.shape[0]):
    print(f"Barra {i}: Esforço Normal: {esforcos_normais[i]:.2f} N")


#Caso não utilize json 
'''coord = np.array([[0, 0], [3, 0], [6, 0], [3, 2]])  # Coordenadas dos nós
conect = np.array([[0, 1], [1, 2], [2, 3], [3, 1], [3, 0]])  # Conexões entre os nós
restricao = np.array([0, 2])  # Restrições para cada nó restringe x e y 
dadosElem = np.array([210e6, 0.007854])  # Dados do material [E(Pa), A(m2)]
cargasNos = np.array([[0, 0], [0, -50000], [0, 0], [100000, 0]])  # Cargas nodais(N)'''