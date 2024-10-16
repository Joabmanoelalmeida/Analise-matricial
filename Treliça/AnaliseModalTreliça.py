''' Autor _ JOAB MANOEL ALMEIDA SANTOS
    Disciplina: Mecânica Computacional das estruturas
    Professor: Dr. Eduardo Toledo
    Universidade Federal de Alagoas
    Mestrando em Estruturas e Materiais 
    Implementação para treliça plana
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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
        rho = 1000  # Densidade do material
        
        # Massa total da barra
        m_barra = rho * A * L/6

        # Matriz de massa consistente no sistema local
        m_local = m_barra  * np.array([ [2, 0, 1, 0],
                                        [0, 2, 0, 1],
                                        [1, 0, 2, 0],
                                        [0, 1, 0, 2]])

        # Transformação para o sistema global
        '''T = np.array([[self.cos_theta, self.sin_theta, 0, 0],
                      [-self.sin_theta, self.cos_theta, 0, 0],
                      [0, 0, self.cos_theta, self.sin_theta],
                      [0, 0, -self.sin_theta, self.cos_theta]])

        m_global = np.dot(np.dot(T.T, m_local), T)'''
        return m_local

class Trelica:
    def __init__(self, coord, conect, material, restrs, cargasNos):
        self.coord = coord
        self.conect = conect
        self.material = material
        self.restrs = restrs
        self.cargasNos = cargasNos
        self.numNos = coord.shape[0]
        self.numGDL = 2 * self.numNos  # 2 graus de liberdade por nó
        self.K_global = np.zeros((self.numGDL, self.numGDL))
        self.M_global = np.zeros((self.numGDL, self.numGDL))  # Adicionar matriz de massa global
        self.P = self.montar_vetor_forcas()

    def montar_matriz_rigidez(self):
        for i in range(self.conect.shape[0]):
            barra = Barra(self.conect[i, 0], self.conect[i, 1], self.material, self.coord)
            k = barra.matriz_rigidez()
            numero_matriz = np.array([2 * barra.n1, 2 * barra.n1 + 1, 2 * barra.n2, 2 * barra.n2 + 1])  # Graus de liberdade
            for a in range(4):
                for b in range(4):
                    self.K_global[numero_matriz[a], numero_matriz[b]] += k[a, b]

    def montar_matriz_massa(self):
        for i in range(self.conect.shape[0]): #itera sobre self.conect. O self.conect contém as conexões entre os nós do sistema, self.conect.shape[0] retorna o número de  elementos).
            barra = Barra(self.conect[i, 0], self.conect[i, 1], self.material, self.coord)
            m = barra.matriz_massa_consistente()
            numero_matriz = np.array([2 * barra.n1, 2 * barra.n1 + 1, 2 * barra.n2, 2 * barra.n2 + 1])  # Graus de liberdade
            for a in range(4):
                for b in range(4):
                    self.M_global[numero_matriz[a], numero_matriz[b]] += m[a, b]

    def montar_vetor_forcas(self):
        total_gdl = self.numGDL
        P = np.zeros(total_gdl)

        N = 0
        for i in range(self.numNos):
            for j in range(2):  # Dois graus de liberdade
                if i < self.cargasNos.shape[0] and j < self.cargasNos.shape[1]:
                    P[N] = self.cargasNos[i, j]
                else:
                    P[N] = 0  # Atribui 0 se não houver carga
                N += 1
        return P

    def aplicar_condicoes_contorno(self):
        numGDL = self.K_global.shape[0]
        gdl_livres = np.ones(numGDL, dtype=bool)

        for restr in self.restrs:
            gdl_livres[2 * restr] = False  # Grau de liberdade em x
            gdl_livres[2 * restr + 1] = False  # Grau de liberdade em y

        K_reduzida = self.K_global[gdl_livres, :][:, gdl_livres]
        P_reduzido = self.P[gdl_livres]

        return K_reduzida, P_reduzido, gdl_livres

    def calcular_deslocamentos(self):
        self.montar_matriz_rigidez()  # Certificar de que a matriz de rigidez é montada antes de calcular os deslocamentos
        self.montar_matriz_massa()  # Montar a matriz de massa
        print("\nMatriz de Rigidez Global:\n", np.array2string(self.K_global, precision=2, suppress_small=True, floatmode='fixed', max_line_width=120))
        print("\nMatriz de Massa Global:\n", np.array2string(self.M_global, precision=2, suppress_small=True, floatmode='fixed', max_line_width=120))

        K_reduzida, P_reduzido, gdl_livres = self.aplicar_condicoes_contorno()
        U_reduzida = np.linalg.solve(K_reduzida, P_reduzido)

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

    def plotar_trelica(self, U, esforcos_normais):
        plt.figure(figsize=(10, 8))
        
        # Plotar as barras
        for i in range(self.conect.shape[0]):
            n1 = self.conect[i, 0]
            n2 = self.conect[i, 1]
            x_values = [self.coord[n1, 0], self.coord[n2, 0]]
            y_values = [self.coord[n1, 1], self.coord[n2, 1]]
            plt.plot(x_values, y_values, 'b-', linewidth=2)  # Plota a barra em azul

            # Plotar esforços normais nas barras
            centro_x = (self.coord[n1, 0] + self.coord[n2, 0]) / 2
            centro_y = (self.coord[n1, 1] + self.coord[n2, 1]) / 2
            plt.text(centro_x, centro_y, f'{esforcos_normais[i]:.2f} N', fontsize=10, ha='center', color='purple')

        # Plotar os nós
        plt.scatter(self.coord[:, 0], self.coord[:, 1], color='r', s=100)  # Plota os nós em vermelho
        for idx, (x, y) in enumerate(self.coord):
            plt.text(x, y, f'N{idx}', fontsize=12, ha='right', color='black')

        # Plotar os deslocamentos
        deslocamentos = self.coord + U.reshape(-1, 2)  # Adiciona deslocamentos às coordenadas originais
        plt.scatter(deslocamentos[:, 0], deslocamentos[:, 1], color='g', s=100, label='Deslocamentos')  # Plota os deslocamentos em verde
        
        # Plotar as restrições
        for restr in self.restrs:
            plt.scatter(self.coord[restr, 0], self.coord[restr, 1], color='black', s=100)  # Marca o nó como restrito
            plt.plot(self.coord[restr, 0], self.coord[restr, 1], '^', color='black', markersize=30)  # Adiciona o triângulo invertido como apoio
        
        plt.title("Treliça 2D com Deslocamentos, Restrições e Esforços Normais")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid()
        plt.axis('equal')
        plt.legend()
        plt.show()

    def analise_modal(self):
        """
        Realiza a análise modal resolvendo o problema de autovalores e autovetores
        (K_global - λ M_global) Φ = 0, onde λ são os autovalores e Φ os autovetores.
        """
        K_reduzida, _, gdl_livres = self.aplicar_condicoes_contorno()
        M_reduzida = self.M_global[gdl_livres, :][:, gdl_livres]
        # Resolver o problema generalizado de autovalores e autovetores
        autovalores, autovetores = eigh(K_reduzida, M_reduzida)
        # Frequências naturais (rad/s) e os modos de vibração
        frequencias_naturais = np.sqrt(autovalores)
        modos_vibracao = autovetores / np.linalg.norm(autovetores, axis=0)
        return frequencias_naturais, modos_vibracao
    
# Exemplo de dados para a treliça
coord = np.array([[0, 0], [3, 0], [6, 0], [3, 2]])  # Coordenadas dos nós
conect = np.array([[0, 1], [1, 2], [2, 3], [3, 1], [3, 0]])  # Conexões entre os nós
restricao = np.array([0, 2])  # Restrições para cada nó restringe x e y 
dadosElem = np.array([210e6, 0.007854])  # Dados do material [E(Pa), A(m2)]
cargasNos = np.array([[0, 0], [0, -50000], [0, 0], [100000, 0]])  # Cargas nodais(N)

# Instanciação do material e da treliça
material = Material(dadosElem[0], dadosElem[1])
trelica = Trelica(coord, conect, material, restricao, cargasNos)

# Cálculo de deslocamentos e esforços
U = trelica.calcular_deslocamentos()
esforcos_normais = trelica.calcular_esforcos_normais(U)

# Impressão dos resultados no terminal
print("Deslocamentos (em metros):")
for i in range(trelica.numNos):
    print(f"Nó {i}: Deslocamento em X: {U[2*i]:.6f}, Deslocamento em Y: {U[2*i + 1]:.6f}")

print("\nEsforços Normais (em N):")
for i in range(trelica.conect.shape[0]):
    print(f"Barra {i}: Esforço Normal: {esforcos_normais[i]:.2f} N")

# Análise modal
frequencias_naturais, modos_vibracao = trelica.analise_modal()
print("\nFrequências Naturais (rad/s):")
for i, freq in enumerate(frequencias_naturais):
    print(f"Modo {i + 1}: {freq:.4f} rad/s")
    
modos_vibracao_transposta = modos_vibracao.T
print("\nModos de vibração:")
for i, modv in enumerate(modos_vibracao_transposta):
    print(f"Modo {i + 1}: {modv}")


trelica.plotar_trelica(U, esforcos_normais)  
