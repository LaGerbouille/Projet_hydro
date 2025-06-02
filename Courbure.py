import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Courbure():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def courbures_pente_non_nulle(self):
        grilleA = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]]) / (6 * self.pas ** 2)
        A = convolve(self.mnt, grilleA, mode='constant', cval=0.0)
        grilleB = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]]) / (6 * self.pas ** 2)
        B = convolve(self.mnt, grilleB, mode='constant', cval=0.0)
        grilleC = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * self.pas ** 2)
        C = convolve(self.mnt, grilleC, mode='constant', cval=0.0)
        grilleD = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6 * self.pas ** 2)
        D = convolve(self.mnt, grilleD, mode='constant', cval=0.0)
        grilleE = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / (6 * self.pas ** 2)
        E = convolve(self.mnt, grilleE, mode='constant', cval=0.0)
        grilleF = np.array([[-1, 2, -1], [2, 5, 2], [-1, 2, -1]]) / 9
        F = convolve(self.mnt, grilleF, mode='constant', cval=0.0)
        fx = D
        fy = E
        fxx = 2 * A
        fyy = 2 * B
        fxy = C
        eps = 0.01
        p = (fx ** 2 + fy ** 2) * ((fx ** 2 + fy ** 2) < eps) * np.nan
        q = p + 1
        kv = -(fxx * fx ** 2 + 2 * fxy * fx * fy + fyy * fy ** 2) / (p * (q ** 3) ** (1 / 2))
        kh = -(fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) / (p * q ** (1 / 2))
        return kv, kh


    def courbures_pente_nulle(self):
        grilleA = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]]) / (6 * self.pas ** 2)
        A = convolve(self.mnt, grilleA, mode='constant', cval=0.0)
        grilleB = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]]) / (6 * self.pas ** 2)
        B = convolve(self.mnt, grilleB, mode='constant', cval=0.0)
        grilleC = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * self.pas ** 2)
        C = convolve(self.mnt, grilleC, mode='constant', cval=0.0)
        kmin = -A - B - np.sqrt((A - B) ** 2 + C ** 2)
        kmax = -A - B + np.sqrt((A - B) ** 2 + C ** 2)
        return kmin, kmax


    def trace_courbure(self):
        kmin, kmax = self.courbures_pente_nulle()
        kv, kh = self.courbures_pente_non_nulle()
        a, b = self.mnt.shape
        eps = 0.01
        COULEUR = np.zeros((a, b))
        for i in range(a):
            for j in range(b):
                if kv[i][j] == kh[i][j] == np.nan:
                    if kmin[i][j] >= eps and kmax[i][j] >= eps:
                        COULEUR[i][j] = 10
                    elif abs(kmin[i][j]) < eps and kmax[i][j] >= eps:
                        COULEUR[i][j] = 11
                    elif abs(kmin[i][j]) < eps and abs(kmax[i][j]) < eps:
                        COULEUR[i][j] = 12
                    elif kmin[i][j] <= -eps and kmax[i][j] >= eps:
                        COULEUR[i][j] = 13
                    elif kmin[i][j] <= -eps and abs(kmax[i][j]) < eps:
                        COULEUR[i][j] = 14
                    elif kmin[i][j] <= -eps and kmax[i][j] <= -eps:
                        COULEUR[i][j] = 15
                else:
                    if kv[i][j] >= eps and kh[i][j] >= eps:
                        COULEUR[i][j] = 1
                    elif kv[i][j] >= eps and kh[i][j] <= -eps:
                        COULEUR[i][j] = 2
                    elif kv[i][j] >= eps and abs(kh[i][j]) < eps:
                        COULEUR[i][j]=3
                    elif abs(kv[i][j]) < eps and kh[i][j] >= eps:
                        COULEUR[i][j] = 4
                    elif abs(kv[i][j]) < eps and abs(kh[i][j]) < eps:
                        COULEUR[i][j] = 5
                    elif abs(kv[i][j]) < eps and kh[i][j] <= -eps:
                        COULEUR[i][j] = 6
                    elif kv[i][j] <= -eps and kh[i][j] >= eps:
                        COULEUR[i][j] = 7
                    elif kv[i][j] <= -eps and abs(kh[i][j]) < eps:
                        COULEUR[i][j] = 8
                    elif kv[i][j] <= -eps and kh[i][j] <= -eps:
                        COULEUR[i][j] = 9

        couleurs = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#800000', '#008000', '#000080', '#808000', '#800080', '#008080', '#C0C0C0', '#FFA500', '#A52A2A']

        # Créer un colormap avec les couleurs
        cmap = mcolors.ListedColormap(couleurs)

        # Créer la figure et l'axe
        plt.figure(figsize=(8, 6))
        plt.imshow(COULEUR, cmap=cmap, origin='upper')
        plt.colorbar(ticks=range(1, 16), label='Classes de Dikau')
        plt.title('Classification de Dikau des fonds marins')
        plt.xlabel('Longitude (pixel)')
        plt.ylabel('Latitude (pixel)')
        plt.show()