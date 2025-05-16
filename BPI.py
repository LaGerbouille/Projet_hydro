import numpy as np
import matplotlib.pyplot as plt

class BPI():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def bpi_carre(self):
        n_lignes = len(self.mnt)
        n_colonnes = len(self.mnt[0])

        bpi = np.zeros((n_lignes, n_colonnes))

        for i in range(1, n_lignes - 1):
            for j in range(1, n_colonnes - 1):
                voisins = [
                    self.mnt[i - 1][j - 1], self.mnt[i - 1][j], self.mnt[i - 1][j + 1],
                    self.mnt[i][j - 1], self.mnt[i][j + 1],
                    self.mnt[i + 1][j - 1], self.mnt[i + 1][j], self.mnt[i + 1][j + 1]
                ]
                bpi[i][j] = self.mnt[i][j] - (sum(voisins) / 8)
        return bpi


    def bpi_cercle(self):
        n_lignes = len(self.mnt)
        n_colonnes = len(self.mnt[0])

        bpi = np.zeros((n_lignes, n_colonnes))

        for i in range(2, n_lignes - 2):
            for j in range(2, n_colonnes - 2):
                voisins = [
                    self.mnt[i - 2][j - 1], self.mnt[i - 2][j], self.mnt[i - 2][j + 2],
                    self.mnt[i - 1][j - 2], self.mnt[i - 1][j - 1], self.mnt[i - 1][j], self.mnt[i - 1][j + 1],
                    self.mnt[i - 1][j + 2],
                    self.mnt[i][j - 2], self.mnt[i][j - 1], self.mnt[i][j + 1], self.mnt[i][j + 2],
                    self.mnt[i + 1][j - 2], self.mnt[i + 1][j - 1], self.mnt[i + 1][j], self.mnt[i + 1][j + 1],
                    self.mnt[i + 1][j + 2],
                    self.mnt[i + 2][j - 2], self.mnt[i + 2][j], self.mnt[i + 2][j + 1]
                ]
                bpi[i][j] = self.mnt[i][j] - (sum(voisins) / len(voisins))  # Moyenne des 20 voisins
        return bpi


    def affichage_bpi_cercle(self):
        plt.figure()
        plt.imshow(self.bpi_cercle(), origin='lower', cmap='magma_r')
        plt.title(f'BPI cercle de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()

    def affichage_bpi_carre(self):

        plt.figure()
        plt.imshow(self.bpi_carre(), origin='lower', cmap='magma_r')
        plt.title(f'BPI de {self.name}')
        plt.colorbar(label='BPI')
        plt.show()
