import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LightSource
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Calcul_attribut (object):
    def __init__(self, mnt, name):
        self.mnt = mnt
        self.name = name
        self.prof_min = 0
        self.prof_max = 0
        self.prof_moy = 0
        self.ecart_type = 0
        self.pas = 1

    # ne pas changer, sert à créer l'instance de la classe
    @classmethod
    def from_file(cls, chemin):
        data = np.loadtxt(chemin)
        return cls(data, chemin[4:-4])

    # sert à afficher le graphe du MNT
    def affiche(self):
        # Dimensions des terrains artificiels
        x = np.arange(0, self.mnt.shape[0])
        y = np.arange(0, self.mnt.shape[0])
        X, Y = np.meshgrid(x, y)

        cmap = plt.cm.gist_earth
        img = plt.contourf(X, Y, self.mnt, levels=100, cmap=cmap)
        plt.contour(X, Y, self.mnt, levels=5, colors='black')
        plt.title(self.name)
        plt.colorbar(img, label='Altitude [m]')
        plt.show()

    def affiche_3D(self):
        cmap = plt.cm.cubehelix

        # Dimensions de l'image à afficher
        x_mnt = np.arange(self.mnt.shape[1])
        y_mnt = np.arange(self.mnt.shape[0])
        X_MNT, Y_MNT = np.meshgrid(x_mnt, y_mnt)

        # Normaliser les z pour définir la palette
        norm = Normalize(vmin=np.nanmin(self.mnt), vmax=np.nanmax(self.mnt))
        my_col = cmap(norm(self.mnt))
        # Illumination pour le modèle
        ls = LightSource(azdeg=-45, altdeg=35)
        rgb = ls.shade(self.mnt, cmap=cmap, vert_exag=4, blend_mode='soft')

        # Figure 3D
        fig, axe = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        # Choix du point de vue
        axe.view_init(elev=35., azim=15)
        # Afficher la surface avec illumination
        # Augmenter les valeurs rstride et cstride pour accélérer l'affichage
        surf = axe.plot_surface(X_MNT, Y_MNT, self.mnt, facecolors=rgb, linewidth=0, antialiased=False, rstride=3, cstride=3)
        m = cm.ScalarMappable(cmap=cmap, norm=norm)

        plt.colorbar(m, ax=axe, shrink=.8)
        plt.tight_layout()
        plt.show()

    def calcul_caracteristique_global(self):
        self.prof_min = np.min(self.mnt)
        self.prof_max = np.max(self.mnt)
        self.prof_moy = np.mean(self.mnt)
        self.ecart_type = np.std(self.mnt)

        print(f'Profondeur :\n\t - min : {self.prof_min} \n\t - max : {self.prof_max} \n\t - moyenne : {self.prof_moy} \n\t - ecart type : {self.ecart_type}')

        plt.figure()
        plt.title(f'Histogramme de {self.name}')
        plt.xlabel("Profondeur (m)")
        plt.ylabel("Fréquence")
        plt.hist(self.mnt.ravel(), bins=50, color='steelblue', edgecolor='black')
        plt.show()

    def TPP(self):
        fx = (self.mnt[1:,1:] - self.mnt[1:,:-1]) / self.pas
        fy = (self.mnt[:-1,:-1] - self.mnt[1:,:-1]) / self.pas

        return fx, fy

    def FCN(self):
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, :-2]) / (2 * self.pas)
        fy = (self.mnt[:-2, 1:-1] - self.mnt[2:, 1:-1]) / (2 * self.pas)

        return fx, fy

    def Evans(self):
        x = np.arange(1, 100)
        y = np.arange(1, 100)

        z1 = self.mnt[:-2, :-2]
        z2 = self.mnt[:-2, 1:-1]
        z3 = self.mnt[:-2, 2:]
        z4 = self.mnt[1:-1, :-2]
        z5 = self.mnt[1:-1, 1:-1]
        z6 = self.mnt[1:-1, 2:]
        z7 = self.mnt[2:, :-2]
        z8 = self.mnt[2:, 1:-1]
        z9 = self.mnt[2:, 2:]

        A = (z1+z3+z4+z6+z7+z9)/(6*self.pas**2) - (z2+z5+z8)/(3*self.pas**2)
        B = (z1+z2+z3+z7+z8+z9)/(6*self.pas**2) - (z4+z5+z6)/(3*self.pas**2)
        C = (z3+z7-z1-z9)/(4*self.pas**2)
        D = (z3+z6+z9-z1-z4-z7)/(6*self.pas**2)
        E = (z1+z2+z3-z7-z8-z9)/(6*self.pas**2)
        F = (5*z5+2*(z2+z4+z6+z8)-(z1+z3+z7+z9))/9

        return A*x**2 + B*y**2 * C*x*y + D*x + E*y + F

    def pente(self, fx, fy):
        return np.arctan(np.sqrt(fx ** 2 + fy ** 2)) * 180 / np.pi

    def exposition(self, fx, fy):
        return np.arctan2(-fx, -fy) * 180 / np.pi

    def graphe_exposition_TPP(self):
        fx, fy = self.TPP()
        exposition = self.exposition(fx, fy)

        plt.figure()
        plt.imshow(exposition, origin='lower', cmap='magma_r')
        plt.title(f'Exposition de {self.name}')
        plt.colorbar(label='Exposition [°]')
        plt.show()

    def bpi(self):
        def bpi_carre(self):
            n_lignes = len(self.mnt)
            n_colonnes = len(self.mnt[0])

            print('Nombre de lignes du fichier :', n_lignes)
            print('Nombre de colonnes du fichier :', n_colonnes)

            bpi = np.zeros((n_lignes, n_colonnes))

            for i in range(1, n_lignes - 1):
                for j in range(1, n_colonnes - 1):
                    voisins = [
                        self.mnt[i-1][j-1], self.mnt[i-1][j], self.mnt[i-1][j+1],
                        self.mnt[i][j-1],                   self.mnt[i][j+1],
                        self.mnt[i+1][j-1], self.mnt[i+1][j], self.mnt[i+1][j+1]
                    ]
                    bpi[i][j] = self.mnt[i][j] - (sum(voisins) / 8)

            plt.figure()
            plt.imshow(bpi, origin='lower', cmap='magma_r')
            plt.title(f'BPI de {self.name}')
            plt.colorbar(label='BPI')
        def bpi_cercle(self):
            n_lignes = len(self.mnt)
            n_colonnes = len(self.mnt[0])

            bpi = np.zeros((n_lignes, n_colonnes))

            for i in range(2, n_lignes - 2):
                for j in range(2, n_colonnes - 2):
                    voisins = [
                        self.mnt[i-2][j-1], self.mnt[i-2][j], self.mnt[i-2][j+2],
                        self.mnt[i-1][j-2], self.mnt[i-1][j-1], self.mnt[i-1][j], self.mnt[i-1][j+1], self.mnt[i-1][j+2],
                        self.mnt[i][j-2], self.mnt[i][j-1], self.mnt[i][j+1], self.mnt[i][j+2],
                        self.mnt[i+1][j-2], self.mnt[i+1][j-1], self.mnt[i+1][j], self.mnt[i+1][j+1], self.mnt[i+1][j+2],
                        self.mnt[i+2][j-2], self.mnt[i+2][j], self.mnt[i+2][j+1]
                    ]
                    bpi[i][j] = self.mnt[i][j] - (sum(voisins) / len(voisins))  # Moyenne des 20 voisins

            plt.figure()
            plt.imshow(bpi, origin='lower', cmap='magma_r')
            plt.title(f'BPI cercle de {self.name}')
            plt.colorbar(label='BPI')
        bpi_carre(self)
        bpi_cercle(self)
        plt.show()


if __name__ == '__main__':
    fichier = "double_sin.txt"
    map = Calcul_attribut.from_file("MNT/" + fichier)
    map.affiche()
    map.bpi()
