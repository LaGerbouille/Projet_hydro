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
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, 1:-1]) / self.pas
        fy = (self.mnt[:-2, 1:-1] - self.mnt[1:-1, 1:-1]) / self.pas

        return fx, fy

    def FCN(self):
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, :-2]) / (2 * self.pas)
        fy = (self.mnt[:-2, 1:-1] - self.mnt[2:, 1:-1]) / (2 * self.pas)

        return fx, fy

    def Evans(self):
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

    def pente(self, fx, fy):
        return np.arctan(np.sqrt(fx ** 2 + fy ** 2)) * 180 / np.pi

    def exposition(self, fx, fy):
        return np.arctan2(-fx, -fy) * 180 / np.pi

    def graphe_pente_TPP(self):
        fx, fy = self.TPP()
        pente = self.pente(fx, fy)

        plt.figure()
        plt.imshow(pente, origin='lower', cmap='magma_r')
        plt.title(f'Pente de {self.name}')
        plt.colorbar(label='Pente [°]')

    def graphe_exposition_TPP(self):
        fx, fy = self.TPP()
        exposition = self.exposition(fx, fy)

        plt.figure()
        plt.imshow(exposition, origin='lower', cmap='magma_r')
        plt.title(f'Exposition de {self.name}')
        plt.colorbar(label='Exposition [°]')
        plt.show()


    def deriv_mnt(self):


    def affichage_pente_theorique_et_reelle(self):
        fx, fy = deriv_mnt(X, Y)
        pente_reel = self.pente(fx, fy)
        fx, fy = self.TPP(self.mnt)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(self.mnt)
        pente_fcn = self.pente(fx, fy)

        # Normaliser les palettes entre 0° et pmax
        pmax = 50
        normalize = Normalize(0, pmax)
        cmap = 'cividis_r'

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        im = ax[0].imshow(pente_tpp, origin='lower', cmap=cmap, norm=normalize)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        im = ax[1].imshow(pente_fcn, origin='lower', cmap=cmap, norm=normalize)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        im = ax[2].imshow(pente_reel, origin='lower', cmap=cmap, norm=normalize)
        ax[2].set_title('Réel')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        plt.tight_layout()


if __name__ == '__main__':
    fichier = "plan.txt"
    map = Calcul_attribut.from_file("MNT/" + fichier)
    map.affiche_3D() # affiche le MNT
    map.pente_TPP()
