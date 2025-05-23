import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Pente():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

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

        # A = (z1 + z3 + z4 + z6 + z7 + z9) / (6 * self.pas ** 2) - (z2 + z5 + z8) / (3 * self.pas ** 2)
        # B = (z1 + z2 + z3 + z7 + z8 + z9) / (6 * self.pas ** 2) - (z4 + z5 + z6) / (3 * self.pas ** 2)
        # C = (z3 + z7 - z1 - z9) / (4 * self.pas ** 2)
        D = (z3 + z6 + z9 - z1 - z4 - z7) / (6 * self.pas ** 2)
        E = (z1 + z2 + z3 - z7 - z8 - z9) / (6 * self.pas ** 2)

        fx = D
        fy = E

        return fx, fy


    def pente(self, fx, fy):
        return np.arctan(np.sqrt(fx ** 2 + fy ** 2)) * 180 / np.pi


    def deriv_mnt(self):
        fy, fx = np.gradient(self.mnt, self.pas, self.pas)
        return fx, fy


    def exposition(self, fx, fy):
        return np.arctan2(-fx, -fy) * 180 / np.pi


    def affichage_pente(self):
        fx, fy = self.TPP()
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN()
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans()
        pente_evans = self.pente(fx, fy)

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

        im = ax[2].imshow(pente_evans, origin='lower', cmap=cmap, norm=normalize)
        ax[2].set_title('Evans')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        for a in ax:
            a.set_xlabel("X")
            a.set_ylabel("Y")

        plt.suptitle(self.name)
        plt.tight_layout()
        plt.show()

    def affichage_differences_pente(self):
        fx, fy = self.TPP()
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN()
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans()
        pente_evans = self.pente(fx, fy)
        erreur_tpp = pente_tpp - pente_reel
        erreur_fcn = pente_fcn - pente_reel
        erreur_evans = pente_evans - pente_reel

        normalize1 = CenteredNorm(0)
        normalize2 = CenteredNorm(0)
        normalize3 = CenteredNorm(0)
        cmap = 'seismic'

        # Normalisation par figure
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(erreur_tpp, origin='lower', cmap=cmap, norm=normalize1)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Erreur[°]', cax=cax)

        im = ax[1].imshow(erreur_fcn, origin='lower', cmap=cmap, norm=normalize2)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Erreur[°]', cax=cax)

        plt.suptitle('Normalisations différentes')

        # Normalisation partagée
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        im = ax[0].imshow(erreur_tpp, origin='lower', cmap=cmap, norm=normalize3)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Erreur[°]', cax=cax)

        im = ax[1].imshow(erreur_fcn, origin='lower', cmap=cmap, norm=normalize3)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Erreur[°]', cax=cax)

        plt.suptitle('Normalisation partagée')

    def affichage_exposition(self):
        fx, fy = self.TPP()
        exposition_tpp = self.exposition(fx, fy)
        fx, fy = self.FCN()
        exposition_fcn = self.exposition(fx, fy)
        fx, fy = self.Evans()
        exposition_evans = self.exposition(fx, fy)

        # Normaliser les palettes entre 0° et pmax
        pmax = 50
        normalize = Normalize(0, pmax)
        cmap = 'twilight'

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        im = ax[0].imshow(exposition_tpp, origin='lower', cmap=cmap, norm=normalize)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Exposition[°]', cax=cax)

        im = ax[1].imshow(exposition_fcn, origin='lower', cmap=cmap, norm=normalize)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Exposition[°]', cax=cax)

        im = ax[2].imshow(exposition_evans, origin='lower', cmap=cmap, norm=normalize)
        ax[2].set_title('Evans')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Exposition[°]', cax=cax)

        for a in ax:
            a.set_xlabel("X")
            a.set_ylabel("Y")

        plt.suptitle(self.name)
        plt.tight_layout()
        plt.show()
