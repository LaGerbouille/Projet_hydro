import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import CenteredNorm
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.ndimage import convolve
import cmocean


class Pente():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    # -------------------------------------------------- CALCULS ------------------------------------------------------
    def TPP(self, mnt):
        fx = convolve(mnt, np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]) / self.pas, mode='constant', cval=np.nan)
        fy = convolve(mnt, np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]) / self.pas, mode='constant', cval=np.nan)

        return fx, fy

    def FCN(self, mnt):
        fx = convolve(mnt, np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2*self.pas), mode='constant', cval=np.nan)
        fy = convolve(mnt, np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]) / (2*self.pas), mode='constant', cval=np.nan)

        return fx, fy

    def Evans(self, mnt):
        fx = convolve(mnt, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6*self.pas**2), mode='constant', cval=np.nan)
        fy = convolve(mnt, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / (6*self.pas**2), mode='constant', cval=np.nan)

        return fx, fy

    def pente(self, fx, fy):
        return np.degrees(np.arctan(np.sqrt(fx ** 2 + fy ** 2)))

    def exposition(self, fx, fy):
        return np.degrees(np.arctan2(-fx, -fy))

    def deriv_terrain_theorique(self, name):
        x = np.arange(0, 101)
        y = np.arange(0, 101)
        X, Y = np.meshgrid(x, y)

        fx, fy = 0, 0

        if name == "plan":
            fx = 0.07
            fy = 0.1
        elif name == "double_sin":
            fx = 0.5 * np.cos(X / 10 + 3 * np.sin(Y / 20))
            fy = (3 / 4) * np.cos(X / 10 + 3 * np.sin(Y / 20)) * np.cos(Y / 20) + (2 / 5) * np.cos(Y / 5)
        elif name == "sin_card":
            d = np.sqrt((X - 40)**2 + (Y - 50)**2)
            fx = 10 * (0.1 * d * np.cos(0.1 * d) - np.sin(0.1 * d)) / ((0.1 * d)**2) * (X - 40) / d
            fy = 10 * (0.1 * d * np.cos(0.1 * d) - np.sin(0.1 * d)) / ((0.1 * d)**2) * (Y - 50) / d

        return fx, fy

    def calcul_incertitudes(self, sigma, N):
        w, h = self.mnt.shape
        pentes = np.zeros((3, N, w, h))
        for methode in range(3):
            for i in range(N):
                bruit_blanc = np.random.normal(loc=0.0, scale=sigma, size=(w, h))
                mnt_bruite = self.mnt + bruit_blanc

                if methode == 0:
                    fx, fy = self.TPP(mnt_bruite)
                elif methode == 1:
                    fx, fy = self.FCN(mnt_bruite)
                elif methode == 2:
                    fx, fy = self.Evans(mnt_bruite)

                pentes[methode, i] = self.pente(fx, fy)

        moyenne_pente_TPP = np.nanmean(np.nanmean(pentes[0], axis=0))
        ecart_type_pente_TPP = np.nanstd(np.nanstd(pentes[0], axis=0))

        moyenne_pente_FCN = np.nanmean(np.nanmean(pentes[1], axis=0))
        ecart_type_pente_FCN = np.nanstd(np.nanstd(pentes[1], axis=0))

        moyenne_pente_Evans = np.nanmean(np.nanmean(pentes[2], axis=0))
        ecart_type_pente_Evans = np.nanstd(np.nanstd(pentes[2], axis=0))

        return [moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans]

    # ---------------------------------------------------- AFFICHAGE --------------------------------------------------
    def affichage_pente(self):
        fx, fy = self.TPP(self.mnt)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(self.mnt)
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans(self.mnt)
        pente_evans = self.pente(fx, fy)

        cmap = 'cividis_r'

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        im = ax[0].imshow(pente_tpp, origin='lower', cmap=cmap)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        im = ax[1].imshow(pente_fcn, origin='lower', cmap=cmap)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Pente[°]', cax=cax)

        im = ax[2].imshow(pente_evans, origin='lower', cmap=cmap)
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

    def affichage_differences_pente(self, sigma):
        bruit_blanc = np.random.normal(loc=0.0, scale=sigma, size=(101, 101))

        mnt_bruite = self.mnt + bruit_blanc

        fx, fy = self.deriv_terrain_theorique(self.name)
        pente_reel = self.pente(fx, fy)
        fx, fy = self.TPP(mnt_bruite)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(mnt_bruite)
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans(mnt_bruite)
        pente_evans = self.pente(fx, fy)

        erreur_tpp = pente_tpp - pente_reel
        erreur_fcn = pente_fcn - pente_reel
        erreur_evans = pente_evans - pente_reel

        moyenne_tpp = np.nanmean(erreur_tpp)
        moyenne_fcn = np.nanmean(erreur_fcn)
        moyenne_evans = np.nanmean(erreur_evans)

        ecart_type_tpp = np.nanstd(erreur_tpp)
        ecart_type_fcn = np.nanstd(erreur_fcn)
        ecart_type_evans = np.nanstd(erreur_evans)

        print(f'TPP (σ = {sigma}) : \n\t - moyenne : {moyenne_tpp} \n\t - ecart type : {ecart_type_tpp}')
        print(f'FCN (σ = {sigma}) : \n\t - moyenne : {moyenne_fcn} \n\t - ecart type : {ecart_type_fcn}')
        print(f'Evans (σ = {sigma}) : \n\t - moyenne : {moyenne_evans} \n\t - ecart type : {ecart_type_evans}')

        normalize1 = CenteredNorm(0)
        normalize2 = CenteredNorm(0)
        normalize3 = CenteredNorm(0)
        cmap = 'seismic'

        # Normalisation par figure
        # fig, ax = plt.subplots(1, 3, figsize=(12, 5))
        # im = ax[0].imshow(erreur_tpp, origin='lower', cmap=cmap, norm=normalize1)
        # ax[0].set_title('TPP')
        # divider = make_axes_locatable(ax[0])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, label='Erreur[°]', cax=cax)
        #
        # im = ax[1].imshow(erreur_fcn, origin='lower', cmap=cmap, norm=normalize2)
        # ax[1].set_title('FCN')
        # divider = make_axes_locatable(ax[1])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, label='Erreur[°]', cax=cax)
        #
        # im = ax[2].imshow(erreur_evans, origin='lower', cmap=cmap, norm=normalize2)
        # ax[2].set_title('Evans')
        # divider = make_axes_locatable(ax[2])
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # plt.colorbar(im, label='Erreur[°]', cax=cax)
        #
        # plt.suptitle(f'Normalisations différentes. σ = {sigma}')

        # Normalisation partagée
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

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

        im = ax[2].imshow(erreur_evans, origin='lower', cmap=cmap, norm=normalize3)
        ax[2].set_title('Evans')
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Erreur[°]', cax=cax)

        plt.suptitle(f'Normalisation partagée. σ = {sigma}')
        plt.show()

    def affichage_incertitude_monte_carlo(self, sigma, N):
        moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes(sigma, N)

        print(f'TPP (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_TPP} \n\t - ecart type : {ecart_type_pente_TPP}')
        print(f'FCN (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_FCN} \n\t - ecart type : {ecart_type_pente_FCN}')
        print(f'Evans (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_Evans} \n\t - ecart type : {ecart_type_pente_Evans}')

    def affichage_incertitudes_fonction_ecart_type(self, list_sigma, N):
        moy_TPP = np.zeros(len(list_sigma))
        std_TPP = np.zeros(len(list_sigma))
        moy_FCN = np.zeros(len(list_sigma))
        std_FCN = np.zeros(len(list_sigma))
        moy_Evans = np.zeros(len(list_sigma))
        std_Evans = np.zeros(len(list_sigma))

        for i, sigma in enumerate(list_sigma):
            moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes(sigma, N)
            moy_TPP[i] = moyenne_pente_TPP
            std_TPP[i] = ecart_type_pente_TPP
            moy_FCN[i] = moyenne_pente_FCN
            std_FCN[i] = ecart_type_pente_FCN
            moy_Evans[i] = moyenne_pente_Evans
            std_Evans[i] = ecart_type_pente_Evans

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].plot(list_sigma, moy_TPP, label='TPP')
        ax[0].plot(list_sigma, moy_FCN, label='FCN')
        ax[0].plot(list_sigma, moy_Evans, label='Evans')
        ax[0].set_title(f'Moyenne pour N = {N}, σ dans [{list_sigma[0]}, {list_sigma[-1]}]')
        ax[0].legend()

        ax[1].plot(list_sigma, std_TPP, label='TPP')
        ax[1].plot(list_sigma, std_FCN, label='FCN')
        ax[1].plot(list_sigma, std_Evans, label='Evans')
        ax[1].set_title(f'Ecart-type pour N = {N}, σ dans [{list_sigma[0]}, {list_sigma[-1]}]')
        ax[1].legend()

        plt.show()

    def affichage_exposition(self):
        fx, fy = self.TPP(self.mnt)
        exposition_tpp = self.exposition(fx, fy)
        fx, fy = self.FCN(self.mnt)
        exposition_fcn = self.exposition(fx, fy)
        fx, fy = self.Evans(self.mnt)
        exposition_evans = self.exposition(fx, fy)

        cmap = 'twilight_shifted'

        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        im = ax[0].imshow(exposition_tpp, origin='lower', cmap=cmap)
        ax[0].set_title('TPP')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Exposition[°]', cax=cax)

        im = ax[1].imshow(exposition_fcn, origin='lower', cmap=cmap)
        ax[1].set_title('FCN')
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, label='Exposition[°]', cax=cax)

        im = ax[2].imshow(exposition_evans, origin='lower', cmap=cmap)
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

    def affiche_3D(self):
        cmap = cmocean.cm.tarn

        # Dimensions de l'image à afficher
        x_mnt = np.arange(self.mnt.shape[1])
        y_mnt = np.arange(self.mnt.shape[0])
        X_MNT, Y_MNT = np.meshgrid(x_mnt, y_mnt)

        fx, fy = self.TPP(self.mnt)
        pente = self.pente(fx, fy)
        pente = np.where(np.isnan(pente), 0, pente)

        # Normaliser les pentes pour définir la palette
        norm_pente = Normalize(vmin=np.nanmin(pente), vmax=np.nanmax(pente))
        my_col = cmap(norm_pente(pente))
        # Illumination pour le modèle
        ls = LightSource(azdeg=-45, altdeg=35)

        # Figure 3D
        fig, axe = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
        # Choix du point de vue
        axe.view_init(elev=35., azim=15)
        # Afficher la surface avec illumination
        # Augmenter les valeurs rstride et cstride pour accélérer l'affichage
        surf = axe.plot_surface(X_MNT, Y_MNT, self.mnt, facecolors=my_col, linewidth=0, antialiased=False, rstride=1,
                                cstride=1)
        m = cm.ScalarMappable(cmap=cmap, norm=norm_pente)

        plt.title(self.name)
        plt.colorbar(m, label='Pente[°]', ax=axe, shrink=.8)
        plt.tight_layout()
        plt.show()