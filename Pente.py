import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import CenteredNorm
from matplotlib import cm
from matplotlib.colors import LightSource
from scipy.ndimage import convolve, generic_filter
import cmocean


class Pente():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    # -------------------------------------------------- CALCULS ------------------------------------------------------
    def TPP(self, mnt):
        fx = convolve(mnt, np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]) / self.pas, mode='nearest')
        fy = convolve(mnt, np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]) / self.pas, mode='nearest')

        return fx, fy

    def FCN(self, mnt):
        fx = convolve(mnt, np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]]) / (2*self.pas), mode='nearest')
        fy = convolve(mnt, np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]]) / (2*self.pas), mode='nearest')

        return fx, fy

    def Evans(self, mnt):
        fx = convolve(mnt, np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6*self.pas**2), mode='nearest')
        fy = convolve(mnt, np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / (6*self.pas**2), mode='nearest')

        return fx, fy

    def pente(self, fx, fy):
        return np.degrees(np.sqrt(fx ** 2 + fy ** 2)) # modification

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

    def calcul_incertitudes_modele_theorique(self, sigma, N):
        w, h = self.mnt.shape
        pentes = np.zeros((3, N, w, h))

        for methode in range(3):
            for i in range(N):
                bruit_blanc = np.random.normal(loc=0.0, scale=sigma, size=(w, h))
                mnt_bruite = self.mnt + bruit_blanc
                fx, fy = 0, 0
                if methode == 0:
                    fx, fy = self.TPP(mnt_bruite)
                elif methode == 1:
                    fx, fy = self.FCN(mnt_bruite)
                elif methode == 2:
                    fx, fy = self.Evans(mnt_bruite)

                pentes[methode, i] = self.pente(fx, fy)

        moyenne_pente_TPP = np.nanmean(np.nanmean(pentes[0], axis=0))
        ecart_type_pente_TPP = np.nanmean(np.nanstd(pentes[0], axis=0))

        moyenne_pente_FCN = np.nanmean(np.nanmean(pentes[1], axis=0))
        ecart_type_pente_FCN = np.nanmean(np.nanstd(pentes[1], axis=0))

        moyenne_pente_Evans = np.nanmean(np.nanmean(pentes[2], axis=0))
        ecart_type_pente_Evans = np.nanmean(np.nanstd(pentes[2], axis=0))

        return [moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans]

    def calcul_incertitudes_terrain_reel(self, incertitude, N):
        pentes = np.zeros((3, N, incertitude.shape[0], incertitude.shape[1]))

        for methode in range(3):
            for i in range(N):
                bruit_blanc = np.random.normal(loc=0.0, scale=incertitude)
                mnt_bruite = self.mnt + bruit_blanc
                fx, fy = 0, 0
                if methode == 0:
                    fx, fy = self.TPP(mnt_bruite)
                elif methode == 1:
                    fx, fy = self.FCN(mnt_bruite)
                elif methode == 2:
                    fx, fy = self.Evans(mnt_bruite)

                pentes[methode, i] = self.pente(fx, fy)

        moyenne_pente_TPP = np.nanmean(np.nanmean(pentes[0], axis=0))
        ecart_type_pente_TPP = np.nanmean(np.nanstd(pentes[0], axis=0))

        moyenne_pente_FCN = np.nanmean(np.nanmean(pentes[1], axis=0))
        ecart_type_pente_FCN = np.nanmean(np.nanstd(pentes[1], axis=0))

        moyenne_pente_Evans = np.nanmean(np.nanmean(pentes[2], axis=0))
        ecart_type_pente_Evans = np.nanmean(np.nanstd(pentes[2], axis=0))

        return [moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans]

    def coords_grille(self, r=3):
        """Retourne les coordonnées xi, yi d'une grille carrée centrée en (0,0)"""
        s = np.arange(-(r // 2), r // 2 + 1)
        yi, xi = np.meshgrid(s, -s)
        return xi.flatten(), yi.flatten()

    def moindres_carres(self, r=3):
        return generic_filter(self.mnt, lambda values: self.fit_surface_quadrique(values, r), size=(r, r))

    def fit_surface_quadrique(self, values, r=3):
        """Fonction appliquée localement par generic_filter"""
        zi = values.reshape((-1, 1))
        xi, yi = self.coords_grille(r)
        xi = xi.reshape((-1, 1))
        yi = yi.reshape((-1, 1))
        un = np.ones_like(xi)

        M = np.hstack([xi ** 2, yi ** 2, xi * yi, xi, yi, un])

        # fonction de numpy.linalg retournant la solution de l'équation : a @ x = b
        Xhat = la.lstsq(M, zi, rcond=None)[0].flatten()
        D = Xhat[3]
        E = Xhat[4]
        return self.pente(E, -D)

    def scatter_pente(self, pente_x, pente_y):
        # Aplatir et retirer les NaN
        x = pente_x.flatten()
        y = pente_y.flatten()
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        return x, y

    # ---------------------------------------------------- AFFICHAGE --------------------------------------------------
    #Q2
    def affichage_pente(self):
        fx, fy = self.TPP(self.mnt)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(self.mnt)
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans(self.mnt)
        pente_evans = self.pente(fx, fy)

        pente_moindres_carres = self.moindres_carres(3)
        diff = self.moindres_carres(3) - pente_evans
        diff[np.isnan(diff) | (diff < 1e-10)] = 0

        print("Différence pente méthodes Moindres carrés - Evans. Grille 3 x 3 : \n", diff)

        cmap = 'cubehelix_r'

        plt.imshow(pente_tpp, origin='lower', cmap=cmap)
        plt.title('TPP')
        plt.colorbar(label='Pente[°]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.suptitle(self.name)
        plt.show()

        plt.imshow(pente_fcn, origin='lower', cmap=cmap)
        plt.title('FCN')
        plt.colorbar(label='Pente[°]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.suptitle(self.name)
        plt.show()

        plt.imshow(pente_evans, origin='lower', cmap=cmap)
        plt.title('Evans')
        plt.colorbar(label='Pente[°]')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.suptitle(self.name)
        plt.show()

        plt.imshow(pente_moindres_carres, origin='lower', cmap=cmap)
        plt.title('Moindres Carrés')
        plt.colorbar(label='Pente[°]')
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.suptitle(self.name)
        plt.show()

    #Q3
    def affichage_3D(self):
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

        axe.set_xlabel("X [pixels]")
        axe.set_ylabel("Y [pixels]")
        axe.set_zlabel("Altitude [m]")

        plt.title(self.name)
        plt.colorbar(m, label='Pente[°]', ax=axe, shrink=.8)
        plt.tight_layout()
        plt.show()

    #Q4
    def affichage_comparaison_pente_theorique(self):
        fx, fy = self.deriv_terrain_theorique(self.name)
        pente_reel = self.pente(fx, fy)
        fx, fy = self.TPP(self.mnt)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(self.mnt)
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans(self.mnt)
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

        print(f'TPP : \n\t - moyenne : {moyenne_tpp} \n\t - ecart type : {ecart_type_tpp}')
        print(f'FCN : \n\t - moyenne : {moyenne_fcn} \n\t - ecart type : {ecart_type_fcn}')
        print(f'Evans : \n\t - moyenne : {moyenne_evans} \n\t - ecart type : {ecart_type_evans}')

        normalize3 = CenteredNorm(0)

        cmap = 'seismic'

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

        plt.suptitle(f'Normalisation partagée. Comparaison des méthodes de calcul de pente.')
        plt.show()

        # Diagrammes croisés
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        x, y = self.scatter_pente(pente_tpp, pente_fcn)

        ax[0].set_title("Pente TPP vs FCN")
        ax[0].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[0].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[0].legend()

        x, y = self.scatter_pente(pente_fcn, pente_evans)

        ax[1].set_title("Pente FCN vs Evans")
        ax[1].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[1].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[1].legend()

        x, y = self.scatter_pente(pente_evans, pente_tpp)

        ax[2].set_title("Pente Evans vs TPP")
        ax[2].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[2].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[2].legend()

        plt.suptitle('Ecarts entre méthodes')
        plt.show()

    def affichage_comparaison_pente_reel(self):
        fx, fy = self.TPP(self.mnt)
        pente_tpp = self.pente(fx, fy)
        fx, fy = self.FCN(self.mnt)
        pente_fcn = self.pente(fx, fy)
        fx, fy = self.Evans(self.mnt)
        pente_evans = self.pente(fx, fy)

        erreur_tpp_fcn = pente_tpp - pente_fcn
        erreur_fcn_evans = pente_fcn - pente_evans
        erreur_evans_tpp = pente_evans - pente_tpp

        moyenne_tpp_fcn = np.nanmean(erreur_tpp_fcn)
        moyenne_fcn_evans = np.nanmean(erreur_fcn_evans)
        moyenne_evans_tpp = np.nanmean(erreur_evans_tpp)

        ecart_type_tpp_fcn = np.nanstd(erreur_tpp_fcn)
        ecart_type_fcn_evans = np.nanstd(erreur_fcn_evans)
        ecart_type_evans_tpp = np.nanstd(erreur_evans_tpp)

        print(f'TPP : \n\t - moyenne : {moyenne_tpp_fcn} \n\t - ecart type : {ecart_type_tpp_fcn}')
        print(f'FCN : \n\t - moyenne : {moyenne_fcn_evans} \n\t - ecart type : {ecart_type_fcn_evans}')
        print(f'Evans : \n\t - moyenne : {moyenne_evans_tpp} \n\t - ecart type : {ecart_type_evans_tpp}')

        normalize3 = CenteredNorm(0)

        cmap = 'seismic'

        plt.imshow(erreur_tpp_fcn, origin='lower', cmap=cmap, norm=normalize3)
        plt.title('TPP - FCN')
        plt.colorbar(label='Erreur[°]')
        plt.show()

        plt.imshow(erreur_fcn_evans, origin='lower', cmap=cmap, norm=normalize3)
        plt.title('FCN - Evans')
        plt.colorbar(label='Erreur[°]')
        plt.show()

        plt.imshow(erreur_evans_tpp, origin='lower', cmap=cmap, norm=normalize3)
        plt.title('Evans - TPP')
        plt.colorbar(label='Erreur[°]')
        plt.show()

        # Diagrammes croisés
        fig, ax = plt.subplots(1, 3, figsize=(12, 5))

        x, y = self.scatter_pente(pente_tpp, pente_fcn)

        ax[0].set_title("Pente TPP vs FCN")
        ax[0].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[0].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[0].set_xlabel("Pente TPP [°]")
        ax[0].set_ylabel("Pente FCN [°]")
        ax[0].legend()

        x, y = self.scatter_pente(pente_fcn, pente_evans)

        ax[1].set_title("Pente FCN vs Evans")
        ax[1].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[1].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[1].set_xlabel("Pente FCN [°]")
        ax[1].set_ylabel("Pente Evans [°]")
        ax[1].legend()

        x, y = self.scatter_pente(pente_evans, pente_tpp)

        ax[2].set_title("Pente Evans vs TPP")
        ax[2].scatter(x, y, s=5, alpha=0.5, c='blue', edgecolors='none')
        ax[2].plot([x.min(), x.max()], [x.min(), x.max()], 'r--', label='y = x')  # ligne diagonale
        ax[2].set_xlabel("Pente Evans [°]")
        ax[2].set_ylabel("Pente TPP [°]")
        ax[2].legend()

        plt.suptitle('Ecarts entre méthodes')
        plt.show()

    #Q5
    def affichage_incertitudes_modele_artificiel_fonction_ecart_type(self, list_sigma, N):
        moy_TPP = np.zeros(len(list_sigma))
        std_TPP = np.zeros(len(list_sigma))
        moy_FCN = np.zeros(len(list_sigma))
        std_FCN = np.zeros(len(list_sigma))
        moy_Evans = np.zeros(len(list_sigma))
        std_Evans = np.zeros(len(list_sigma))

        for i, sigma in enumerate(list_sigma):
            moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes_modele_theorique(sigma, N)
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
        ax[0].set_xlabel("Écart-type du bruit σ")
        ax[0].set_ylabel("Pente moyenne [°]")
        ax[0].legend()

        ax[1].plot(list_sigma, std_TPP, label='TPP')
        ax[1].plot(list_sigma, std_FCN, label='FCN')
        ax[1].plot(list_sigma, std_Evans, label='Evans')
        ax[1].set_title(f'Ecart-type pour N = {N}, σ dans [{list_sigma[0]}, {list_sigma[-1]}]')
        ax[1].set_xlabel("Écart-type du bruit σ")
        ax[1].set_ylabel("Écart-type de la pente [°]")
        ax[1].legend()

        plt.show()

    def affichage_incertitudes_terrain_reel(self, incertitude, N):
        moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes_terrain_reel(incertitude, N)

        print(
            f'TPP : \n\t - moyenne : {moyenne_pente_TPP} \n\t - ecart type : {ecart_type_pente_TPP}')
        print(
            f'FCN : \n\t - moyenne : {moyenne_pente_FCN} \n\t - ecart type : {ecart_type_pente_FCN}')
        print(
            f'Evans : \n\t - moyenne : {moyenne_pente_Evans} \n\t - ecart type : {ecart_type_pente_Evans}')

    #Q6
    def affichage_incertitude_monte_carlo_artificiel(self, sigma, N):
        moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes_modele_theorique(sigma, N)

        print(
            f'TPP (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_TPP} \n\t - ecart type : {ecart_type_pente_TPP}')
        print(
            f'FCN (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_FCN} \n\t - ecart type : {ecart_type_pente_FCN}')
        print(
            f'Evans (N = {N}, σ = {sigma}) : \n\t - moyenne : {moyenne_pente_Evans} \n\t - ecart type : {ecart_type_pente_Evans}')

    def affichage_incertitude_monte_carlo_reel(self, incert, N):
        moyenne_pente_TPP, ecart_type_pente_TPP, moyenne_pente_FCN, ecart_type_pente_FCN, moyenne_pente_Evans, ecart_type_pente_Evans = self.calcul_incertitudes_terrain_reel(incert, N)

        print(
            f'TPP (N = {N}) : \n\t - moyenne : {moyenne_pente_TPP} \n\t - ecart type : {ecart_type_pente_TPP}')
        print(
            f'FCN (N = {N}) : \n\t - moyenne : {moyenne_pente_FCN} \n\t - ecart type : {ecart_type_pente_FCN}')
        print(
            f'Evans (N = {N}) : \n\t - moyenne : {moyenne_pente_Evans} \n\t - ecart type : {ecart_type_pente_Evans}')