import numpy as np

from BPI import *
from Pente import *
from Rugosite import *
from Courbure import *

def trace_classification_courbures(courbure):
    kmin, kmax = courbure.courbures_pente_nulle()
    kv, kh = courbure.courbures_pente_non_nulle()
    a, b = courbure.mnt.shape
    eps = 0.005

    COULEUR = np.zeros((a, b))
    for i in range(a):
        for j in range(b):
            if np.isnan(kv[i][j]) and np.isnan(kh[i][j]):
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
                    COULEUR[i][j] = 3
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

    cmap = plt.get_cmap('tab20', 15)
    plt.imshow(COULEUR, cmap=cmap, origin='lower')
    cbar = plt.colorbar(ticks=range(1, 16), label='Classes de Dikau')
    cbar.set_ticklabels(['nose', 'shoulder slope', 'hollow shoulder', 'spur', 'planar slope', 'hollow', 'spur foot', 'foot slope', 'hollow foot', 'peak', 'ridge', 'plain', 'saddle', 'channel', 'pit'])
    plt.title('Classification de Dikau des fonds marins')
    plt.xlabel('Longitude (pixel)')
    plt.ylabel('Latitude (pixel)')
    plt.show()


if __name__ == '__main__':
    fichier = "z_Zone1_8m.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    mnt2 = gaussian_filter(mnt, 3, mode='constant', cval=0.0)
    pas = 1
    name = fichier[:-4]

    bpi = BPI(mnt, pas, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)
    courbure = Courbure(mnt2, pas, name)

    trace_classification_courbures(courbure)