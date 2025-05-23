import numpy as np

from BPI import *
from Pente import *
from Rugosite import *
from Courbure import *

# sert à afficher le graphe 2D du MNT
def affiche_2D():
    # Dimensions des terrains artificiels
    x = np.arange(0, mnt.shape[0])
    y = np.arange(0, mnt.shape[0])
    X, Y = np.meshgrid(x, y)

    cmap = plt.cm.gist_earth
    img = plt.contourf(X, Y, mnt, levels=100, cmap=cmap)
    plt.contour(X, Y, mnt, levels=5, colors='black')
    plt.title(name)
    plt.colorbar(img, label='Altitude [m]')
    plt.show()

def affiche_caracteristique_global():
        prof_min = np.min(mnt)
        prof_max = np.max(mnt)
        prof_moy = np.mean(mnt)
        ecart_type = np.std(mnt)

        print(
            f'Profondeur :\n\t - min : {prof_min} \n\t - max : {prof_max} \n\t - moyenne : {prof_moy} \n\t - ecart type : {ecart_type}')

        plt.figure()
        plt.title(f'Histogramme de {name}')
        plt.xlabel("Profondeur (m)")
        plt.ylabel("Fréquence")
        plt.hist(mnt.ravel(), bins=50, color='steelblue', edgecolor='black')
        plt.show()

if __name__ == '__main__':
    fichier = "z_Zone1_8m.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    pas = 1
    name = fichier[:-4]

    bpi = BPI(mnt, pas, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)
    courbure = Courbure(mnt, pas, name)

    # pente.affichage_incertitudes_fonction_ecart_type(np.arange(0, 7, 0.5), 1000)

    pente.affiche_3D()