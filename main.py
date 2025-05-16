from matplotlib import cm
from matplotlib.colors import LightSource
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

# affiche le MNT en 3D
def affiche_3D():
    cmap = plt.cm.cubehelix

    # Dimensions de l'image à afficher
    x_mnt = np.arange(mnt.shape[1])
    y_mnt = np.arange(mnt.shape[0])
    X_MNT, Y_MNT = np.meshgrid(x_mnt, y_mnt)

    # Normaliser les z pour définir la palette
    norm = Normalize(vmin=np.nanmin(mnt), vmax=np.nanmax(mnt))
    my_col = cmap(norm(mnt))
    # Illumination pour le modèle
    ls = LightSource(azdeg=-45, altdeg=35)
    rgb = ls.shade(mnt, cmap=cmap, vert_exag=4, blend_mode='soft')

    # Figure 3D
    fig, axe = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "3d"})
    # Choix du point de vue
    axe.view_init(elev=35., azim=15)
    # Afficher la surface avec illumination
    # Augmenter les valeurs rstride et cstride pour accélérer l'affichage
    surf = axe.plot_surface(X_MNT, Y_MNT, mnt, facecolors=rgb, linewidth=0, antialiased=False, rstride=3,
                            cstride=3)
    m = cm.ScalarMappable(cmap=cmap, norm=norm)

    plt.title(name)
    plt.colorbar(m, ax=axe, shrink=.8)
    plt.tight_layout()
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
    fichier = "double_sin.txt"
    mnt = np.loadtxt("MNT/" + fichier)
    pas = 8
    pas_bpi=1
    name = fichier[:-4]

    bpi = BPI(mnt, pas_bpi, name)
    pente = Pente(mnt, pas, name)
    rugosite = Rugosite(mnt, pas, name)
    courbure = Courbure(mnt, pas, name)
    bpi.affichage_bpi_carre()
    bpi.affichage_bpi_cercle()
    
