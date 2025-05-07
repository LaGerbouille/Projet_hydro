import numpy as np
import matplotlib.pyplot as plt

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
        x = np.arange(0, 101)
        y = np.arange(0, 101)
        X, Y = np.meshgrid(x, y)

        cmap = plt.cm.gist_earth
        img = plt.contourf(X, Y, self.mnt, levels=100, cmap=cmap)
        plt.contour(X, Y, self.mnt, levels=5, colors='black')
        plt.title(self.name)
        plt.colorbar(img, label='Altitude [m]')
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

    def pente_TPP(self):
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, 1:-1]) / self.pas
        fy = (self.mnt[:-2, 1:-1] - self.mnt[1:-1, 1:-1]) / self.pas

        # pente
        pente = np.arctan(np.sqrt(fx ** 2 + fy ** 2)) * 180 / np.pi

        plt.figure()
        plt.imshow(pente, origin='lower', cmap='magma_r')
        plt.title(f'Pente de {self.name}')
        plt.colorbar(label='Pente [°]')

        # exposition
        exposition = np.arctan2(-fx, -fy) * 180 / np.pi

        plt.figure()
        plt.imshow(exposition, origin='lower', cmap='magma_r')
        plt.title(f'Exposition de {self.name}')
        plt.colorbar(label='Exposition [°]')
        plt.show()


if __name__ == '__main__':
    fichier = "plan.txt"
    map = Calcul_attribut.from_file("MNT/" + fichier)
    map.affiche()
    map.pente_TPP()
