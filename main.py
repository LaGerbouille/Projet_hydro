import numpy as np
import matplotlib.pyplot as plt

class Calcul_attribut (object):
    def __init__(self, mnt, name):
        self.mnt = mnt
        self.name = name

    @classmethod
    def mnt(cls, fichier):
        data = np.loadtxt(fichier)
        return cls(data, fichier[4:-4])

    def affiche(self):
        # Dimensions des terrains artificiels
        x = np.arange(0, 101)
        y = np.arange(0, 101)
        X, Y = np.meshgrid(x, y)

        cmap = plt.cm.gist_earth
        img = plt.contourf(X, Y, self.mnt, levels=100, cmap=cmap)
        plt.title(self.name)
        plt.colorbar(img, label='Altitude [m]')
        plt.show()

if __name__ == '__main__':
    fichier = "plan.txt"
    map = Calcul_attribut.mnt("MNT/" + fichier)
    map.affiche()
