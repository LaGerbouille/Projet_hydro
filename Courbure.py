import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Courbure():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def courbures_pente_non_nulle(self):
        grilleA = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]]) / (6 * self.pas ** 2)
        A = convolve(self.mnt, grilleA, mode='constant', cval=0.0)
        grilleB = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]]) / (6 * self.pas ** 2)
        B = convolve(self.mnt, grilleB, mode='constant', cval=0.0)
        grilleC = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * self.pas ** 2)
        C = convolve(self.mnt, grilleC, mode='constant', cval=0.0)
        grilleD = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / (6 * self.pas ** 2)
        D = convolve(self.mnt, grilleD, mode='constant', cval=0.0)
        grilleE = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]) / (6 * self.pas ** 2)
        E = convolve(self.mnt, grilleE, mode='constant', cval=0.0)
        grilleF = np.array([[-1, 2, -1], [2, 5, 2], [-1, 2, -1]]) / 9
        F = convolve(self.mnt, grilleF, mode='constant', cval=0.0)
        fx = D
        fy = E
        fxx = 2 * A
        fyy = 2 * B
        fxy = C
        eps = 0.05
        p = (fx ** 2 + fy ** 2) * (fx ** 2 + fy ** 2)
        p[abs(p) < eps] = np.nan
        q = p + 1
        kv = -(fxx * fx ** 2 + 2 * fxy * fx * fy + fyy * fy ** 2) / (p * (q ** 3) ** (1 / 2))
        kh = -(fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) / (p * q ** (1 / 2))
        return kv, kh


    def courbures_pente_nulle(self):
        grilleA = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1]]) / (6 * self.pas ** 2)
        A = convolve(self.mnt, grilleA, mode='constant', cval=0.0)
        grilleB = np.array([[1, 1, 1], [-2, -2, -2], [1, 1, 1]]) / (6 * self.pas ** 2)
        B = convolve(self.mnt, grilleB, mode='constant', cval=0.0)
        grilleC = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / (4 * self.pas ** 2)
        C = convolve(self.mnt, grilleC, mode='constant', cval=0.0)
        kmin = -A - B - np.sqrt((A - B) ** 2 + C ** 2)
        kmax = -A - B + np.sqrt((A - B) ** 2 + C ** 2)
        return kmin, kmax