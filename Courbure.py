import numpy as np

class Courbure():
    def __init__(self, mnt, pas, name):
        self.mnt = mnt
        self.pas = pas
        self.name = name

    def courbures_TPP(self):
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, 1:-1]) / self.pas
        fy = (self.mnt[:-2, 1:-1] - self.mnt[1:-1, 1:-1]) / self.pas
        z1 = self.mnt[:-2, :-2]
        z2 = self.mnt[:-2, 1:-1]
        z3 = self.mnt[:-2, 2:]
        z4 = self.mnt[1:-1, :-2]
        z5 = self.mnt[1:-1, 1:-1]
        z6 = self.mnt[1:-1, 2:]
        z7 = self.mnt[2:, :-2]
        z8 = self.mnt[2:, 1:-1]
        z9 = self.mnt[2:, 2:]
        if self.pente(fx, fy) == 0:
            kv = np.nan
            kh = np.nan
        else:
            fxx = (z6 - 2 * self.mnt[1:-1, 1:-1] + self.mnt[1:-1, :-2]) / self.pas ** 2
            fyy = (self.mnt[2:, 1:-1] - 2 * self.mnt[1:-1, 1:-1] + self.mnt[:-2, 1:-1]) / self.pas ** 2
            fxy = (self.mnt[:-1, 1:] - self.mnt[:-1, :-1] - self.mnt[1:, 1:] + self.mnt[1:, :-1]) / (self.pas ** 2)
            p = fx ** 2 + fy ** 2
            q = p + 1
            kv = -(fxx * fx ** 2 + 2 * fxy * fx * fy + fyy * fy ** 2) / (p * (q ** 3) ** (1 / 2))
            kh = -(fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) / (p * q ** (1 / 2))

        return kv, kh


    def courbures_FCN(self):
        fx = (self.mnt[1:-1, 2:] - self.mnt[1:-1, :-2]) / (2 * self.pas)
        fy = (self.mnt[:-2, 1:-1] - self.mnt[2:, 1:-1]) / (2 * self.pas)
        z1 = self.mnt[:-2, :-2]
        z2 = self.mnt[:-2, 1:-1]
        z3 = self.mnt[:-2, 2:]
        z4 = self.mnt[1:-1, :-2]
        z5 = self.mnt[1:-1, 1:-1]
        z6 = self.mnt[1:-1, 2:]
        z7 = self.mnt[2:, :-2]
        z8 = self.mnt[2:, 1:-1]
        z9 = self.mnt[2:, 2:]
        delta_x = delta_y = self.pas
        if self.pente(fx, fy) == 0:
            kv = np.nan
            kh = np.nan
        else:
            fxx = (z6 - 2 * z5 + z4) / (delta_x ** 2)
            fyy = (z2 - 2 * z5 + z8) / (delta_y ** 2)
            fxy = (z1 - z3 - z7 + z9) / (4 * delta_x * delta_y)
            p = fx ** 2 + fy ** 2
            q = p + 1
            kv = -(fxx * fx ** 2 + 2 * fxy * fx * fy + fyy * fy ** 2) / (p * (q ** 3) ** (1 / 2))
            kh = -(fxx * fy ** 2 - 2 * fxy * fx * fy + fyy * fx ** 2) / (p * q ** (1 / 2))
        return kv, kh


    def courbures_approxi(self):
        z1 = self.mnt[:-2, :-2]
        z2 = self.mnt[:-2, 1:-1]
        z3 = self.mnt[:-2, 2:]
        z4 = self.mnt[1:-1, :-2]
        z5 = self.mnt[1:-1, 1:-1]
        z6 = self.mnt[1:-1, 2:]
        z7 = self.mnt[2:, :-2]
        z8 = self.mnt[2:, 1:-1]
        z9 = self.mnt[2:, 2:]

        A = (z1 + z3 + z4 + z6 + z7 + z9) / (6 * self.pas ** 2) - (z2 + z5 + z8) / (3 * self.pas ** 2)
        B = (z1 + z2 + z3 + z7 + z8 + z9) / (6 * self.pas ** 2) - (z4 + z5 + z6) / (3 * self.pas ** 2)
        C = (z3 + z7 - z1 - z9) / (4 * self.pas ** 2)
        kmin = -A - B - np.sqrt((A - B) ** 2 + C ** 2)
        kmax = -A - B + np.sqrt((A - B) ** 2 + C ** 2)
        kmean = (kmin + kmax) / 2

        return kmean