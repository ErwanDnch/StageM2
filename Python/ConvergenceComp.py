import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

dir = "/home/erwan/Master/Stage/Python/ConvergenceFiles/"
files = []

header = ["Fp.x", "Fp.y", "Fmu.x", "Fmu.y", "t"]
for file in os.listdir(dir):
    files.append((pd.read_csv(dir+file, sep=' ', header = None, names = header), file))

print(files[0])

for df, legend in reversed(files):
    plt.plot(df["t"],df["Fp.x"], label=legend)

plt.legend()
plt.show()
