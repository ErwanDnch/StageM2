import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("/home/erwan/Master/Stage/SlipFromSim/csv_data/U0.2_D0.1_LEVEL11/data_o0.0_phi0_om0.csv")

plt.plot(df["t"], df["T_l"], label='LEFT')
plt.plot(df["t"], df["min_T_r"], label='RIGHT')
plt.legend()
plt.show()
