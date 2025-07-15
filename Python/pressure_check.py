import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame()

df1 = pd.read_csv("data_amont.csv")
df2 = pd.read_csv("data_amont.csv")

#plt.plot(df1["max(p)"])
plt.plot(df1["max(p)"]-df2["max(p)"])
plt.show()
