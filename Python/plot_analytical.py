import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import pandas as pd

# Constants (set realistic example values)
U = 0.2       # translational velocity (m/s)

mu = 0.3      # friction coefficient
L = 0.6       # length of the rod (m)
R = 0.05      # radius (m)

# Range of o values
o_values = np.linspace(-0.5, 0.5, 20)
phi_solutions = []

df = pd.read_csv("variables.csv")
phi = df["phi"]
omega = df["omega"]

def eq_phi_dyn(phi, L, o, mu, U, omega, CD_phi, CD_R):
    return U**2 * np.sin(phi)**2 * np.cos(phi) - mu * (U**2 * np.cos(phi)**2 - 2*omega*U*(o-R*phi) + omega**2 * (L**2/12 + o**2 + (R*phi)**2 - 2*o*R*phi))

def phi_slide_dyn(L, o, mu, U, omega):
    phi_solution = fsolve(eq_phi_dyn, 1, args=(L, o, mu, U, omega))

for o in o_values:
    F_phi = U**2 * np.sin(phi)**2 * np.cos(phi)
    F_N = (U**2 * np.cos(phi)**2 - 2*omega*U*(o-R*phi) + omega**2 * (L**2/12 + o**2 + (R*phi)**2 - 2*o*R*phi))
    phi_solutions.append(np.argmax(F_phi > mu * F_N))

plt.plot(o_values, phi_solutions, label=r'$\phi$ solution vs $o$')
plt.xlabel('Offset $o$ (m)')
plt.ylabel(r'$\phi$ (rad)')
plt.title(r'$\frac{F_\phi}{\mu F_R} = 1$ solution')
plt.grid(True)
plt.legend()
plt.show()
