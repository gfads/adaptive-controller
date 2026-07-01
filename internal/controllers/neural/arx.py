import numpy as np

data = np.loadtxt("plant_io.csv", delimiter=",", skiprows=1)
u = data[:,0]
y = data[:,1]

na, nb, nk = 2, 2, 1  # ARX orders
N = len(y)
phi = []
for t in range(max(na, nb+nk-1), N):
    y_terms = [-y[t-i-1] for i in range(na)]
    u_terms = [u[t-j-nk] for j in range(nb)]
    phi.append(y_terms + u_terms)
Phi = np.array(phi)
Y = y[max(na, nb+nk-1):]

theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
a = theta[:na]
b = theta[na:]

print("ARX model:")
print(f"y_t = {a} * past y + {b} * past u")
