import numpy as np
import os
import matplotlib.pyplot as plt

RHO = 1.194
D = 0.7 * 0.0254
mu = 1.81e-5

def obtain_pressures(file):
    data = np.genfromtxt(file, delimiter=',')
    to_remove = np.append(np.append(np.array([0, 1]), np.arange(18, 36, 1)), np.arange(40, 68, 1))
    data = np.delete(data, to_remove, 1)
    pressures = []
    std_errs = []
    for port in data.T:
        pressures.append(np.mean(port))
        std_errs.append(np.std(port) / np.sqrt(len(port)))
    pressures = np.array(pressures)
    std_errs = np.array(std_errs)
    return pressures, std_errs

Cds = []
Res = []
for file in os.listdir('./pressure_data'):
    pressures, err = obtain_pressures('./pressure_data/' + file)
    if int(file.strip("Hz.csv")) < 20:
        pressures[3] = (pressures[2] + pressures[4]) / 2
    P_A = pressures[18]
    P_E = pressures[19]
    angles = 20 * np.arange(0, 18, 1)

    C_p = np.empty_like(pressures[:18])
    for i, pressure in enumerate(pressures[:18]):
        C_p[i] = (pressure - P_E) / (1.1 * (P_A - P_E))
    C_d = np.sum((C_p * np.cos(np.deg2rad(angles)))) * (-np.pi / 18)
    Cds.append(C_d)

    q = 1.1 * (P_A - P_E)
    V = np.sqrt(2 * q / RHO)
    Re = RHO * V * D / mu
    Res.append(Re)

    print(f"Data for {file.strip(".csv")}:")
    print(f"C_d: {C_d:.4f}")
    print(f"Re: {Re:.2e}\n")

    plt.figure()
    plt.plot(angles, C_p)
    plt.title(f"Pressure Coefficient Distribution at {file.strip(".csv")}")
    plt.xlabel("Angle [degrees]")
    plt.ylabel(r"$C_p$")
    plt.grid()

plt.figure()
plt.scatter(Res, Cds)
plt.title(r"$C_D$ vs Reynolds Number")
plt.xlabel("Reynolds Number")
plt.ylabel(r"$C_D$")
plt.grid()
plt.show()