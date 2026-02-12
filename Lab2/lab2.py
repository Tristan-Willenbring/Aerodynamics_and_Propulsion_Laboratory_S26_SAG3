import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2)

# Constants
rho_water = 1.940 / (12**3) # slug in^-3
rho_air = 0.00237 # slug ft^-3
g = 32.2 * 12 # in s^-2
p_ref = 14.7 # psi

# Pull data from lab2_data.csv
data = np.genfromtxt('lab2_data.csv', delimiter=',', skip_header=1)
print(data)

# calculate the dH for each of the test sections
delta_H = np.zeros_like(data)
delta_H[:, 0:2] = data[:, 0:2]
delta_H[:, 7] = data[:, 7]
reference_data = data[0, 2:7]
for i in range(len(data)):
    delta_H[i, 2:7] = data[i, 2:7] - reference_data

print('delta_h:')
print(delta_H)

# evaluate the pressures for each dH
pressures = delta_H
pressures[:, 2:7] = p_ref - rho_water * g * pressures[:, 2:7] / 12

print(pressures)

# calculate the dynamic pressure and delta_P
# dynamic pressure
dynamic_pressure = pressures[:, 6] - pressures[:, 5]
print('dynamic_pressure:')
print(dynamic_pressure)

# delta_P
delta_P = pressures[:, 3] - pressures[:, 4]
print('delta_P:')
print(delta_P)

# calculate the best fit line
A = np.vstack([delta_P, np.ones_like(delta_P)]).T
m, c = np.linalg.lstsq(A, dynamic_pressure, rcond=None)[0]
print(m, c)

# calculate the velocities
velos = np.sqrt(144 * dynamic_pressure * 2 / rho_air)
fan_speed = pressures[:, 1]
A = np.vstack([fan_speed, np.ones_like(fan_speed)]).T
m1, c1 = np.linalg.lstsq(A, velos, rcond=None)[0]
print(m1, c1)
print(velos)

# plot the data
plt.figure(1)
plt.scatter(delta_P, dynamic_pressure, marker='o', color='b', facecolors='none', label='Calculated Values')
plt.plot(delta_P, m*delta_P + c, linestyle='--', color='r', linewidth=1, label='Best fit')
plt.legend()
plt.xlabel(r'$\Delta P = P_{A}-P_{E}$ (psi)')
plt.ylabel(r'$q = \frac{1}{2}\rho V^2$ (psi)')
plt.grid()
plt.title('Pitot Dynamic Pressure vs Tunnel Pressure Difference')

plt.figure(2)
plt.scatter(fan_speed, velos, marker='o', color='b', facecolors='none', label='Calculated Values')
plt.plot(fan_speed, m1*fan_speed + c1, linestyle='--', color='r', linewidth=1, label='Best fit')
plt.legend()
plt.xlabel(r'Fan Speed (Hz)')
plt.ylabel(r'V ($ft s^{-1}$)')
plt.grid()
plt.title('Tunnel Velocity vs Fan Speed')

plt.show()
