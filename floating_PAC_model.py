#%%
#!/usr/bin/env python

import time
import sympy as sm
import matplotlib.pyplot as plt
import numpy as np

#%% 
# Funcs

def plot_FK(f_FK, q_repl, p_repl):
    s_evals = np.linspace(0,1,11)
    FK_evals = np.empty((s_evals.size,2,))
    FK_evals[:] = np.nan
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(f_FK.evalf(subs=q_repl|p_repl|{d:0}|{s:s_evals[i_s]})).squeeze() # very slow TODO - different eval method
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1])
    plt.xlim(FK_evals[0,0]-1.2,FK_evals[0,0]+1.2)
    plt.ylim(FK_evals[0,1]-1.2,FK_evals[0,1]+1.2)
    ax.set_aspect('equal','box')
    plt.show()

#%%
# Init

# parameters
m, g, L, D = sm.symbols('m g L D')
p = sm.Matrix([m, g, L, D])
p_vals = [1.0, 9.81, 1.0, 0.1]

# configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
q = sm.Matrix([theta_0, theta_1, x, z, phi])
q_vals = [1.0, 1.0, 0.0, 0.0, 0.0]
dtheta_0, dtheta_1, dx, dz, dphi = sm.symbols('dtheta_0 dtheta_1 dx dz dphi')
dq = sm.Matrix([dtheta_0, dtheta_1, dx, dz, dphi])
dq_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
alpha = sm.symbols('alpha') # tip orientation in object base frame

# object coordinates in global frame (forward kinematics)
fk_x, fk_z = sm.symbols('fk_x fk_z')
fk = sm.Matrix([fk_x, fk_z])

# integration variables
s, v, d = sm.symbols('s v d')

#%% 
# Forward Kinematics
tic = time.perf_counter()

# spine x,z in object base frame
alpha = theta_0*v + 0.5*theta_1*v**2
fk[0] = L*sm.integrate(sm.cos(alpha),(v, 0, s))
fk[1] = -L*sm.integrate(sm.sin(alpha),(v, 0, s)) # -ve to match rotation sense in robot frame

# 3DOF floating base
rot_phi = sm.rot_axis3(phi)[:2,:2]                  # +ve rotations around robot Y-axis
rot_alpha = sm.rot_axis3(alpha.subs(v,s))[:2,:2]    #
fk = sm.Matrix([x, z]) + rot_phi@(fk + D*rot_alpha@sm.Matrix([0, d]))

# plot_FK(fk, dict(zip(q,q_vals)), dict(zip(p,p_vals))) # Test FK

toc = time.perf_counter()
print("FK gen time: " + str(toc-tic))

#%% 
# Potential (gravity) vector
tic = time.perf_counter()

# Energy
U_tip = 0.5*sm.integrate((fk[1].subs(s,1)),(d,-1/2,1/2)) # tip mass
U_base = 0.5*sm.integrate((fk[1].subs(s,0)),(d,-1/2,1/2)) # base mass

# Potential force
G = sm.Matrix([m*g*(U_base + U_tip)]).jacobian(q)

toc = time.perf_counter()
print("G gen time: " + str(toc-tic))

#%% 
# Inertia matrix
tic = time.perf_counter()

J_tip = (fk.subs(s, 1)).jacobian(q) # mass at tip
J_base = (fk.subs(s, 0)).jacobian(q) # mass at base

B = 0.5*m*sm.integrate(J_tip.transpose()@J_tip, (d, -1/2, 1/2)) + \
    0.5*m*sm.integrate(J_base.transpose()@J_base, (d, -1/2, 1/2))

toc = time.perf_counter()
print("B gen time: " + str(toc-tic))

#%% 
# Centrifugal/Coriolis matrix
tic = time.perf_counter()

C = sm.zeros(5,5)      
for i in range(5):            
    for j in range(5):    
        for k in range(5):
            Christoffel = 0.5*(sm.diff(B[i,j],q[k]) + sm.diff(B[i,k],q[j]) - sm.diff(B[j,k],q[i]))
            C[i,j] = C[i,j] + Christoffel*dq[k]
# # TODO - check if this is correct

toc = time.perf_counter()
print("C gen time: " + str(toc-tic))

#%%
# Functions for numerical evaluation

f_FK = sm.lambdify((q,p,s,d), fk)
f_G = sm.lambdify((q,p), G)
f_B = sm.lambdify((q,p), B)
f_C = sm.lambdify((q,p,dq), C)

#%%
# Test output

# print(f_FK(q_vals,p_vals,0.5,0.0))
# print(f_G(q_vals,p_vals))
# print(f_B(q_vals,p_vals))
# print(f_C(q_vals,p_vals,dq_vals))
