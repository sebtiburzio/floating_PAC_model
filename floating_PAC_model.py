#%%
#!/usr/bin/env python

import time
import sympy as sm
import dill

#%%
# Init

# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
num_masses = 2  # Number of masses to discretise along length (not including end mass)

# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
q = sm.Matrix([theta_0, theta_1, x, z, phi])
theta = sm.Matrix([theta_0, theta_1])
dtheta_0, dtheta_1, dx, dz, dphi = sm.symbols('dtheta_0 dtheta_1 dx dz dphi')
dq = sm.Matrix([dtheta_0, dtheta_1, dx, dz, dphi])

# Object coordinates in global frame (forward kinematics)
fk_x, fk_z = sm.symbols('fk_x fk_z')
fk = sm.Matrix([fk_x, fk_z])
alpha = sm.symbols('alpha') # tip orientation in object base frame

# Integration variables
s, v, d = sm.symbols('s v d')

#%% 
# Forward Kinematics
tic = time.perf_counter()

# Spine x,z in object base frame
alpha = theta_0*v + 0.5*theta_1*v**2
fk[0] = L*sm.integrate(sm.cos(alpha),(v, 0, s))
fk[1] = -L*sm.integrate(sm.sin(alpha),(v, 0, s)) # -ve to match rotation sense in robot frame

# FK of midpoint and endpoint in base frame (for curvature IK)
fk_mid_fixed = fk.subs(s, 0.5)
fk_end_fixed = fk.subs(s, 1)
J_mid_fixed = fk_mid_fixed.jacobian(sm.Matrix([theta_0, theta_1]))
J_end_fixed = fk_end_fixed.jacobian(sm.Matrix([theta_0, theta_1]))

# 3DOF floating base
rot_phi = sm.rot_axis3(phi)[:2,:2]                  # +ve rotations around robot Y-axis
rot_alpha = sm.rot_axis3(alpha.subs(v,s))[:2,:2]    #
fk = sm.Matrix([x, z]) + rot_phi@(fk + D*rot_alpha@sm.Matrix([0, d]))

toc = time.perf_counter()
print("FK gen time: " + str(toc-tic))
f_FK = sm.lambdify((q,p,s,d), fk, "mpmath")
f_FK_mf = sm.lambdify((theta,p), fk_mid_fixed, "mpmath")
f_FK_ef = sm.lambdify((theta,p), fk_end_fixed, "mpmath")
f_J_mf = sm.lambdify((theta,p), J_mid_fixed, "mpmath")
f_J_ef = sm.lambdify((theta,p), J_end_fixed, "mpmath")

dill.dump(f_FK, open("./generated_functions/f_FK", "wb"))
dill.dump(f_FK_mf, open("./generated_functions/f_FK_mf", "wb"))
dill.dump(f_FK_ef, open("./generated_functions/f_FK_ef", "wb"))
dill.dump(f_J_mf, open("./generated_functions/f_J_mf", "wb"))
dill.dump(f_J_ef, open("./generated_functions/f_J_ef", "wb"))

#%% 
# Potential (gravity) vector
tic = time.perf_counter()

# Energy
U = m_E*sm.integrate((fk[1].subs(s,1)),(d,-1/2,1/2))
for i in range(num_masses):
    U += (m_L/num_masses)*sm.integrate((fk[1].subs(s,i/num_masses)),(d,-1/2,1/2))

# Potential force
G = sm.Matrix([9.81*(U)]).jacobian(q)

toc = time.perf_counter()
print("G gen time: " + str(toc-tic))
f_G = sm.lambdify((q,p), G, "mpmath")
dill.dump(f_G, open("./generated_functions/f_G", "wb"))

#%% 
# Inertia matrix
tic = time.perf_counter()

J = (fk.subs(s, 1)).jacobian(q)
B = 0.5*m_E*sm.integrate(J.transpose()@J, (d, -1/2, 1/2))
for i in range(num_masses):
    J = (fk.subs(s, i/num_masses)).jacobian(q)
    B += 0.5*(m_L/num_masses)*sm.integrate(J.transpose()@J, (d, -1.2, 1/2))

toc = time.perf_counter()
print("B gen time: " + str(toc-tic))
f_B = sm.lambdify((q,p), B, "mpmath")
dill.dump(f_B, open("./generated_functions/f_B", "wb"))

#%% 
# Centrifugal/Coriolis matrix
# tic = time.perf_counter()

# C = sm.zeros(5,5)      
# for i in range(5):            
#     for j in range(5):    
#         for k in range(5):
#             Christoffel = 0.5*(sm.diff(B[i,j],q[k]) + sm.diff(B[i,k],q[j]) - sm.diff(B[j,k],q[i]))
#             C[i,j] = C[i,j] + Christoffel*dq[k]

# toc = time.perf_counter()
# print("C gen time: " + str(toc-tic))
# f_C = sm.lambdify((q,p,dq), C, "mpmath")
# dill.dump(f_C, open("f_C", "wb"))

#%%
# Test output

# q_vals = [1.0, 1.0, 0.0, 0.0, 0.0]
# dq_vals = [0.0, 0.0, 0.0, 0.0, 1.0]

# print(f_FK(q_vals,p_vals,0.5,0.0))
# print(f_G(q_vals,p_vals))
# print(f_B(q_vals,p_vals))
# print(f_C(q_vals,p_vals,dq_vals))