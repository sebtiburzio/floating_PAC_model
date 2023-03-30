#%%
#!/usr/bin/env python

import time
import sympy as sm
import dill

#%%
# Init

# constant parameters
m, g, L, D = sm.symbols('m g L D')
p = sm.Matrix([m, g, L, D])

# configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
q = sm.Matrix([theta_0, theta_1, x, z, phi])
dtheta_0, dtheta_1, dx, dz, dphi = sm.symbols('dtheta_0 dtheta_1 dx dz dphi')
dq = sm.Matrix([dtheta_0, dtheta_1, dx, dz, dphi])

# object coordinates in global frame (forward kinematics)
fk_x, fk_z = sm.symbols('fk_x fk_z')
fk = sm.Matrix([fk_x, fk_z])
alpha = sm.symbols('alpha') # tip orientation in object base frame

# integration variables
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
f_FK = sm.lambdify((q,p,s,d), fk)
f_FK_mf = sm.lambdify((theta_0,theta_1,p), fk_mid_fixed)
f_FK_ef = sm.lambdify((theta_0,theta_1,p), fk_end_fixed)
f_J_mf = sm.lambdify((theta_0,theta_1,p), J_mid_fixed)
f_J_ef = sm.lambdify((theta_0,theta_1,p), J_end_fixed)
dill.dump(f_FK, open("./generated_functions/f_FK", "wb"))
dill.dump(f_FK_mf, open("./generated_functions/f_FK_mf", "wb"))
dill.dump(f_FK_ef, open("./generated_functions/f_FK_ef", "wb"))
dill.dump(f_J_mf, open("./generated_functions/f_J_mf", "wb"))
dill.dump(f_J_ef, open("./generated_functions/f_J_ef", "wb"))

#%% 
# Potential (gravity) vector
tic = time.perf_counter()

# Energy
U_tip = 0.5*sm.integrate((fk[1].subs(s,1)),(d,-1/2,1/2)) # tip mass TODO - update for multiple masses on length and endpoint mass
U_base = 0.5*sm.integrate((fk[1].subs(s,0)),(d,-1/2,1/2)) # base mass

# Potential force
G = sm.Matrix([m*g*(U_base + U_tip)]).jacobian(q)

toc = time.perf_counter()
print("G gen time: " + str(toc-tic))
f_G = sm.lambdify((q,p), G)
dill.dump(f_G, open("./generated_functions/f_G", "wb"))

#%% 
# Inertia matrix
tic = time.perf_counter()

J_tip = (fk.subs(s, 1)).jacobian(q) # mass at tip
J_base = (fk.subs(s, 0)).jacobian(q) # mass at base

B = 0.5*m*sm.integrate(J_tip.transpose()@J_tip, (d, -1/2, 1/2)) + \
    0.5*m*sm.integrate(J_base.transpose()@J_base, (d, -1/2, 1/2))

toc = time.perf_counter()
print("B gen time: " + str(toc-tic))
f_B = sm.lambdify((q,p), B)
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
# f_C = sm.lambdify((q,p,dq), C)
# dill.dump(f_C, open("f_C", "wb"))

#%%
# Test output

# q_vals = [1.0, 1.0, 0.0, 0.0, 0.0]
# dq_vals = [0.0, 0.0, 0.0, 0.0, 1.0]

# print(f_FK(q_vals,p_vals,0.5,0.0))
# print(f_G(q_vals,p_vals))
# print(f_B(q_vals,p_vals))
# print(f_C(q_vals,p_vals,dq_vals))