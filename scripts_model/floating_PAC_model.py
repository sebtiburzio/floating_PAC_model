#!/usr/bin/env python
#%%
import time
import sympy as sm
import pickle

#%%
# Init

# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
num_masses = 6  # Number of masses to discretise along length (not including end mass)
gamma = sm.symbols('gamma')  # Gravity direction

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

# Spine x,z in object base frame, defined as if it was reflected in the robot XY plane
alpha = theta_0*v + 0.5*theta_1*v**2 # negative curvature so sense matches robot frame Y axis rotation
fk[0] = -L*sm.integrate(sm.sin(alpha),(v, 0, s)) # x. when theta=0, x=0.
fk[1] = -L*sm.integrate(sm.cos(alpha),(v, 0, s)) # z. when theta=0, z=-L. 
# A manual subsitution is needed here to get around a SymPy bug: https://github.com/sympy/sympy/issues/25093
# TODO - remove when fix included in SymPy release
fk = fk.subs(1/sm.sqrt(theta_1), sm.sqrt(1/theta_1))

# FK of midpoint and endpoint in base frame (for curvature IK)
# TODO these should probably be in fixed_PAC_model.py but first try moving them there had some errors so leaving for now
fk_mid_fixed = fk.subs(s, 0.5)
fk_end_fixed = fk.subs(s, 1)
J_mid_fixed = fk_mid_fixed.jacobian(sm.Matrix([theta_0, theta_1]))
J_end_fixed = fk_end_fixed.jacobian(sm.Matrix([theta_0, theta_1]))

# 3DOF floating base
rot_phi = sm.rot_axis3(phi)[:2,:2] # +ve rotations around robot base Y axis (CW in XZ plane)
rot_alpha = sm.rot_axis3(alpha.subs(v,s))[:2,:2]
fk = sm.Matrix([x, z]) + rot_phi@(fk + D*rot_alpha@sm.Matrix([d, 0])) # Position
fka = sm.Matrix([fk, phi + alpha.subs(v,s)]) # Position and orientation

# Jacobian of end pose wrt floating base configuration
J_end_wrt_base = fka.subs([(s, 1),(d,0)]).jacobian(sm.Matrix([x, z, phi]))

toc = time.perf_counter()
print("FK gen time: " + str(toc-tic))

pickle.dump(fk, open("../generated_functions/floating/fk", "wb"))
pickle.dump(fka, open("../generated_functions/floating/fka", "wb"))
pickle.dump(fk_mid_fixed, open("../generated_functions/fixed/fk_mid_fixed", "wb"))
pickle.dump(fk_end_fixed, open("../generated_functions/fixed/fk_end_fixed", "wb"))
pickle.dump(J_mid_fixed, open("../generated_functions/fixed/J_mid_fixed", "wb"))
pickle.dump(J_end_fixed, open("../generated_functions/fixed/J_end_fixed", "wb"))
pickle.dump(J_end_wrt_base, open("../generated_functions/floating/J_end_wrt_base", "wb"))
f_FK = sm.lambdify((q,p,s,d), fk, "mpmath")
f_FKA = sm.lambdify((q,p,s,d), fka, "mpmath")
f_FK_mf = sm.lambdify((theta,p), fk_mid_fixed, "mpmath")
f_FK_ef = sm.lambdify((theta,p), fk_end_fixed, "mpmath")
f_J_mf = sm.lambdify((theta,p), J_mid_fixed, "mpmath")
f_J_ef = sm.lambdify((theta,p), J_end_fixed, "mpmath")
f_J_eb = sm.lambdify((x,z,phi,p), J_end_wrt_base, "mpmath")

#%% 
# Potential (gravity) vector
tic = time.perf_counter()

# Energy
U = 0.08*sm.integrate(((sm.sin(gamma)*fk[0] + sm.cos(gamma)*fk[1]).subs(s,0)),(d,-1/2,1/2)) # % Base mass currently just FT sensor flange mass. Should be combined with cable clamp and adapter (these are currently included in cable weight)
U += m_E*sm.integrate(((sm.sin(gamma)*fk[0] + sm.cos(gamma)*fk[1]).subs(s,1)),(d,-1/2,1/2))
for i in range(num_masses):
    U += (m_L/num_masses)*sm.integrate(((sm.sin(gamma)*fk[0] + sm.cos(gamma)*fk[1]).subs(s,i/num_masses + 1/(num_masses*2))),(d,-1/2,1/2))

# Potential force
G = sm.Matrix([9.81*(U.subs(gamma,0))]).jacobian(q).T
Gv = sm.Matrix([9.81*(U)]).jacobian(q).T

toc = time.perf_counter()
print("G gen time: " + str(toc-tic))

pickle.dump(G, open("../generated_functions/floating/G", "wb"))
pickle.dump(Gv, open("../generated_functions/floating/Gv", "wb"))

#%% 
# Inertia matrix
tic = time.perf_counter()

J = (fk.subs(s, 0)).jacobian(q)
B = 0.08*sm.integrate(J.T@J, (d, -1/2, 1/2)) #Base mass currently just FT sensor flange mass. Should be combined with cable clamp and adapter (these are currently included in cable weight)
J = (fk.subs(s, 1)).jacobian(q)
B += m_E*sm.integrate(J.T@J, (d, -1/2, 1/2))
for i in range(num_masses):
    J = (fk.subs(s, i/num_masses + 1/(num_masses*2))).jacobian(q)
    B += (m_L/num_masses)*sm.integrate(J.T@J, (d, -1/2, 1/2))

toc = time.perf_counter()
print("B gen time: " + str(toc-tic))

pickle.dump(B, open("../generated_functions/floating/B", "wb"))

#%% 
# Centrifugal/Coriolis matrix
tic = time.perf_counter()

C = sm.zeros(5,5)      
for i in range(5):            
    for j in range(5):    
        for k in range(5):
            Christoffel = 0.5*(sm.diff(B[i,j],q[k]) + sm.diff(B[i,k],q[j]) - sm.diff(B[j,k],q[i]))
            C[i,j] = C[i,j] + Christoffel*dq[k]

toc = time.perf_counter()
print("C gen time: " + str(toc-tic))

pickle.dump(C, open("../generated_functions/floating/C", "wb"))

# %%
