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
num_masses = 4  # Number of masses to discretise along length (not including end mass)

# Configuration variables
theta_0, theta_1 = sm.symbols('theta_0 theta_1')
theta = sm.Matrix([theta_0, theta_1])
dtheta_0, dtheta_1 = sm.symbols('dtheta_0 dtheta_1')
dtheta = sm.Matrix([dtheta_0, dtheta_1])
ddtheta_0, ddtheta_1 = sm.symbols('ddtheta_0 ddtheta_1')
ddtheta = sm.Matrix([ddtheta_0, ddtheta_1])

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
alpha = -(theta_0*v + 0.5*theta_1*v**2) # negative curvature so sense matches robot frame Y axis rotation
fk[0] = L*sm.integrate(sm.sin(alpha),(v, 0, s)) # x. when theta=0, x=0.
fk[1] = -L*sm.integrate(sm.cos(alpha),(v, 0, s)) # z. when theta=0, z=-L. 
# A manual subsitution is needed here to get around a SymPy bug: https://github.com/sympy/sympy/issues/25093
# TODO - remove when fix included in SymPy release
fk = fk.subs(1/sm.sqrt(theta_1), sm.sqrt(1/theta_1))

# FK position at d in cross section
rot_alpha = sm.rot_axis3(alpha.subs(v,s))[:2,:2]
fk = fk + D*rot_alpha@sm.Matrix([0, d])

toc = time.perf_counter()
print("FK gen time: " + str(toc-tic))

pickle.dump(fk, open("./generated_functions/fixed/fk", "wb"))
f_FK = sm.lambdify((theta,p,s,d), fk, "mpmath")

#%% 
# Potential (gravity) vector
tic = time.perf_counter()

# Energy
U = m_E*sm.integrate((fk[1].subs(s,1)),(d,-1/2,1/2))
for i in range(num_masses):
    U += (m_L/num_masses)*sm.integrate((fk[1].subs(s,i/num_masses)),(d,-1/2,1/2))

# Potential force
G = sm.Matrix([9.81*(U)]).jacobian(theta).T

toc = time.perf_counter()
print("G gen time: " + str(toc-tic))

pickle.dump(G, open("./generated_functions/fixed/G", "wb"))

#%% 
# Inertia matrix
tic = time.perf_counter()

J = (fk.subs(s, 1)).jacobian(theta)
B = 0.5*m_E*sm.integrate(J.transpose()@J, (d, -1/2, 1/2))
for i in range(num_masses):
    J = (fk.subs(s, i/num_masses)).jacobian(theta)
    B += 0.5*(m_L/num_masses)*sm.integrate(J.transpose()@J, (d, -1/2, 1/2))

toc = time.perf_counter()
print("B gen time: " + str(toc-tic))

pickle.dump(B, open("./generated_functions/fixed/B", "wb"))

#%%
# Centrifugal/Coriolis matrix
tic = time.perf_counter()

C = sm.zeros(2,2)      
for i in range(2):            
    for j in range(2):    
        for k in range(2):
            Christoffel = 0.5*(sm.diff(B[i,j],theta[k]) + sm.diff(B[i,k],theta[j]) - sm.diff(B[j,k],theta[i]))
            C[i,j] = C[i,j] + Christoffel*dtheta[k]

toc = time.perf_counter()
print("C gen time: " + str(toc-tic))

pickle.dump(C, open("./generated_functions/fixed/C", "wb"))

# %%
# Linear factorisation by masses
E = B*ddtheta + C*dtheta + G
Y = E.jacobian(sm.Matrix([m_L,m_E]))

pickle.dump(Y, open("./generated_functions/fixed/Y", "wb"))

# %%
# Factor out m_L for identification
dE_dmL = E.diff(m_L)
E_mL_0 = E.subs(m_L,0)

pickle.dump(dE_dmL, open("./generated_functions/fixed/dE_dmL", "wb"))
pickle.dump(E_mL_0, open("./generated_functions/fixed/E_mL_0", "wb"))

# %%
# Factor out m_E for identification
dE_dmE = E.diff(m_E)
E_mE_0 = E.subs(m_E,0)

pickle.dump(dE_dmE, open("./generated_functions/fixed/dE_dmE", "wb"))
pickle.dump(E_mE_0, open("./generated_functions/fixed/E_mE_0", "wb"))

# %%
