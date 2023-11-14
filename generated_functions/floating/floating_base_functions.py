import pickle
import numpy as np
import mpmath as mp
import sympy as sm

# Import forward kinematics
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
q = sm.Matrix([theta_0, theta_1, x, z, phi])
dtheta_0, dtheta_1, dx, dz, dphi = sm.symbols('dtheta_0 dtheta_1 dx dz dphi')
dq = sm.Matrix([dtheta_0, dtheta_1, dx, dz, dphi])
# Integration variables
s, d = sm.symbols('s d')
# Gravity direction
gamma = sm.symbols('gamma')  

# Load serialised functions # TODO (maybe) - swap order of theta a p arguments to match matlab code style
f_FK = sm.lambdify((q,p,s,d), pickle.load(open("./generated_functions/floating/fk", "rb")), "mpmath")
def eval_fk(q, p_vals, s, d): 
    return np.array(f_FK(q, p_vals, s, d).apply(mp.re).tolist(), dtype=float)

f_FKA = sm.lambdify((q,p,s,d), pickle.load(open("./generated_functions/floating/fka", "rb")), "mpmath")
def eval_fka(q, p_vals, s, d): 
    return np.array(f_FKA(q, p_vals, s, d).apply(mp.re).tolist(), dtype=float)

f_J_end_wrt_base = sm.lambdify((x,z,phi,p), pickle.load(open("./generated_functions/floating/J_end_wrt_base", "rb")), "mpmath")
def eval_J_end_wrt_base(x, z, phi, p_vals):
    return np.array(f_J_end_wrt_base(x, z, phi, p_vals).apply(mp.re).tolist(), dtype=float)

# Loading the dynamic functions takes a long time # TODO - fix, probably in Fresnel computation

# f_G = sm.lambdify((q,p), pickle.load(open("./generated_functions/floating/G", "rb")), "mpmath")
# def eval_G(q, p_vals): 
#     return np.array(f_G(q, p_vals).apply(mp.re).tolist(), dtype=float)

# f_Gv = sm.lambdify((q,gamma,p), pickle.load(open("./generated_functions/floating/Gv", "rb")), "mpmath")
# def eval_Gv(q, gamma, p_vals): 
#     return np.array(f_Gv(q, gamma, p_vals).apply(mp.re).tolist(), dtype=float)

# f_B = sm.lambdify((q,p), pickle.load(open("./generated_functions/floating/B", "rb")), "mpmath")
# def eval_B(q, p_vals): 
#     return np.array(f_B(q, p_vals).apply(mp.re).tolist(), dtype=float)

# f_C = sm.lambdify((q,dq,p), pickle.load(open("./generated_functions/floating/C", "rb")), "mpmath")
# def eval_C(q, dq, p_vals): 
#     return np.array(f_C(q, dq, p_vals).apply(mp.re).tolist(), dtype=float)
