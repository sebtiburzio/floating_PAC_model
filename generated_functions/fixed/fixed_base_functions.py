import pickle
import numpy as np
import mpmath as mp
import sympy as sm

# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1 = sm.symbols('theta_0 theta_1')
theta = sm.Matrix([theta_0, theta_1])
dtheta_0, dtheta_1 = sm.symbols('dtheta_0 dtheta_1')
dtheta = sm.Matrix([dtheta_0, dtheta_1])
# Integration variables
s, d = sm.symbols('s d')
# Gravity direction
gamma = sm.symbols('gamma')  

# Load serialised functions # TODO (maybe) - swap order of theta a p arguments to match matlab code style
f_FK = sm.lambdify((theta,p,s,d), pickle.load(open("./generated_functions/fixed/fk", "rb")), "mpmath")
def eval_fk(theta, p_vals, s, d): 
    return np.array(f_FK(theta, p_vals, s, d).apply(mp.re).tolist(), dtype=float)

f_FK_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/fk_mid_fixed", "rb")), "mpmath")
def eval_midpt(theta, p_vals): 
    return np.array(f_FK_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_FK_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/fk_end_fixed", "rb")), "mpmath")
def eval_endpt(theta, p_vals): 
    return np.array(f_FK_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_J_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/J_mid_fixed", "rb")), "mpmath")
def eval_J_midpt(theta, p_vals): 
    return np.array(f_J_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_J_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/J_end_fixed", "rb")), "mpmath")
def eval_J_endpt(theta, p_vals): 
    return np.array(f_J_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

# Loading the dynamic functions takes a long time # TODO - fix, probably in Fresnel computation

# f_G = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/G", "rb")), "mpmath")
# def eval_G(theta, p_vals): 
#     return np.array(f_G(theta, p_vals).apply(mp.re).tolist(), dtype=float)

# f_Gv = sm.lambdify((theta,gamma,p), pickle.load(open("./generated_functions/fixed/Gv", "rb")), "mpmath")
# def eval_Gv(theta, gamma, p_vals): 
#     return np.array(f_Gv(theta, gamma, p_vals).apply(mp.re).tolist(), dtype=float)

# f_B = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fixed/B", "rb")), "mpmath")
# def eval_B(theta, p_vals): 
#     return np.array(f_B(theta, p_vals).apply(mp.re).tolist(), dtype=float)

# f_C = sm.lambdify((theta,dtheta,p), pickle.load(open("./generated_functions/fixed/C", "rb")), "mpmath")
# def eval_C(theta, dtheta, p_vals): 
#     return np.array(f_C(theta, dtheta, p_vals).apply(mp.re).tolist(), dtype=float)

