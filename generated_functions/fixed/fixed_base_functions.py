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
# Integration variables
s, d = sm.symbols('s d')

# Load serialised functions
f_FK = sm.lambdify((theta,p,s,d), pickle.load(open("./generated_functions/fixed/fk", "rb")), "mpmath")
def eval_fk(theta, p_vals, s, d): 
    return np.array(f_FK(theta, p_vals, s, d).apply(mp.re).tolist(), dtype=float)

f_FK_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_mid_fixed", "rb")), "mpmath")
def eval_midpt(theta, p_vals): 
    return np.array(f_FK_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_FK_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_end_fixed", "rb")), "mpmath")
def eval_endpt(theta, p_vals): 
    return np.array(f_FK_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_J_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_mid_fixed", "rb")), "mpmath")
def eval_J_midpt(theta, p_vals): 
    return np.array(f_J_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)

f_J_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/J_end_fixed", "rb")), "mpmath")
def eval_J_endpt(theta, p_vals): 
    return np.array(f_J_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)