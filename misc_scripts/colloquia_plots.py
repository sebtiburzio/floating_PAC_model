#%%

import os
import pickle
import numpy as np
import mpmath as mp
import sympy as sm
import matplotlib.pyplot as plt

#%%
# Import forward kinematics
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
p_vals = [1.0, 1.0, 1.0, 0.01]
# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
theta = sm.Matrix([theta_0, theta_1])
q = sm.Matrix([theta_0, theta_1, x, z, phi])
# Integration variables
s, d = sm.symbols('s d')
# Load functions
f_FK_mid = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_mid_fixed", "rb")), "mpmath")
f_FK_end = sm.lambdify((theta,p), pickle.load(open("./generated_functions/fk_end_fixed", "rb")), "mpmath")
f_FK = sm.lambdify((q,p,s,d), pickle.load(open("./generated_functions/fk", "rb")), "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_fk(q, p_vals, s, d): return np.array(f_FK(q, p_vals, s, d).apply(mp.re).tolist(), dtype=float)
def eval_midpt(theta, p_vals): return np.array(f_FK_mid(theta, p_vals).apply(mp.re).tolist(), dtype=float)
def eval_endpt(theta, p_vals): return np.array(f_FK_end(theta, p_vals).apply(mp.re).tolist(), dtype=float)

#%%
def plot(q_repl, dir=None, n=0):
    s_evals = np.linspace(0,1,31)
    FK_evals = np.empty((s_evals.size,2,1))
    FK_evals[:] = np.nan
    for i_s in range(s_evals.size):
       FK_evals[i_s] = eval_fk(q_repl,p_vals,s_evals[i_s],0.0)

    fig, (ax_fk, ax_curv)  = plt.subplots(1,2)
    ax_fk.plot(FK_evals[:,0],FK_evals[:,1],'k')
    ax_fk.plot([-0.05,0.05],[0,0],'grey')
    ax_fk.set_xlim(-0.8,0.8)
    ax_fk.set_ylim(-1.1,0.2)
    ax_fk.set_xlabel('x')
    ax_fk.set_ylabel('y')
    
    ax_curv.plot([0,1],[q_repl[0],q_repl[0]])
    ax_curv.plot([0,1],[0,q_repl[1]])
    ax_curv.plot([0,1],[q_repl[0],q_repl[0]+q_repl[1]])
    ax_curv.set_xlim(-0.1,1.1)
    ax_curv.set_ylim(-10,10)
    ax_curv.legend(['$q_0=$'+str(round(q_repl[0],2)),'$q_1$='+str(round(q_repl[1],2)), '$c$'], loc='upper left')
    ax_curv.set_xlabel('s')

    ax_fk.set_aspect('equal')

    fig.set_figwidth(12)

    if(dir is None):
        plt.show()
    else:
        plt.savefig(dir + str(n) +'.png', bbox_inches='tight', pad_inches=0.1)
        plt.clf()
        plt.close(fig)

# %%
# Generate images
def gen_imgs(q00, q01, q10, q11):
    frames = 30
    dir = './colloquia_anims/' + str(q00) + '_' + str(q01) + '_to_' + str(q10) + '_' + str(q11) + '/'
    if not os.path.exists(dir):
            os.makedirs(dir)
    for n in range(frames):
        q_repl = [1e-3 + q00 + n*(q10-q00)/frames, 1e-3 + q01 + n*(q11-q01)/frames, 0, 0, 0]
        plot(q_repl, dir, n)

# %%
