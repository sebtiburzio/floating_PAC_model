#!/usr/bin/env python
#%%
import pickle
import numpy as np
import mpmath as mp
import sympy as sm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

#%%
# Plotting functions # TODO - change to importing from utils
def plot_FK(q_repl):
    FK_evals = get_FK(q_repl)
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1],'tab:orange')
    ax.scatter(FK_evals[10,0],FK_evals[10,1],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[-1,0],FK_evals[-1,1],s=2,c='m',zorder=2.5)
    plt.xlim(FK_evals[0,0]-0.8,FK_evals[0,0]+0.8)
    plt.ylim(FK_evals[0,1]-0.8,FK_evals[0,1]+0.2)
    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    ax.grid(True)
    plt.show()

def get_FK(q_repl,num_pts=21):
    s_evals = np.linspace(0,1,num_pts)
    FK_evals = np.zeros((s_evals.size,2,1))
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(eval_fk(q_repl,p_vals,s_evals[i_s],0.0))
    return FK_evals.squeeze()

#%%
# RHS ODE

def f_dyn(t, y, F):
    q = y[:5]
    dq = y[5:]

    if abs(q[0]) < 1e-5:
        q[0] = 1e-5
    if abs(q[1]) < 1e-5:
        q[1] = 1e-5

    G = eval_G(q)
    B = eval_B(q)

    LHS = (-G -K@q -D@dq + F).transpose().squeeze() # Not using C matrix

    return np.concatenate((dq, np.linalg.inv(B)@LHS))

#%%
# Import functions # TODO - change to loading from module
# Constant parameters
m_L, m_E, L, D = sm.symbols('m_L m_E L D')  # m_L - total mass of cable, m_E - mass of weighted end
p = sm.Matrix([m_L, m_E, L, D])
# Configuration variables
theta_0, theta_1, x, z, phi = sm.symbols('theta_0 theta_1 x z phi')
theta = sm.Matrix([theta_0, theta_1])
q = sm.Matrix([theta_0, theta_1, x, z, phi])
# Integration variables
s, d = sm.symbols('s d')

# Load forward kinematics
f_FK = sm.lambdify((q,p,s,d), pickle.load(open("../generated_functions/fixed/fk", "rb")), "mpmath")
# Load EOM functions, replace constant parameters
p_vals = [1.0, 0.5, 1.0, 0.1]
F_G = pickle.load(open("../generated_functions/fixed/G", "rb"))
F_B = pickle.load(open("../generated_functions/fixed/B", "rb"))
F_G = F_G.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
F_B = F_B.subs([(m_L,p_vals[0]),(m_E,p_vals[1]),(L,p_vals[2]),(D,p_vals[3])])
f_G = sm.lambdify(q, F_G, "mpmath")
f_B = sm.lambdify(q, F_B, "mpmath")
# Convenience functions to extract real floats from complex mpmath matrices
def eval_G(q): return np.array(f_G(q[0],q[1],q[2],q[3],q[4]).apply(mp.re).tolist(), dtype=float)
def eval_B(q): return np.array(f_B(q[0],q[1],q[2],q[3],q[4]).apply(mp.re).tolist(), dtype=float)
def eval_fk(q, p_vals, s, d): return np.array(f_FK(q, p_vals, s, d).apply(mp.re).tolist(), dtype=float)

#%%
# Stiffness and damping

k_o = 0.1
k_x = 0
k_z = 1e4
k_phi = 0
b_o = 0.1
b_x = 1e-1
b_z = 1e-1
b_phi = 1e-1

K = np.array([[k_o,        1/2*k_o,    0,      0,      0    ],
              [1/2*k_o,    1/3*k_o,    0,      0,      0    ],
              [0,          0,          k_x,    0,      0    ],
              [0,          0,          0,      k_z,    0    ],
              [0,          0,          0,      0,      k_phi]])

D = np.array([[b_o,        1/2*b_o,    0,      0,      0    ],
              [1/2*b_o,    1/3*b_o,    0,      0,      0    ],
              [0,          0,          b_x,    0,      0    ],
              [0,          0,          0,      b_z,    0    ],
              [0,          0,          0,      0,      b_phi]])

#%%
# Set up

# Actuation
F = np.zeros((5,))

# Initial conditions
q_0 = np.array([1.0, 1.0, 0.0, 0.0, np.pi])
dq_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#%%
# Solve ODE
q_ev = solve_ivp(f_dyn, [0, 1.0], np.concatenate((q_0, dq_0)), method='BDF', args=(F,))
# TODO - previously integration solver crashes just after 3.75s. Now takes really long (maybe still crash, haven't tested to 3.75s)

#%%
import matplotlib
matplotlib.use("Agg")

FPS = 30
t_reg = np.arange(0, q_ev.t[-1], 1/FPS)
q_reg = np.zeros((t_reg.size, 5))
for i in range(5):
    q_reg[:,i] = np.interp(t_reg, q_ev.t, q_ev.y[i,:].transpose())

fig, ax = plt.subplots()
curve, = plt.plot([], [])
plt.xlim(-(p_vals[2]+0.1), (p_vals[2]+0.1))
plt.ylim(-(p_vals[2]+0.1), 0.1)
ax.set_aspect('equal')
ax.grid(True)

writer = FFMpegWriter(fps=FPS)
with writer.saving(fig, '../sim_videos/' + 'test.mp4', 200):
    for frame in range(t_reg.size-1):
        FK_evals = get_FK(q_reg[frame,:])
        curve.set_data(FK_evals[:,0], FK_evals[:,1])
        ax.set_xlim(FK_evals[0,0]-1.2,FK_evals[0,0]+1.2)
        ax.set_ylim(FK_evals[0,1]-1.2,FK_evals[0,1]+1.2)

        writer.grab_frame()

    print("Finished")
    plt.close(fig)

# %%
# # Old ODE solver
# from scipy.integrate import odeint
# q_ev = odeint(f_dyn, np.concatenate((q_0, dq_0)), t = np.linspace(0, 0.2, 101), args = (F,), tfirst=True)