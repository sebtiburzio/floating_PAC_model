#!/usr/bin/env python

#%matplotlib ipympl
import numpy as np
import dill
from scipy.integrate import solve_ivp
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#%%
# Plotting functions

def plot_FK(q_repl):
    s_evals = np.linspace(0,1,11)
    FK_evals = np.empty((s_evals.size,2,))
    FK_evals[:] = np.nan
    for i_s in range(s_evals.size):
       FK_evals[i_s] = f_FK(q_repl,p_vals,s_evals[i_s],0.0).squeeze()
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1])
    plt.xlim(FK_evals[0,0]-1.2,FK_evals[0,0]+1.2)
    plt.ylim(FK_evals[0,1]-1.2,FK_evals[0,1]+1.2)
    ax.set_aspect('equal','box')
    plt.show()

#%%
# RHS ODE

def f_dyn(t, y, F):
    q = y[:5]
    dq = y[5:]

    if abs(q[0]) < 1e-5:
        q[0] = 1e-5
    if abs(q[1]) < 1e-5:
        q[1] = 1e-5

    G = f_G(q, p_vals)  # TODO evaluate p_vals during function generation
    B = f_B(q, p_vals)

    C = np.zeros((5,5)) # TODO just remove C if not going to use
    # C = f_C(q, p_vals, dq)

    LHS = (-C@dq -G -K@q -D@dq + F).transpose().squeeze()

    return np.concatenate((dq, np.linalg.inv(B)@LHS))

#%%
# Import EOM functions

f_FK = dill.load(open('./generated_functions/f_FK','rb'))
f_G = dill.load(open('./generated_functions/f_G','rb'))
f_B = dill.load(open('./generated_functions/f_B','rb'))
# f_C = dill.load(open('./generated_functions/f_C','rb'))

#%%
# Stiffness and damping

k_o = 0.1
k_x = 1e4
k_z = 0
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

# Parameters
p_vals = [1.0, 0.5, 1.0, 0.1]

# Actuation
F = np.zeros((5,))

# Initial conditions
q_0 = np.array([1.0, 1.0, 0.0, 0.0, np.pi])
dq_0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

#%%
# Solve ODE

# TODO - integration solver crashes just after 3.75s
q_ev = solve_ivp(f_dyn, [0, 3.75], np.concatenate((q_0, dq_0)), method='BDF', args=(F,))

# Old ODE solver
# q_ev = odeint(f_dyn, np.concatenate((q_0, dq_0)), t = np.linspace(0, 1, 101), args = (F,), tfirst=True)

# %%
# Animated plot of evolution

# Init plot
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False)
ax.set_aspect('equal')
ax.grid()
line, = ax.plot([], [], '-', lw=2)

s_evals = np.linspace(0,1,11)
FK_evals = np.empty((s_evals.size,2,))
FK_evals[:] = np.nan

# Resample configuration evolution to even timestep
t_reg = np.arange(0,q_ev.t[-1], 1/60)
q_reg = np.zeros((t_reg.size, 5))
for i in range(5):
    q_reg[:,i] = np.interp(t_reg, q_ev.t, q_ev.y[i,:].transpose())

def animate(frame):
    for i_s in range(s_evals.size):
        FK_evals[i_s] = f_FK(q_reg[frame,:], p_vals, s_evals[i_s], 0.0).squeeze()
    line.set_data(FK_evals[:,0], FK_evals[:,1])
    ax.set_xlim(FK_evals[0,0]-1.2,FK_evals[0,0]+1.2)
    ax.set_ylim(FK_evals[0,1]-1.2,FK_evals[0,1]+1.2)
    return line,

ani = animation.FuncAnimation(fig, animate, len(t_reg), interval=1/60, blit=True)
plt.show()

# TODO
# check if issues evalauting fk at -ve curvature?
# trry without high stifffness on z - might be causing issues?
# %%
