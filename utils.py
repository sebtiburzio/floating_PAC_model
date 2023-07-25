import numpy as np
import matplotlib.pyplot as plt

def rot_XZ_on_Y(XZs,angles):
    # HACK - the angles are -ve because R_angles needs to be transposed for einsum to work
    # I don't know how to get einsum to work otherwise
    R_angles = np.array([[np.cos(-angles), np.sin(-angles)], 
                        [-np.sin(-angles), np.cos(-angles)]]).T
    if len(XZs.shape) == 1:
        return R_angles@XZs
    else:
        return np.einsum('ijk,ik->ij', R_angles, XZs)

def get_FK(p_vals,q_repl,f_fk,num_pts=21):
    s_evals = np.linspace(0,1,num_pts)
    FK_evals = np.zeros((s_evals.size,2,1))
    for i_s in range(s_evals.size):
       FK_evals[i_s] = np.array(f_fk(q_repl,p_vals,s_evals[i_s],0.0))
    return FK_evals.squeeze()

# Plot FK based on theta config and optionally an fk target for comparison
def plot_FK(p_vals,q_repl,f_fk,fk_targets=None):
    FK_evals = get_FK(p_vals,q_repl,f_fk)
    fig, ax = plt.subplots()
    ax.plot(FK_evals[:,0],FK_evals[:,1],'tab:orange')
    ax.scatter(FK_evals[10,0],FK_evals[10,1],s=2,c='m',zorder=2.5)
    ax.scatter(FK_evals[-1,0],FK_evals[-1,1],s=2,c='m',zorder=2.5)
    plt.xlim(FK_evals[0,0]-1.1*p_vals[2],FK_evals[0,0]+1.1*p_vals[2])
    plt.ylim(FK_evals[0,1]-1.1*p_vals[2],FK_evals[0,1]+1.1*p_vals[2])

    if fk_targets is not None:
        plt.scatter(0,0,c='tab:red',marker='+')
        plt.scatter(fk_targets[0],fk_targets[1],c='tab:green',marker='+')
        plt.scatter(fk_targets[2],fk_targets[3],c='tab:blue',marker='+')

    fig.set_figwidth(8)
    ax.set_aspect('equal','box')
    ax.grid(True)
    plt.show()

def plot_fk_targets(fk_targets,i):
    plt.scatter(0,0,c='tab:red',marker='+')
    plt.scatter(fk_targets[i,0],fk_targets[i,1],c='tab:green',marker='+')
    plt.scatter(fk_targets[i,2],fk_targets[i,3],c='tab:blue',marker='+')
    plt.axis('equal')
    plt.grid(True)

# target_evaluators = [eval_midpt, eval_endpt, eval_J_midpt, eval_J_endpt]
def find_curvature(p_vals,theta_guess,target_evaluators,fk_target,epsilon=0.01,max_iterations=10):  
    error_2norm_last = np.inf
    for i in range(max_iterations):
        error = (np.vstack([target_evaluators[0](theta_guess,p_vals), target_evaluators[1](theta_guess,p_vals)]) - fk_target.reshape(4,1))
        error_2norm = np.linalg.norm(error)
        if error_2norm < epsilon:
            print("Converged after " + str(i) + " iterations")
            return theta_guess, True
        else:
            if np.isclose(error_2norm, error_2norm_last):
                print("Error stable after iteration " + str(i))
                return theta_guess, False
            elif error_2norm > error_2norm_last:
                print("Error increasing after iteration " + str(i))
                return theta_guess_last, False
            else:
                theta_guess_last = theta_guess
                error_2norm_last = error_2norm
                J = np.vstack([target_evaluators[2](theta_guess, p_vals), target_evaluators[3](theta_guess, p_vals)])
                theta_guess = theta_guess - (np.linalg.pinv(J)@error).squeeze()
    print("Max iterations reached (check why)")
    return theta_guess, False