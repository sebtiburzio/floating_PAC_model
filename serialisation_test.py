#%%
import sympy as sm
import dill
import cloudpickle
import pickle

#%%
# Test example for stackexchange question

a,b = sm.symbols('a b')

# Simple function
f = sm.integrate(a*b**2, (b,0,1))
F = sm.lambdify((a,b), f, "mpmath")

# Fails with below error:
# PicklingError: Can't pickle : it's not the same object as mpmath.ctx_iv.ivmpf
F_dump = dill.dumps(F)

# Works:
F_dump = dill.dumps(F, recurse=True)
# And loads properly:
F_load = dill.loads(F_dump)

# Function with Fresnel integral
f_fresnel = sm.integrate(sm.cos(a*b**2), (b,0,1))
F_fresnel = sm.lambdify((a,b), f_fresnel, "mpmath")

# Fails with same error as before:
# PicklingError: Can't pickle : it's not the same object as mpmath.ctx_iv.ivmpf
F_fresnel_dump = dill.dumps(F_fresnel)

# Fails after 30s with below error:
# RecursionError: maximum recursion depth exceeded in comparison
F_fresnel_dump = dill.dumps(F_fresnel, recurse=True)

# Cloudpickle works for saving:
F_dump = cloudpickle.dumps(F_fresnel)
# But fails with below error when trying to load:
# TypeError: mpq.__new__() missing 1 required positional argument: 'p'
F_load = cloudpickle.loads(F_dump)