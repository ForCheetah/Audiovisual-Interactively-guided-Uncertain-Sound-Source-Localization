from scipy import optimize as op
import numpy as np

beta=[1,1]
c=np.array(beta)
a1=[2,5]
a2=[5,2]


A_ub=np.array([a1,a2])
B_ub=np.array([10,12])
res=op.linprog(-c,A_ub,B_ub)
print(int(-res.fun))
