import numpy as np
import matplotlib.pyplot as plt
import math
import argparse


parser = argparse.ArgumentParser(
                    prog='DelayNetwork',
                    description='Implements a delay network using ssm')
parser.add_argument('-q', '--padeapproximant', default = '6') 
parser.add_argument('-t', '--theta', default = '0.01')

#define input signal- A sinusoidal
def get_sin_input(k=3,N=500):
    n = np.arange(N)
    u = np.sin(2*np.pi*k/N*n)
    return u


#Get SSM for exp(-theta*s) a delay network
#Using Pade approximant of p/q with p=q-1 for numerical stability
#        -v0  -v0  ...  -v0
#         v1   0   ...   0     B = [v0  0  0  0  0].T
#   A =   0    v2  ...   0     C = [w0  w1  ...  w_q-1]
#         0   ...  ...   0     D = 0
#         0    0   v_q-1 0 
#   where theta is the delay
#   vi = (q+i)(q-i)/(i+1)* 1/theta     w_i = (i+1)/q*(-1)^(q-1-i)
#   v0 = q*(q!)*(q-1!)*(1/theta)
def  get_ssm_delay_nw(q,theta):
    v0 = q*(math.factorial(q))*(1/theta)*math.factorial(q-1)

    #Update A matrix
    A1 = np.array([np.full(q,-v0)])
    A2 = np.zeros((q-1,q))
    np.fill_diagonal(A2,[((q+i)*(q-i)/((i+1)* theta))for i in range(q-1)])
    A = np.vstack((A1,A2))

    #Update B matrix
    B = np.zeros((q,1))
    B[0] = v0

    #Update C Matrix
    C = np.array([(pow(-1,(q-1-i))*((i+1)/q)) for i in range(q)])
    C = C.reshape((1,q))

    return A,B,C

#ZOH Discretization
def discretize(A, B, C, step):
    I = np.eye(A.shape[0])
    BL = np.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb, C


if __name__=="__main__":
    args = parser.parse_args()

    q = int(args.padeapproximant)
    theta = float(args.theta)

    #Initialize Hidden State X
    X = np.zeros((q,1))
    u = get_sin_input()
    step = 3/500
    A,B,C = get_ssm_delay_nw(q,theta)
    Aa,Bb,Cc = discretize(A, B, C, step)
    
    Y = []
    u1 = u.reshape((-1,1,1))
    for i in range(u.shape[0]):
        X = np.dot(Aa,X) + np.dot(Bb,u1[i])
        res = np.dot(Cc,X)
        Y.append(res)
    y1 = np.hstack(Y).reshape(-1)
    
    plt.plot(u,label = 'input')
    plt.plot(y1,label = 'output')
    plt.legend()
    plt.savefig(f'plot-dn-{args.padeapproximant}-{args.theta}.png')
    plt.close()
