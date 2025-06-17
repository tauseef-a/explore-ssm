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
#
#                                  | (-1)          i<j        B = [bi].T where bi=(2i+1)(-1)^i
#   A =   [aij] where aij = (2i+1) | (-1)^(i-j+1)  i>=1       C = P(theta) where P is shifted legendre polynomial
#                                                             D = 0
#
def  get_ssm_delay_nw(q):
    A = np.zeros((q,q))
    B = np.zeros((q,1))
    for i in range(q):
        for j in range(q):
            A[i,j] = np.pow(-1,(i-j+1)) if i>=j else (-1)
            A[i,j] *=(2*i+1)
        B[i,0] = (2*i+1)*np.pow((-1),i)
    C = np.ones((1,q))
    X = np.zeros((q,1))
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
    A,B,C = get_ssm_delay_nw(q)
    Aa,Bb,Cc = discretize(A/theta, B/theta, C, step)
    
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
