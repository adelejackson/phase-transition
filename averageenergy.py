import numpy as np
from scipy.linalg import expm
from scipy import integrate
import pylab as pl
f = open('energy.txt', 'r')

pl.title("Average energy of each bond")
pl.xlabel("Value of h")
pl.ylabel("Average energy per bond")
dim = 0
m = 5
tau = 0.5
numsteps = 50
h = 0
start = 0
end = 2.0
step = 0.1
cache = {}

for line in f:
    a, b, c = line.strip().split()
    a = int(a)
    c = float(c)
    cache[(a, b)] = c
f.close()
f = open('energy.txt', 'a')
hx = np.array([[0, 1], [1, 0]])
hz = np.array([[1, 0], [0, -1]])


values = []
for dim in range(4, 149, 12):
    vector = np.random.rand(2)
    vector = vector/np.sqrt(sum(vector**2))
    vector = vector.reshape(1, 2, 1)
    bond = np.array([1])
    matrices = ([vector for x in xrange(dim)])
    matrices[0] = vector.reshape(2, 1)
    matrices[-1] = vector.reshape(1, 2)
    bonds = [bond for x in xrange(dim-1)]
    values = []
    for h in np.arange(start, end, step):
        if (dim, str(h)) in cache:
            values.append(cache[(dim, str(h))])
            continue
        operator = expm(tau*(np.kron(hz, hz)+h*np.kron(hx, np.eye(2)))).reshape(2, 2, 2, 2)
        lastoperator = expm(tau*(np.kron(hz, hz) + h*np.kron(hx, np.eye(2)) + h*np.kron(np.eye(2), hx))).reshape(2, 2, 2, 2)
        for i in range (numsteps):
            F = matrices[0]
            F = np.tensordot(F, np.diag(bonds[0]), (-1, 0))
            F = np.tensordot(F, matrices[1], (-1, 0))
            Fop = np.tensordot(F, operator, ([0, 1], [0, 1]))
            Fop = np.transpose(Fop, (1, 2, 0))
            Fop = np.tensordot(Fop, np.diag(bonds[1]), (-1, 0))
            Fop = Fop.reshape(Fop.shape[0], -1)
            U, S, V = np.linalg.svd(Fop, full_matrices = False)
            r = min(len(S[np.nonzero(S)]), m)
            U = U[:,:r].reshape(2, -1)
            V = V[:r,:].reshape(r, 2, -1)
            S = S[:r]/np.sqrt(sum(S[:r]**2))
            V = np.tensordot(V, np.linalg.inv(np.diag(bonds[1])), (-1, 0))
            matrices[0] = U
            matrices[1] = V
            bonds[0] = S
            for j in range(1, dim-2):
                F = np.tensordot(np.diag(bonds[j-1]), matrices[j], (1, 0))
                F = np.tensordot(F, np.diag(bonds[j]), (-1, 0))
                F = np.tensordot(F, matrices[j+1], (-1, 0))
                Fop = np.tensordot(F, operator, ([1, 2], [0, 1]))
                Fop = np.transpose(Fop, (0, 2, 3, 1))
                Fop = np.tensordot(Fop, np.diag(bonds[j+1]), (-1, 0))
                shapes = Fop.shape
                Fop = Fop.reshape(shapes[0]*shapes[1], -1)
                U, S, V = np.linalg.svd(Fop, full_matrices = False)
                U1, S1, V1 = np.linalg.svd(F, full_matrices = False)
                r = min(len(S[np.nonzero(S)]), m)
                U = U[:,:r].reshape(-1, 2, r)
                V = V[:r,:].reshape(r, 2, -1)
                U = np.tensordot(np.linalg.inv(np.diag(bonds[j-1])), U, (1, 0))
                S = S[:r]/np.sqrt(sum(S[:r]**2))
                V = np.tensordot(V, np.linalg.inv(np.diag(bonds[j+1])), (-1, 0))
                matrices[j] = U
                bonds[j] = S
                matrices[j+1] = V
            j = dim-2
            F = np.tensordot(np.diag(bonds[j-1]), matrices[j], (1, 0))
            F = np.tensordot(F, np.diag(bonds[j]), (-1, 0))
            F = np.tensordot(F, matrices[j+1], (-1, 0))
            Fop = np.tensordot(F, lastoperator, ([1, 2], [0, 1]))
            shapes = Fop.shape
            Fop = Fop.reshape(shapes[0]*shapes[1], -1)
            U, S, V = np.linalg.svd(Fop, full_matrices = False)
            r = min(len(S[np.nonzero(S)]), m)
            U = U[:,:r].reshape(-1, 2, r)
            V = V[:r,:].reshape(r, 2)
            U = np.tensordot(np.linalg.inv(np.diag(bonds[j-1])), U, (1, 0))
            S = S[:r]/np.sqrt(sum(S[:r]**2))
            matrices[j] = U
            bonds[j] = S
            matrices[j+1] = V
        F = np.tensordot(matrices[0], matrices[0].conj(), (0, 0))
        for i in range(1, dim):
            F = np.tensordot(F, np.diag(bonds[i-1]), (0, 0))
            F = np.tensordot(F, np.diag(bonds[i-1]), (0, 0))
            F = np.tensordot(F, matrices[i], (0, 0))
            F = np.tensordot(F, matrices[i], ([0, 1], [0, 1]))
        print dim
        for i in range(len(matrices)):
            matrices[i] = matrices[i]/np.sqrt(F)
        for i in range(len(bonds)):
            bonds[i] = bonds[i]/np.sqrt(F)
    
        
        resultop = (np.kron(hz, hz)+h*np.kron(hx, np.eye(2))).reshape(2, 2, 2, 2)
        lastresultop = (np.kron(hz, hz) + h*np.kron(hx, np.eye(2)) + h*np.kron(np.eye(2), hx)).reshape(2, 2, 2, 2)
        
        value = 0
        F = np.tensordot(matrices[0], np.diag(bonds[0]), (-1, 0))
        F = np.tensordot(F, matrices[1], (-1, 0))
        F = np.tensordot(F, np.diag(bonds[1]), (-1, 0))
        Fop = np.tensordot(F, resultop, ([0, 1], [1, 2]))
        Fop = np.transpose(Fop, (1, 2, 0))
        F = np.tensordot(Fop, F, ([0, 1], [0, 1]))
        F = np.tensordot(F, matrices[2], (0, 0))
        F = np.tensordot(F, matrices[2], ([0, 1], [0, 1]))
        for i in range(3, dim):
            F = np.tensordot(F, np.diag(bonds[i-1]), (0, 0))
            F = np.tensordot(F, np.diag(bonds[i-1]), (0, 0))
            F = np.tensordot(F, matrices[i], (0, 0))
            F = np.tensordot(F, matrices[i], ([0, 1], [0, 1]))
        #print F
        value += F
        for i in range(1, dim-2):
            F = np.tensordot(np.diag(bonds[i-1]), matrices[i], (1, 0))
            F = np.tensordot(F, np.diag(bonds[i]), (-1, 0))
            F = np.tensordot(F, matrices[i+1], (-1, 0))
            Fop = np.tensordot(F, resultop, ([1, 2], [1, 2]))
            Fop = np.transpose(Fop, (0, 2, 3, 1))
            Fop = np.tensordot(Fop, F, ([1, 2], [1, 2]))
            F = np.tensordot(matrices[0], matrices[0], (0, 0))
            for j in range(1, dim):
                if j == i:
                    F = np.tensordot(F, Fop, ([0, 1], [0, 2]))
                    continue
                if j == i+1:
                    continue
                F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
                F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
                F = np.tensordot(F, matrices[j], (0, 0))
                F = np.tensordot(F, matrices[j], ([0, 1], [0, 1]))
            #print F
            value += F
        F = np.tensordot(np.diag(bonds[dim-3]), matrices[dim-2], (1, 0))
        F = np.tensordot(F, np.diag(bonds[dim-2]), (-1, 0))
        F = np.tensordot(F, matrices[dim-1], (-1, 0))
        Fop = np.tensordot(F, lastresultop, ([1, 2], [1, 2]))
        Fop = np.tensordot(Fop, F, ([1, 2], [1, 2]))
        F = np.tensordot(matrices[0], matrices[0], (0, 0))
        for j in range(1, dim-1):
            if j == dim-2:
                F = np.tensordot(F, Fop, ([0, 1], [0, 1]))
                continue
            F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
            F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
            F = np.tensordot(F, matrices[j], (0, 0))
            F = np.tensordot(F, matrices[j], ([0, 1], [0, 1]))
        #print F
        value += F
        values.append(-1*value/dim)
        f.write("%d %s %f\n" % (dim, str(h), -1*value/dim))
    pl.plot(np.arange(start, end, step), values, 'x')
from scipy import integrate

f = lambda k,g : -2*np.sqrt(1+g**2-2*g*np.cos(k))/np.pi/2.
E0_exact=[]
for g in np.arange(0,2,0.01):
    E0_exact.append(integrate.quad(f, 0, np.pi, args=(g,))[0])
pl.plot(np.arange(0, 2, 0.01), E0_exact)
pl.legend(title="Number of particles")
pl.show()
