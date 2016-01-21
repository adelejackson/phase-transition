import numpy as np
from scipy.linalg import expm
import pylab as pl
f = open('graph.txt', 'r')
pl.title("Square of the average expectation value")
pl.xlabel("Value of h")
pl.ylabel("Square of the average expectation value of the spins of the sites")
dim = 0
m = 10
tau = 0.5
numsteps = 500
h = 0
start = 0
end = 2.0
step = 0.1
cache = {}
hx = np.array([[0, 1], [1, 0]])
hz = np.array([[1, 0], [0, -1]])

for line in f:
    a, b, c = line.strip().split()
    a = int(a)
    c = float(c)
    cache[(a, b)] = c

f.close()
f = open('graph.txt', 'a')
values = []
for dim in [4, 8, 18, 36, 72, 144]:
#for dim in [6]:
    vector = np.random.rand(2)
    vector = vector/np.sqrt(sum(vector**2))
    vector = vector.reshape(1, 2, 1)
    bond = np.array([1])
    matrices = ([vector for x in xrange(dim)])
    matrices[0] = vector.reshape(2, 1)
    matrices[-1] = vector.reshape(1, 2)
    bonds = [bond for x in xrange(dim-1)]
    values = []
    xvalues = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9])
    for h in xvalues:
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
        for i in range(len(matrices)):
            matrices[i] = matrices[i]/np.sqrt(F)
        for i in range(len(bonds)):
            bonds[i] = bonds[i]/np.sqrt(F)
    
        value = 0
        for i in range(1, dim):
            F = np.tensordot(matrices[0], hz, (0, 0))
            F = np.tensordot(F, matrices[0], (-1, 0))
            for j in range(1, dim):
                F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
                F = np.tensordot(F, np.diag(bonds[j-1]), (0, 0))
                F = np.tensordot(F, matrices[j], (0, 0))
                if i == j:
                    F = np.tensordot(F, hz, (1, 0))
                    F = np.tensordot(F, matrices[j], ([0, -1], [0, 1]))
                    if i != dim-1:
                        F = np.tensordot(F, np.diag(bonds[j]), (0, 0))
                        F = np.tensordot(F, np.diag(bonds[j]), ([0, 1], [0, 1]))
                        break
                else:
                    F = np.tensordot(F, matrices[j], ([0, 1], [0, 1]))
            value += 2*F
        for i in range(1, dim):
            #print "doing", i
            for j in range(1, dim):
                if i == j:
                    continue
                F = np.tensordot(matrices[0], matrices[0], (0, 0))
                for k in range(1, dim):
                    F = np.tensordot(F, np.diag(bonds[k-1]), (0, 0))
                    F = np.tensordot(F, np.diag(bonds[k-1]), (0, 0))
                    F = np.tensordot(F, matrices[k], (0, 0))
                    if k == i or k == j:
                        F = np.tensordot(F, hz, (1, 0))
                        F = np.tensordot(F, matrices[k], ([0, -1], [0, 1]))
                    elif k == j:
                        F = np.tensordot(F, hz, (1, 0))
                        F = np.tensordot(F, matrices[k], ([0, -1], [0, 1]))
                        if k != dim-1:
                            F = np.tensordot(F, np.diag(bonds[j]), (0, 0))
                            F = np.tensordot(F, np.diag(bonds[j]), ([0, 1], [0, 1]))
                            break
                    else:
                        F = np.tensordot(F, matrices[k], ([0, 1], [0, 1]))
                value += F
        print dim, h, value/(dim*(dim-1))
        values.append(value/(dim*(dim-1)))
        f.write("%d %s %f\n" % (dim, str(h), value/(dim*(dim-1))))
#   pl.plot((xvalues-1)*dim, np.array(values)*dim**0.25, '-o', label=dim)
    pl.plot((xvalues), np.array(values), '-o', label=dim)
pl.legend(title="Number of sites")
pl.show()
