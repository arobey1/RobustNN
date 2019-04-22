import numpy as np
from itertools import product


I = list(np.eye(3, dtype=int))
c = np.array(list(product(I, I)))
d = np.diff(c, axis=1)
d = np.squeeze(np.unique(d, axis=0))
# print(d)

# this works
e = np.einsum('ij,ik->ijk', d, d)
(num_layers, _, _) = e.shape

e[num_layers//2+1:, :, :] *= -1

const = np.arange(7).reshape(7,)
print(const.shape)
print(np.einsum('ijk,i->jk', e, const))

# print(np.append(e, -e, axis=0))
# f = np.append(e, -1. * e, axis=0)
# print(np.unique(f, axis=2))

# e = np.einsum('ij,ik->kji', d, d)
# print(e[:,:,1])
# print(e.shape)

# print(np.einsum('ijk,ikl->ijl',f, e))
# g = np.einsum('ijk,ikl->ijl',f, e)
# print(g[:,:,1s])
# print(np.einsum('mnr,ndr->mdr', e, f))
# print(np.einsum('ijk,ikl->ijl', f, e))
# print(np.matmul(e, f))
