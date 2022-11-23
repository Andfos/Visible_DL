import numpy as np

a = np.array([[0, 2, 3], [4, 0, 6], [7, 8, 0]])
b = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

print(a)
print(b)

mask = a == 0
b[mask] = 0

print(mask)
print(b)




#b = np.nonzero(a)



