import numpy as np
import matplotlib.pyplot as plt
from util.pearson_corr_coeff import pearson_corr_coeff
x = np.random.randint(-1,1,(2,100))
y = np.random.randint(-1,1,(2, 100))
x = np.sin(np.arange(0,100) - 10)
y = np.sin(np.arange(0,100))


#Css = x.T *x

np.delete(x,0,axis=0)

print(x.shape)

print(pearson_corr_coeff(x,y))

print(np.corrcoef(x,y))
plt.figure()
plt.imshow(np.corrcoef(x.squeeze(), y.squeeze()), cmap='plasma')
plt.colorbar()
plt.show()
