import numpy as np
import matplotlib.pyplot as plt
from utilities.pearson_corr_coeff import pearson_corr_coeff
x = np.random.randint(-1,1,(1,10)).astype(np.float)
y = np.random.randint(-1,1,(1, 10))
x[0,2] = np.nan

#Css = x.T *x

#np.delete(x,0,axis=0)


print(pearson_corr_coeff(x,y))


print(np.corrcoef(x,y))
plt.figure()
plt.imshow(np.corrcoef(x.squeeze(), y.squeeze()), cmap='plasma')
plt.colorbar()
plt.show()
