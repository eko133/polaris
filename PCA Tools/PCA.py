import pandas as pd
from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

data=pd.read_excel('/Users/siaga/Desktop/pca_processed.xlsx',index_col=0)
data=data.T
pca=PCA(n_components=2)
pComponents=pca.fit_transform(data)
print(pca.explained_variance_ratio_)
pcaData=pd.DataFrame(data=pComponents,columns = ['principal component 1', 'principal component 2'])
pcaData.index=data.index
loadings=pca.components_
loadings=loadings.T
loadings=pd.DataFrame(loadings)
loadings.index=data.T.index
loadings.to_excel('/Users/siaga/Desktop/loadings.xlsx')
pcaData.to_excel('/Users/siaga/Desktop/pca_results.xlsx')

# plt.scatter(*loadings, alpha=0.3, label='loadings')
# plt.grid()
# plt.show()
