import pandas as pd
from sklearn.decomposition import PCA

data=pd.read_excel('/Users/siaga/Desktop/pca_processed.xlsx',index_col=0)
data=data.T
pca=PCA(n_components=2)
pComponents=pca.fit_transform(data)
print(pca.explained_variance_ratio_)
pcaData=pd.DataFrame(data=pComponents,columns = ['principal component 1', 'principal component 2'])
pcaData.index=data.index
pcaData.to_excel('/Users/siaga/Desktop/pca_results.xlsx')