import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition


def factorize(data, method='pca', n_components = 100):
    scaled_features = StandardScaler().fit_transform(data.values)
    scaled_data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)
    n_components = n_components
    if method == 'pca':
        factorization = decomposition.PCA(n_components=n_components)
    elif method == 'kpca':
        factorization = decomposition.KernelPCA(n_components=n_components, kernel='linear')
    elif method =='ica':
        factorization = decomposition.FastICA(n_components=n_components, random_state=0)
    elif method == 'fa':
        factorization = decomposition.FactorAnalysis(n_components=n_components, random_state=0)
    components = factorization.fit_transform(scaled_data)
    factorized_data = pd.DataFrame(data=components, columns=[f'F{i}' for i in range(1,n_components+1)])
    factorized_data.index = data.index
    # loadings = pca.components_
    # loadings = loadings.T
    # loadings = pd.DataFrame(loadings, columns=[f'PC{i}' for i in range(1,n_components+1)])
    # loadings.index = data.T.index
    return factorized_data
