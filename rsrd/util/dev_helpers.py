from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

def plot_pca(X:torch.tensor , name="PCA", show=False, save=True):  
    # X is image feature (w, h, c)
    if len(X.shape) == 3:
        X = X.unsqueeze(0).detach().cpu()
    assert len(X.shape) == 4
    patch_h = X.shape[1]
    patch_w = X.shape[2]
    total_features = X.squeeze(0).squeeze(0).reshape(-1, X.shape[-1])
    pca = PCA(n_components=3)
    pca.fit(total_features)
    
    pca_features = pca.transform(total_features)

    # min_max scale
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / \
                        (pca_features[:, 0].max() - pca_features[:, 0].min())

    
    plt.imshow(pca_features[0 : patch_h*patch_w, 0].reshape(patch_h, patch_w), cmap='gist_rainbow')
    plt.axis('off')
    if save:
        plt.savefig(f'{name}.png', bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    