import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = np.load(r"D:\SSC\Data\Processed\X.npy")
wave = np.load(r"D:\SSC\Data\Processed\wavelength_grid.npy")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=5)
pca.fit(X_scaled)

components = pca.components_

for i, comp in enumerate(components):
    plt.figure(figsize=(10, 4))
    plt.plot(wave, comp)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Component Weight")
    plt.title(f"PCA Component {i+1}")
    plt.grid(True, alpha=0.3)
    plt.show()