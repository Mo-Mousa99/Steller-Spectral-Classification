import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load dataset
X = np.load(r"D:\SSC\Data\Processed\X.npy")
wave = np.load(r"D:\SSC\Data\Processed\wavelength_grid.npy")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=5)
pca.fit(X_scaled)

components = pca.components_

# Plot
plt.figure(figsize=(10,6))

for i, comp in enumerate(components):
    plt.plot(wave, comp, label=f"PC {i+1}")

plt.xlabel("Wavelength (Å)")
plt.ylabel("Component Weight")
plt.title("First 5 PCA Spectral Components")
plt.legend()

plt.show()