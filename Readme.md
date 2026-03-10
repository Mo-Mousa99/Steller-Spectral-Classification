# **Star Spectral Classification using Machine Learning and Deep Learning**



### Project Overview



This project builds a machine learning pipeline to classify stellar spectra into three temperature-based groups using data from the Sloan Digital Sky Survey (SDSS).



The system processes real astronomical spectra stored in FITS format, extracts spectral features, and compares classical machine learning methods with a deep learning model.



The goal is to demonstrate how modern artificial intelligence techniques can be applied to astrophysical data analysis.



\_\_\_\_\_\_\_



### Dataset



The dataset was obtained from the Sloan Digital Sky Survey (SDSS).



Spectra were retrieved using SQL queries from the SDSS SkyServer database.



Each spectrum represents the flux of a star measured across wavelengths from approximately 3800 Å to 9200 Å.



Stars were grouped into three classes based on spectral subclass:



|Group|Spectral Types|
|-|-|
|Hot|O, B, A|
|Medium|F, G|
|Cool|K, M|





After preprocessing, the dataset contains approximately:



2983 spectra



Each spectrum contains:



3846 wavelength samples



\_\_\_\_\_\_\_



### Data Processing Pipeline



The pipeline consists of the following stages:

1. Query stellar spectra from SDSS using SQL.
2. Download corresponding FITS files.
3. Extract spectral flux data and wavelength values.
4. Normalize spectra to remove magnitude differences.
5. Convert spectra into numerical arrays for machine learning.
6. Deep Learning CNN
7. Stellar Class Predection



\_\_\_\_\_\_\_



### Feature Extraction



Principal Component Analysis (PCA) was applied to reduce dimensionality.



###### Original feature space:



3846 wavelength features



###### Reduced feature space:



50 principal components



These components preserve approximately 95.6% of the spectral variance.



\_\_\_\_\_\_\_



### Methods



###### 1\. Preprocessing



\- FITS spectra were loaded using Astropy

\- Wavelength values were extracted from `loglam`

\- Flux values were normalized

\- Spectra were interpolated onto a common wavelength grid



This converts raw astronomical spectra into numerical arrays suitable for machine learning



###### 2\. Classical Machine Learning



\- Principal Component Analysis (PCA) reduced the 3846-dimensional spectra to 50 components

\- Logistic Regression and Random Forest were trained on PCA features



###### 3\. Deep Learning



\- A 1D Convolutional Neural Network (CNN) was trained directly on the normalized spectra. It learns local spectral patterns such as absorption line shapes and continuum variation.



\_\_\_\_\_\_\_



### Models Implemented



Three classification models were trained and evaluated.





#### Results



|Model|Accuracy|
|-|-|
|Logestic Regression|~93%|
|Random Forest|~92%|
|CNN|~97%|





The CNN achieved the best performance by learning local spectral features directly from the raw spectra.



\_\_\_\_\_\_\_



##### Example Results



###### CNN Confusion Matrix



!\[CNN Confusion Matrix](Figures/Figure\_6\_CNN\_Confusion\_Matrix.png)



###### PCA Spectral Components



!\[PCA Spectral Components](Figures/Figure\_4\_Visualize\_PCA.png)



\_\_\_\_\_\_\_



Quick Results Demo



If you want to quickly reproduce the results of the trained model without training:



1. Install dependencies

 	pip install -r requirements.txt

2\. Run:

Python Scripts/show\_results.py



This script will load the pretrained CNN model, and evaluate it on the processed dataset to print the classification report and display the confusion matrix.



\_\_\_\_\_\_\_

How to Run



1. Clone the repository
   git clone https://github.com/Mo-Mousa99/Steller-Spectral-Classification
   cd stellar-spectral-classification
2. Create and activate the environment

 	conda create -n star-classifier python=3.11 -y

 	conda activate star-classifier

 	pip install -r requirements.txt

3\. Run baseline model (Logestic Regression)

 	python scripts/train\_baseline.py

4\. Run Random Forest model

 	python scripts/random\_forest\_model.py

5\. Run the CNN model

 	python scripts/cnn\_model.py

6\. Test/Evaluate the trained model

 	python scripts/error\_analysis.py

7\. Visualize PCA components

 	python scripts/visualize\_pca.py



###### FITS Data From



https://www.sdss.org/

https://cas.sdss.org/dr19



###### Technologies Used



Python >= 3.10

Miniconda

NumPy

Scikit-learn

PyTorch

Astropy

Matplotlib

ChatGPT



\_\_\_\_\_\_\_



###### Possible Future Improvements



    •    Expanding classification to all seven stellar spectral types (O, B, A, F, G, K, M)

    •    Training on larger SDSS datasets

    •    Using deeper neural network architectures

    •    Applying the pipeline to galaxy or quasar classification tasks



\_\_\_\_\_\_\_



License



This project is intended for educational and research purposes.

