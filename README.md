# **README**

This code was developed by Ippocratis D. Saltas for the analysis presented in ***Ippocratis D. Saltas & Georgios Lukes-Gerakopoulos (2024)***. For a detailed presentation of the analysis/results see the published paper online.

#### **Introduction**

The **goal** of this project is to build a deep learning network classifier which is able to classify order and chaos in 2-dimensional, discreet orbits on the plane, which result from Hamiltonian systems. Such orbits result from the so--called Poicare sections or Poincare maps.

As our reference map we use the so--called Standard Map which is one of the most representative Poincare maps, and exhibits universal features of the way order and chaos emerges in Hamiltonian systems. We first use it to train and validate our deep model, and then we apply our trained model on other Poincare maps. We show that the trained ML model is able to identify chaos and order in other systems too with reasonably high accuracy, due to the universality features of chaotic/ordered dynamics of Hamiltonian systems. 

#### **The code's modules & folders**

 - functions.py: Definitions of functions generating the Poincare maps. 

 - generate_data.py: Data preparation and augmentation functions.

 - model.py: Deep network model, and testing functionality.

 - Main_notebook.ipynb: Main analysis + plots.

 - The folder "external_orbits" contains required additional orbits for testing which are not produced in this code. They are provided in tabulated column format.

#### **Package versions used**
 
- Numpy 1.24.1

- Tensorflow 2.15.0

- Keras 2.15.0. 

- The code was tested on a Jupyter notebook using a machine with a GPU processor.

#### **Execution**

All files should be placed in the same folder. The code runs just by executing the notebook "Main_notebook.ipynb" cell-by-cell. When loading/saving a model make sure you modify appropriately the path in the Jupyter notebook.
