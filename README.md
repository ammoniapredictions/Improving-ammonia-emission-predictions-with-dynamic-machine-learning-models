This repository provides the full code and data used to reproduce the results presented in the article.

- The analyses were performed using both **R** and **Python**:
  - **R** was used to run the **ALFAM2** model and the **random forest** models.
  - **Python** was used to implement and train the **neural network** models.

- All required R and Python packages (with versions) are listed in the [`dependencies.txt`](dependencies.txt) file.

- `00_data/`: contains the raw input data from **ALFAM2** (version 2.56), downloaded from  
  [https://github.com/AU-BCE-EE/ALFAM2](https://github.com/AU-BCE-EE/ALFAM2)

- `02_scripts/`: contains the code used for data preparation and model implementation (ALFAM2, random forest, and neural networks).

- `results.ipynb`: this notebook compiles and presents all the results shown in the article.
