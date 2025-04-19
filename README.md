# **Propane Consumption Prediction using Hierarchical Bayesian Model**

## **Project Description**:
This project uses a **Hierarchical Bayesian Model** to predict **propane consumption** across multiple tanks in different regions. The model incorporates several features, such as **tank level**, **temperature**, and **usage rate**, and accounts for differences between **regions** using **group-level effects**. The model is implemented using **PyMC3** and utilizes **MCMC sampling** for inference.

## **Objective**:
The goal is to predict propane consumption in different regions and analyze how **tank level**, **temperature**, and **usage rate** influence the propane consumption. The model also considers **regional differences** in propane usage, which is handled using a hierarchical structure.

## **Libraries Used**:
- **PyMC3**: A Python library for Bayesian statistical modeling and probabilistic machine learning.
- **ArviZ**: A library for visualization and diagnostics of Bayesian models.
- **Matplotlib**: Used for plotting the results.
- **Pandas**: For handling and manipulating the data.
- **NumPy**: For numerical operations.

## **Steps in the Model**:
1. **Data Generation**: Synthetic data is generated for three regions (A, B, C), where each region contains several tanks with varying features.
2. **Data Preprocessing**: The data is encoded, and features like **tank level**, **temperature**, and **usage rate** are extracted along with the target variable **consumption**.
3. **Bayesian Modeling**: A **Hierarchical Bayesian Model** is built to predict propane consumption. The model includes:
   - **Hyperpriors** for overall mean and standard deviation of consumption.
   - **Region-specific effects** (group-level effects) to model differences between regions.
   - **Tank-level effects** (individual-level) based on features like **tank level**, **temperature**, and **usage rate**.
4. **Sampling**: The model is sampled using **MCMC** to obtain the posterior distributions of the parameters.
5. **Prediction**: After the model is trained, **posterior predictive sampling** is used to generate predictions and compare them to the actual consumption values.
6. **Visualization**: The results are visualized using various plots, including **posterior distributions** for model parameters and **scatter plots** for actual vs predicted consumption.

