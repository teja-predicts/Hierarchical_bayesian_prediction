#import pymc3 as pm  # For Bayesian modeling and MCMC sampling
import numpy as np  # For numerical operations like generating random data
import pandas as pd  # For handling the data (e.g., creating DataFrame)
import arviz as az  # For diagnostics and visualizing the results of the model
import matplotlib.pyplot as plt  # For plotting the results
import pymc as pm

np.random.seed(42)

# Data for 3 regions (A, B, C) and multiple tanks
regions = ['A', 'B', 'C']
tank_data = {
    'Region': ['A']*5 + ['B']*5 + ['C']*5,  # Region A: 5 tanks, Region B: 5 tanks, Region C: 5 tanks
    'Tank_Level': np.random.uniform(50, 100, 15),  # Tank levels between 50 and 100
    'Temperature': np.random.uniform(-5, 35, 15),  # Temperature between -5 and 35
    'Usage_Rate': np.random.uniform(5, 15, 15),    # Usage rate between 5 and 15
    'Consumption': np.random.uniform(200, 500, 15)  # Propane consumption between 200 and 500
}

df = pd.DataFrame(tank_data)

# Encoding regions as integers (0 for A, 1 for B, 2 for C)
df['Region_idx'] = pd.Categorical(df['Region']).codes

regions = df['Region_idx'].unique()  # Get unique region identifiers
n_regions = len(regions)  # Number of unique regions
consumption = df['Consumption'].values  # Target variable: propane consumption
tank_level = df['Tank_Level'].values  # Feature: tank levels
temperature = df['Temperature'].values  # Feature: temperature
usage_rate = df['Usage_Rate'].values  # Feature: usage rate
region_idx = df['Region_idx'].values  # Encoded region indices (0, 1, 2)

with pm.Model() as model:
    # Step 1: Hyperpriors for overall consumption mean and standard deviation
    mu_c = pm.Normal('mu_c', mu=350, sigma=50)  # Overall mean of consumption
    sigma_c = pm.HalfNormal('sigma_c', sigma=50)  # Standard deviation of consumption

    # Step 2: Priors for region-specific consumption effects (group-level)
    region_effects = pm.Normal('region_effects', mu=mu_c, sigma=sigma_c, shape=n_regions)

       # Step 3: Priors for tank-level effects (tank level, temperature, and usage rate)
    tank_level_coeff = pm.Normal('tank_level_coeff', mu=0, sigma=10)
    temperature_coeff = pm.Normal('temperature_coeff', mu=0, sigma=10)
    usage_rate_coeff = pm.Normal('usage_rate_coeff', mu=0, sigma=10)

    # Step 4: Linear model for individual tank consumption
    tank_consumption = pm.Normal('tank_consumption', mu=region_effects[region_idx] + 
                                 tank_level_coeff * tank_level + 
                                 temperature_coeff * temperature + 
                                 usage_rate_coeff * usage_rate, 
                                 sigma=sigma_c, observed=consumption)
    
    # Step 5: Sampling (using MCMC)
    trace = pm.sample(2000, return_inferencedata=True)

    # Step 6: Analyze the results
    az.summary(trace, var_names=['mu_c', 'sigma_c', 'region_effects', 
                             'tank_level_coeff', 'temperature_coeff', 'usage_rate_coeff'])

    # Step 7: Visualize the results
    az.plot_posterior(trace, var_names=['mu_c', 'sigma_c'])
    plt.show()

    # Plot the region-specific effects (group-level effects)
    az.plot_posterior(trace, var_names=['region_effects'])
    plt.show()

    # Plot the coefficients for tank-level, temperature, and usage rate
    az.plot_posterior(trace, var_names=['tank_level_coeff', 'temperature_coeff', 'usage_rate_coeff'])
    plt.show()

    # Step 8: Check the trace for convergence
    az.plot_trace(trace)
    plt.show()
    plt.title("Trace")

    # Step 9: Compare the posterior predictions with actual data
    posterior_predictive = pm.sample_posterior_predictive(trace, model=model)

    print("Length of actual consumption:", len(consumption))
    print("Length of predicted values:", len(posterior_predictive))

    # Inspect the available keys in posterior_predictive
    print(f"Pred are: {posterior_predictive}")
    print(posterior_predictive)

    # Assuming the correct key for the predictions is 'posterior_predictive_tank_consumption'
    # If the key is different, replace 'posterior_predictive_tank_consumption' with the correct one

    # Step: Posterior predictive sampling
    posterior_predictive = pm.sample_posterior_predictive(trace, model=model)

    # Step: Access and properly mean over chains and draws
    predicted_values = posterior_predictive.posterior_predictive['tank_consumption'].mean(("chain", "draw")).values

    # Now predicted_values and consumption are same length

    print(f"Length of actual consumption: {len(consumption)}")
    print(f"Length of predicted values: {len(predicted_values)}")

    # Step: Plot
    plt.scatter(consumption, predicted_values)
    plt.xlabel('Actual Consumption')
    plt.ylabel('Predicted Consumption')
    plt.title('Actual vs Predicted Consumption')
    plt.show()


