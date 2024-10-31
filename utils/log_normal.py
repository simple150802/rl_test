import numpy as np
import pandas as pd
from scipy.optimize import minimize

class LogNormalExecutionSimulator:
    def __init__(self, percentiles:list, arrival_time:list):
        # Load percentiles and z-scores         
        self.percentiles = percentiles
        self.z_scores = [-2.326, -2.326, -0.674, 0, 0.674, 2.326, 2.326]
        
        # Log-transformed percentiles
        self.log_percentiles = np.log(self.percentiles)
        
        # Calculate MLE parameters for log-normal distribution
        self.mu_mle, self.sigma_mle = self.calculate_mle_params()
        
        # Calculate linear fit for sigma and mu
        self.sigma, self.mu = np.polyfit(self.z_scores, self.log_percentiles, 1)
        
        # Minimum and maximum times for adjustment
        self.min_time = self.percentiles[0]
        self.max_time = self.percentiles[-1]

        self.arrival_time = arrival_time
            
    def neg_log_likelihood(self, params, data):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        log_likelihood = -np.sum(
            -np.log(data) - 0.5 * np.log(2 * np.pi * sigma**2) -
            ((np.log(data) - mu)**2) / (2 * sigma**2)
        )
        return -log_likelihood
    
    def calculate_mle_params(self):
        initial_params = [np.mean(self.log_percentiles), np.std(self.log_percentiles)]
        result = minimize(self.neg_log_likelihood, initial_params, args=(self.percentiles,), method='L-BFGS-B')
        mu_mle, sigma_mle = result.x
        return mu_mle, sigma_mle
    
    def generate_execution_times(self):
        execution_times = np.random.lognormal(mean=self.mu_mle, sigma=self.sigma_mle, size=len(self.arrival_time))
        adjusted_times = np.clip(execution_times, self.min_time, self.max_time)
        return adjusted_times
    
if __name__ == '__main__':
# Usage example:
    simulator = LogNormalExecutionSimulator(
        "/content/filtered_duration_output_7_days_2f0f0e5ef55603f948a92533cc3fc43d9db6e7eeac8914c2cf86d381675fbe94.csv",
        "/content/filtered_output_7_days_2f0f0e5ef55603f948a92533cc3fc43d9db6e7eeac8914c2cf86d381675fbe94.csv"
    )
    all_execution_times, execution_times_per_minute = simulator.generate_execution_times()

