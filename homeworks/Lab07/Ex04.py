import matplotlib.pyplot as plt
from scipy.stats import expon
import numpy as np

# Log-likelihood function for exponential distribution
def log_likelihood(lambda_val, data):
    return len(data) * np.log(lambda_val) - lambda_val * np.sum(data)


lambda_ = 1 # Once per year, on average
X = expon.rvs(scale=1/lambda_, size=100, random_state=1)
X[:3] # The first 3 intervals

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(X, bins=20, density=True, alpha=0.7, color='skyblue', label='Observed Data')

# Add the true exponential PDF
x = np.linspace(0, max(X), 100)
plt.plot(x, lambda_ * np.exp(-lambda_ * x), 'r-', lw=2, label=f'True PDF (λ={lambda_})')

plt.title('Histogram of Exponential Random Variables')
plt.xlabel('Time')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# First, let's do a coarse grid search to find approximate region
lambda_grid = np.linspace(0.1, 5, 1000)
ll_values = [log_likelihood(l, X) for l in lambda_grid]
lambda_experimental = lambda_grid[np.argmax(ll_values)]

print(f"Experimental MLE estimate for λ: {lambda_experimental:.4f}")


lambda_hat_analytical = 1/np.mean(X)
print(f"Analytical MLE estimate for λ: {lambda_hat_analytical:.4f}")
