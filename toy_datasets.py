import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class ToyDatasets():
  def __init__(self, num_elements=100, num_features=5):
    self.num_elements = num_elements
    self.num_features = num_features

  def set_num_elements(self, num_elements):
    self.num_elements = num_elements

  def set_num_features(self, num_features):
    self.num_features = num_features

  def visualise_dataset(self, dataset, title):
    plt.plot(dataset)
    plt.xlabel("Timestep"), plt.ylabel("Value"), plt.title(title)

  #########################
  ### SINGLE TIMESERIES ###
  #########################
  def stationary_without_noise(self, offset=5):
    dataset = np.full(self.num_elements, offset)
    return dataset
  
  def stationary_with_gaussian_noise(self, offset=5, std=0.1, noise_factor=1):
    dataset = np.random.normal(offset, std, self.num_elements) / noise_factor
    return dataset
  
  def stationary_with_non_gaussian_noise(self, dof=3, offset=5, noise_factor=10):
    dataset = offset + np.random.standard_t(dof, self.num_elements) / noise_factor
    return dataset
  
  def stationary_with_both_noise(self, offset=5, std=0.1, gnoise_factor=1, dof=3, ngnoise_factor=10):
    dataset = np.random.normal(offset, std, self.num_elements) / gnoise_factor + np.random.standard_t(dof, self.num_elements) / ngnoise_factor
    return dataset

  def linear_trending_without_noise(self, m=2, offset=5):
    dataset = (np.arange(self.num_elements) / self.num_elements - 1/m) * m + offset
    return dataset

  def linear_trending_with_gaussian_noise(self, m=2, offset=5, std=0.1, noise_factor=1):
    linear_trending = self.linear_trending_without_noise(m=m, offset=offset)
    dataset = linear_trending + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset
  
  def linear_trending_with_non_gaussian_noise(self, m=2, offset=5, dof=3, noise_factor=10):
    linear_trending = self.linear_trending_without_noise(m=m, offset=offset)
    dataset = linear_trending + np.random.standard_t(dof, self.num_elements) / noise_factor
    return dataset
  
  def linear_trending_with_both_noise(self, m=2, offset=5, std=0.1, gnoise_factor=1, dof=3, ngnoise_factor=10):
    dataset = self.linear_trending_with_gaussian_noise(m=m, offset=offset, std=0.1, noise_factor=gnoise_factor)
    dataset = dataset + np.random.standard_t(dof, self.num_elements) / ngnoise_factor
    return dataset    
  
  def periodical_linear_without_noise(self, m=2, offset=5):
    dataset = np.tile(
      np.concatenate((
        (np.arange(self.num_elements / 4) / (self.num_elements / 4) - 1 / m) * m + offset,
        (np.arange(self.num_elements / 4) / (self.num_elements / 4) - 1 / m) * -m + offset
      )), 2
    )
    return dataset
  
  def periodical_linear_with_gaussian_noise(self, m=2, offset=5, std=0.1, noise_factor=1):
    periodical = self.periodical_linear_without_noise(m, offset)
    dataset = periodical + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset
  
  def periodical_linear_with_non_gaussian_noise(self, m=2, offset=5, dof=3, noise_factor=10):
    periodical = self.periodical_linear_without_noise(m, offset)
    dataset = periodical + np.random.standard_t(dof, self.num_elements) / noise_factor
    return dataset

  def periodical_linear_with_both_noise(self, m=2, offset=5, std=0.1, gnoise_factor=1, dof=3, ngnoise_factor=10):
    dataset = self.periodical_linear_with_gaussian_noise(m=m, offset=offset, std=std, noise_factor=gnoise_factor)
    dataset = dataset + np.random.standard_t(dof, self.num_elements) / ngnoise_factor
    return dataset
  
  def periodical_sinusoidal_without_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0):
    dataset = offset + amplitude * np.sin(frequency * np.arange(self.num_elements) + phase)
    return dataset

  def periodical_sinusoidal_with_gaussian_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, std=0.1, noise_factor=1):
    periodical = self.periodical_sinusoidal_without_noise(offset, amplitude, frequency, phase)
    dataset = periodical + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset

  def periodical_sinusoidal_with_non_gaussian_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, dof=3, noise_factor=10):
    periodical = self.periodical_sinusoidal_without_noise(offset, amplitude, frequency, phase)
    dataset = periodical + np.random.standard_t(dof, self.num_elements) / noise_factor
    return dataset

  def periodical_sinusoidal_with_both_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, std=0.1, gnoise_factor=1, dof=3, ngnoise_factor=10):
    dataset = self.periodical_sinusoidal_with_gaussian_noise(offset=offset, amplitude=amplitude, frequency=frequency, phase=phase, std=std, noise_factor=gnoise_factor)
    dataset = dataset + np.random.standard_t(dof, self.num_elements) / ngnoise_factor
    return dataset

  def polynomial_without_noise(self, order=2, min_x=-1, max_x=1, y_offset=1):
    x = np.linspace(min_x, max_x, self.num_elements)
    dataset = y_offset + x**order
    return dataset

  def polynomial_with_gaussian_noise(self, order=2, min_x=-1, max_x=1, y_offset=1, std=0.1, noise_factor=1):
    polynomial = self.polynomial_without_noise(order, min_x, max_x, y_offset)
    dataset = polynomial + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset

  def polynomial_with_non_gaussian_noise(self, order=2, min_x=-1, max_x=1, y_offset=1, dof=3, noise_factor=10):
    polynomial = self.polynomial_without_noise(order, min_x, max_x, y_offset)
    dataset = polynomial + np.random.standard_t(dof, self.num_elements) / noise_factor
    return dataset

  ###########################
  ### MULTIPLE TIMESERIES ###
  ###########################
  def financial(self, num_features=3, corr_start_end=(0.8, 1), mean_return=0.0005, volatility=0.02, initial_price=100):
    # correlation matrix between timeseries
    start_range, end_range = corr_start_end
    lower_triangular = np.tril(np.random.uniform(start_range, end_range, size=(num_features, num_features)), k=-1)
    corr = lower_triangular + lower_triangular.T
    np.fill_diagonal(corr, 1)

    # Define parameters
    mean_returns = np.full(num_features, mean_return)  # Mean daily return

    # Generate random returns based on a normal distribution
    returns = np.random.multivariate_normal(mean=mean_returns, cov=corr, size=self.num_elements) * volatility

    # Calculate cumulative returns
    cumulative_returns = np.exp(np.cumsum(returns, axis=0))

    # Calculate prices
    prices = initial_price * cumulative_returns

    # Create DataFrame
    data = {}
    for i, price_timeseries in enumerate(prices.T):
      data[i] = price_timeseries

    df = pd.DataFrame(data)

    return df.to_numpy()