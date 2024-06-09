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
  def linear_trending_without_noise(self, m=2, offset=5):
    dataset = (np.arange(self.num_elements) / self.num_elements - 1/m) * m + offset
    return dataset

  def linear_trending_with_gaussian_noise(self, m=2, offset=5, std=0.1, noise_factor=1):
    linear_trending = self.linear_trending_without_noise(m=m, offset=offset)
    dataset = linear_trending + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset
  
  def linear_trending_with_non_gaussian_noise(self, m=2, offset=5, num_spikes=5, noise_factor=1):
    dataset = self.linear_trending_without_noise(offset)
    indices = np.random.choice(dataset.size, num_spikes, replace=False)  # Select 5 random indices
    mean = np.mean(dataset)
    std_dev = np.std(dataset)
    outliers = mean + noise_factor * std_dev * np.random.randn(len(indices))  # Generating outliers
    dataset[indices] = outliers
    return dataset
  
  def linear_trending_with_both_noise(self, m=2, offset=5, std=0.1, gnoise_factor=1, num_spikes=5, ngnoise_factor=1):
    dataset = self.linear_trending_with_non_gaussian_noise(num_spikes=num_spikes, offset=offset, noise_factor=ngnoise_factor) + np.random.normal(0, std, self.num_elements) / gnoise_factor
    return dataset
  
  def periodical_sinusoidal_without_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0):
    dataset = offset + amplitude * np.sin(frequency * np.arange(self.num_elements) + phase)
    return dataset

  def periodical_sinusoidal_with_gaussian_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, std=0.1, noise_factor=1):
    periodical = self.periodical_sinusoidal_without_noise(offset, amplitude, frequency, phase)
    dataset = periodical + np.random.normal(0, std, self.num_elements) / noise_factor
    return dataset

  def periodical_sinusoidal_with_non_gaussian_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, num_spikes=5, noise_factor=1):
    dataset = self.periodical_sinusoidal_without_noise(offset)
    indices = np.random.choice(dataset.size, num_spikes, replace=False)  # Select 5 random indices
    mean = np.mean(dataset)
    std_dev = np.std(dataset)
    outliers = mean + noise_factor * std_dev * np.random.randn(len(indices))  # Generating outliers
    dataset[indices] = outliers
    return dataset

  def periodical_sinusoidal_with_both_noise(self, offset=5, amplitude=1, frequency=0.1, phase=0, std=0.1, gnoise_factor=1, num_spikes=5, ngnoise_factor=1):
    dataset = self.periodical_sinusoidal_with_non_gaussian_noise(num_spikes=num_spikes, offset=offset, noise_factor=ngnoise_factor) + np.random.normal(0, std, self.num_elements) / gnoise_factor
    return dataset

  ###########################
  ### MULTIPLE TIMESERIES ###
  ###########################
  def financial(self, corr_start_end=(0.8, 1), mean_return=0.0005, volatility=0.02, initial_price=100):
    # correlation matrix between timeseries
    start_range, end_range = corr_start_end
    lower_triangular = np.tril(np.random.uniform(start_range, end_range, size=(self.num_features, self.num_features)), k=-1)
    corr = lower_triangular + lower_triangular.T
    np.fill_diagonal(corr, 1)

    # Define parameters
    mean_returns = np.full(self.num_features, mean_return)  # Mean daily return

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