import matplotlib.pyplot as plt
import numpy as np

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
  def multiple_linear_trending_without_noise(self):
    pass

  def multiple_linear_trending_with_gaussian_noise(self):
    pass

  def multiple_linear_trending_with_non_gaussian_noise(self):
    pass

  def multiple_periodical_sinusoidals_without_noise(self):
    pass

  def multiple_periodical_sinusoidals_with_gaussian_noise(self):
    pass

  def multiple_periodical_sinusoidals_with_non_gaussian_noise(self):
    pass