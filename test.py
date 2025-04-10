from package_sampling.sampling import up_brewer
import numpy as np

probabilities = np.array([0.1, 0.2, 0.3, 0.4])

# Draw a sample using Brewer's method
sample = up_brewer(probabilities)
print(sample)
