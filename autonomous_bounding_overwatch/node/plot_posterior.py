from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import time

fig, ax = plt.subplots(figsize=(6,3))

bo_trust_change = [47.352219661197694, 41.06225759549388, 113.1830853215578,105.78360915366393,
                   75.35529126005285,46.79969791396768,138.95668479631993,76.20553920004842]
se_trust_change = [-298.38772196145084,-314.17070463823745,-275.29095367714734,-334.77686380804005,
                   -328.99396003063936,-399.16490341892603,-225.80887807080077,-225.80887807080077]
ax.boxplot([np.array(bo_trust_change), np.array(se_trust_change)])
ax.set_title('Overall accumulated trust change')
ax.set(ylabel='Value')
ax.set_xticklabels(['Bayesian Optimization', 'Standard design'])
fig.savefig('comparison_bo_se.tif')

plt.show()