import math
import matplotlib.pyplot as plt

# 1.

distribution = [1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36]
sums = list(range(2, 13))

plt.bar(sums, distribution, width = 0.5, edgecolor='black')
plt.show()

# 2.
for S in [2, 11, 5, 7]:
    print(f'Information content for S value {S}: {math.log2(1/distribution[S - 2])} bits')

# 3.
entropy = sum([prob * math.log2(1/prob) for prob in distribution])
print(f'Entropy of S: {entropy}')

distribution_first_4 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]

new_entropy = sum([prob * math.log2(1/prob) for prob in distribution_first_4])

print(f'New entropy: {new_entropy}')
print(f'Information gain: {entropy - new_entropy} bits')