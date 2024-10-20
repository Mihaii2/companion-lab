'''
## Exercise 8
(Independent events, implementation)

Consider the event space corresponding to two tosses of a fair coin, and the events A "heads on toss 1", B "heads on toss 2" and C "the two tosses are equal". Using the `tools.stats.probability` function, find if:

1. events A and B are independent;
1. events A and C are independent.
'''

from tools.stats import probability
from typing import Set, Any
# Two events A and B if their joint probability equals the product of their probabilities

def are_independent(A: Set[Any], B: Set[Any], omega: Set[Any]) -> bool:
    """Check if two events A and B are independent"""
    prob_A = probability(A, omega)
    prob_B = probability(B, omega)
    prob_intersection = probability(A.intersection(B), omega)
    return abs(prob_intersection - (prob_A * prob_B)) < 1e-10  # Using small epsilon for float comparison

# Define the sample space and events
omega = {"TT", "HT", "TH", "HH"}
A = {"HT", "HH"}  # heads on toss 1
B = {"TH", "HH"}  # heads on toss 2
C = {"TT", "HH"}  # the two tosses are equal

# Check independence of A and B
print("Are A and B independent?", are_independent(A, B, omega))

# Check independence of A and C
print("Are A and C independent?", are_independent(A, C, omega))

# Optional: Print probabilities for verification
print(f"P(A) = {probability(A, omega):.2f}")
print(f"P(B) = {probability(B, omega):.2f}")
print(f"P(C) = {probability(C, omega):.2f}")
print(f"P(A ∩ B) = {probability(A.intersection(B), omega):.2f}")
print(f"P(A ∩ C) = {probability(A.intersection(C), omega):.2f}")