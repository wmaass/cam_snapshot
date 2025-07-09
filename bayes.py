import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Simulate Noisy Sensory Perception (Visual & Auditory Inputs)
np.random.seed(42)

num_samples = 1000  # Number of perception events
actual_object = np.random.choice(["Tree", "Person"], size=num_samples, p=[0.7, 0.3])  # 70% chance it's a tree

# Simulating noisy sensory data (vision and sound)
vision_noise = np.random.normal(loc=0, scale=0.3, size=num_samples)  # Visual uncertainty
sound_noise = np.random.normal(loc=0, scale=0.2, size=num_samples)  # Auditory uncertainty

# Sensory signals (higher means more likely to be a person)
vision_signal = (actual_object == "Person") * 0.8 + (actual_object == "Tree") * 0.2 + vision_noise
sound_signal = (actual_object == "Person") * 0.9 + (actual_object == "Tree") * 0.1 + sound_noise

# Bayesian Hypothesis: P(Person | Sensory Data)
def bayesian_perception(prior, likelihood_vision, likelihood_sound):
    """Computes posterior probability of an object being a person given sensory data."""
    prior_tree = 1 - prior  # P(Tree)
    
    # Total probability of the sensory input
    total_prob = (likelihood_vision * likelihood_sound * prior) + ((1 - likelihood_vision) * (1 - likelihood_sound) * prior_tree)
    
    # Bayesian update for P(Person | Sensory Data)
    posterior = (likelihood_vision * likelihood_sound * prior) / total_prob
    return posterior

# Compute probabilities
prior_person = 0.3  # Initial belief that it's a person
likelihood_vision = (vision_signal > 0.5).astype(int).mean()
likelihood_sound = (sound_signal > 0.5).astype(int).mean()
posterior_person = bayesian_perception(prior_person, likelihood_vision, likelihood_sound)

# Create DataFrame for visualization
df = pd.DataFrame({
    "Actual_Object": actual_object,
    "Vision_Signal": vision_signal,
    "Sound_Signal": sound_signal
})

# Step 2: Visualization (Perception as Bayesian Inference)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))  

# **1. Distribution of Noisy Sensory Inputs (Vision)**
sns.histplot(df["Vision_Signal"], bins=30, kde=True, color="gray", ax=axes[0, 0])
axes[0, 0].set_xlabel("Noisy Vision Perception")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].set_title("Neural Noise in Vision Perception")

# **2. Distribution of Noisy Sensory Inputs (Sound)**
sns.histplot(df["Sound_Signal"], bins=30, kde=True, color="blue", ax=axes[0, 1])
axes[0, 1].set_xlabel("Noisy Sound Perception")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].set_title("Neural Noise in Auditory Perception")

# **3. Bayesian Probability Update Visualization**
sns.barplot(x=["Prior P(Person)", "Likelihood (Vision)", "Likelihood (Sound)", "Posterior P(Person)"], 
            y=[prior_person, likelihood_vision, likelihood_sound, posterior_person], 
            palette=["gray", "red", "blue", "green"], ax=axes[1, 0])
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_ylabel("Probability")
axes[1, 0].set_title("Bayesian Perception: Probability of Person Given Sensory Input")

# **4. Scatter Plot of Vision vs. Sound**
sns.scatterplot(x=df["Vision_Signal"], y=df["Sound_Signal"], hue=df["Actual_Object"], palette={"Tree": "green", "Person": "red"}, ax=axes[1, 1])
axes[1, 1].set_xlabel("Vision Signal")
axes[1, 1].set_ylabel("Sound Signal")
axes[1, 1].set_title("Sensory Integration: Vision vs. Sound")

plt.tight_layout()
plt.show()

# Step 3: Print Bayesian Probability & Explanation
print(f"🔍 Prior belief that it's a person: {prior_person:.2%}")
print(f"👀 Likelihood based on vision: {likelihood_vision:.2%}")
print(f"👂 Likelihood based on sound: {likelihood_sound:.2%}")
print(f"🧠 Updated belief (Posterior P(Person | Vision, Sound)): {posterior_person:.2%}")

