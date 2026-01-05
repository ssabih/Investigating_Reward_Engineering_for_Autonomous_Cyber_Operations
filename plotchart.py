import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import wilcoxon

#file_path = 'mydata.csv'
file_path = 'mydata Enhanced ENv.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Apply a moving average with a window size of your choice, that is, 10 and 150
window_size = 1
df['Smoothed_Rewards'] = df['Rewards'].rolling(window=window_size).mean()
df['Smoothed_Rewards_RS'] = df['Rewards_RS'].rolling(window=window_size).mean()

fig, ax = plt.subplots()
# Plot the smoothed Rewards
ax.plot(df['Episodes'], df['Smoothed_Rewards'], color='blue', label='Smoothed Rewards without RS')

# Plot the smoothed RS Rewards
ax.plot(df['Episodes'], df['Smoothed_Rewards_RS'], color='green', label='Smoothed Rewards with RS')

# Add labels and a legend
ax.set_xlabel('Episodes')
ax.set_ylabel('Rewards')
ax.legend()

# Show the plot
#plt.show()

# Drop NaN values resulting from the rolling mean
smoothed_rewards = df['Smoothed_Rewards'].dropna()
smoothed_rewards_rs = df['Smoothed_Rewards_RS'].dropna()


# Shapiro-Wilk Test
stat, p = shapiro(smoothed_rewards_rs)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Shapiro-Wilk: The data seems to be normally distributed')
else:
    print('Shapiro-Wilk: The data does not seem to be normally distributed')


# Perform Wilcoxon Signed-Rank Test
w_statistic, p_value = wilcoxon(smoothed_rewards_rs, smoothed_rewards)
print(f"W-statistic: {w_statistic}")
print(f"P-value: {p_value}")
# Interpret the results
if p_value < 0.05:
    print("Wilcoxon: There is a statistically significant difference between the rewards with and without PBRS.")
else:
    print("Wilcoxon: There is no statistically significant difference between the rewards with and without PBRS.")