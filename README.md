import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Declare & 2. Initialize
# Shape: (3 Seasons, 4 Weeks, 7 Days)
# Seasons: Summer, Monsoon, Winter
seasons = ['Summer', 'Monsoon', 'Winter']
aqi_data = np.random.randint(50, 150, size=(3, 4, 7)) # Summer
aqi_data[2] = np.random.randint(250, 450, size=(4, 7)) # Winter spike

# 3. Organize (Pivoting: Swap Weeks and Seasons)
pivoted_aqi = np.transpose(aqi_data, (1, 0, 2)) 

# 4. Clean up (Clip outliers above 500)
cleaned_aqi = np.clip(aqi_data, 0, 500)

# 5. Computations
avg_seasonal_aqi = np.mean(cleaned_aqi, axis=(1, 2))
# Correlation: AQI vs simulated Humidity
humidity = np.random.uniform(30, 90, size=cleaned_aqi.shape)
corr = np.corrcoef(cleaned_aqi.flatten(), humidity.flatten())[0, 1]

# 6. Output
print(f"Seasonal Averages: {dict(zip(seasons, avg_seasonal_aqi))}")
print(f"Correlation (AQI vs Humidity): {corr:.2f}")

plt.figure(figsize=(10, 4))
sns.heatmap(cleaned_aqi[2], annot=True, cmap="YlOrRd", xticklabels=range(1,8), yticklabels=range(1,5))
plt.title("Winter AQI Heatmap (Weeks vs Days)")
plt.show()
