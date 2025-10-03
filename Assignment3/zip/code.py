import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data from Cleaned CSV file after OpenRefine Cleaning Process
df = pd.read_csv('Health_Data_Cleaned.csv')

# Getting basic idea of data
print(df.head())

print(df.columns) # type: ignore

# Removing group_name column as it has only total values.
df = df[['metric_name', 'data_period', 'est', 'lci', 'uci',
       'geo_name', 'state_abbr', 'period_type', 'source_name']]

# Getting basic statistics of data
print(df.shape)

print(df.head())

print(df.info())

print(df.describe())

# Hypothesis Generation and Testing

## 1. Which cities have the highest and lowest rates of chronic health conditions like obesity, 
## diabetes, and high blood pressure?

obesity = df[df['metric_name'] == 'Obesity']
top_cities = obesity.sort_values(by='est', ascending=False).head(10)
bottom_cities = obesity.sort_values(by='est', ascending=True).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_cities['geo_name'], top_cities['est'], color='orange')
plt.xlabel('Obesity Rate Estimate')
plt.title('Top 10 Cities by Obesity Rate')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(bottom_cities['geo_name'], bottom_cities['est'], color='lightgreen')
plt.xlabel('Obesity Rate Estimate')
plt.title('Bottom 10 Cities by Obesity Rate')
plt.tight_layout()
plt.show()

## 2. Is there a correlation between frequent mental distress and chronic physical 
## health issues like diabetes or cardiovascular disease?

mental = df[df['metric_name'] == 'Frequent Mental Distress'][['geo_name', 'est']].rename(columns={'est': 'mental_distress'})
diabetes = df[df['metric_name'] == 'Diabetes'][['geo_name', 'est']].rename(columns={'est': 'diabetes'})

merged = pd.merge(mental, diabetes, on='geo_name')
correlation = merged[['mental_distress', 'diabetes']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

## 3. Do cities with higher rates of firearm suicides also have higher rates of mental distress 
## or binge drinking?
firearm = df[df['metric_name'] == 'Firearm Suicides'][['geo_name', 'est']].rename(columns={'est': 'firearm_suicide'})
binge = df[df['metric_name'] == 'Binge Drinking'][['geo_name', 'est']].rename(columns={'est': 'binge_drinking'})
mental = mental  # from previous

merged = firearm.merge(binge, on='geo_name').merge(mental, on='geo_name')
correlation_matrix = merged.corr()
# Plot
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()

## 4.	What is the geographic distribution of frequent mental distress across states?
# Filter for frequent mental distress
distress_state_avg = df[df['metric_name'] == 'Frequent Mental Distress']

# Group by state and average
state_distress = distress_state_avg.groupby('state_abbr')['est'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=state_distress, x='state_abbr', y='est', palette='viridis')
plt.title('Average Rate of Frequent Mental Distress by State')
plt.xlabel('State')
plt.ylabel('Frequent Mental Distress Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

## 5. Are certain states or regions consistently above or below national averages in 
## preventable death metrics?
premature = df[df['metric_name'] == 'Premature Deaths (All Causes)']
state_avg = premature.groupby('state_abbr')['est'].mean().reset_index()
national_avg = premature['est'].mean()

state_avg['Above_National'] = state_avg['est'] > national_avg

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=state_avg, x='state_abbr', y='est', hue='Above_National', dodge=False, palette='Set2')
plt.xlabel('State Abbreviation')
plt.ylabel('Estimate')
plt.title('State Estimates Compared to National Average')
plt.legend(title='Above National Avg')
plt.xticks(rotation=90)  # Rotate x labels for readability if needed
plt.tight_layout()
plt.show()

## 6. Do cities with poor access to dental care also report higher physical distress 
## or chronic disease rates?
dental = df[df['metric_name'] == 'Dental Care'][['geo_name', 'est']].rename(columns={'est': 'dental_care'})
physical = df[df['metric_name'] == 'Frequent Physical Distress'][['geo_name', 'est']].rename(columns={'est': 'physical_distress'})

merged = dental.merge(physical, on='geo_name')

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged, x='dental_care', y='physical_distress', color='blue', label='Data Points')
sns.regplot(data=merged, x='dental_care', y='physical_distress', scatter=False, color='red', label='Trend Line')
plt.xlabel('Dental Care (%)')
plt.ylabel('Physical Distress (%)')
plt.title('Dental Care vs Physical Distress')
plt.legend()
plt.tight_layout()
plt.show()

## 7. Do health estimates differ significantly between data sources for similar indicators, 
## and why might that be?
comparison = df.groupby(['metric_name', 'source_name'])['est'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 7))
sns.barplot(data=comparison, x='est', y='metric_name', hue='source_name', palette='muted')
plt.xlabel('Estimate')
plt.ylabel('Health Metric')
plt.title('Comparison of Metrics by Source')
plt.legend(title='Source')
plt.tight_layout()
plt.show()

## 8. Is there a significant association between lack of dental care and chronic diseases like diabetes?
# Filter relevant metrics
relevant_metrics = ['Dental Care', 'Diabetes']
filtered = df[df['metric_name'].isin(relevant_metrics)]

# Pivot to wide format for comparison
pivot_df = filtered.pivot_table(index='geo_name', columns='metric_name', values='est').dropna()

# Correlation and scatterplot
corr = pivot_df.corr().loc['Dental Care', 'Diabetes']
print(f"Correlation between lack of Dental Care and Diabetes: {corr:.2f}")

# Visualization
sns.scatterplot(data=pivot_df, x='Dental Care', y='Diabetes')
plt.title(f'Dental Care vs Diabetes Rates by City (r = {corr:.2f})')
plt.xlabel('Lack of Dental Care (%)')
plt.ylabel('Diabetes Prevalence (%)')
plt.grid(True)
plt.tight_layout()
plt.show()

## 9. What cities have the highest rates of premature deaths from all causes?
# Filter for Premature Deaths metric
premature_deaths = df[
    (df['metric_name'] == 'Premature Deaths (All Causes)')
]

# Sort and get top 15 cities
top_cities = premature_deaths.sort_values(by='est', ascending=False).head(15)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=top_cities, x='est', y='geo_name', palette='Reds_r')
plt.title('Top 15 Cities by Premature Death Rates')
plt.xlabel('Premature Death Rate (per 100,000)')
plt.ylabel('City')
plt.tight_layout()
plt.show()

## 10. Are there health disparities between cities in the same state?
# Filter for a single metric to compare across cities within states
metric = "Obesity"
metric_df = df[df["metric_name"] == metric]

# Plot
plt.figure(figsize=(14, 6))
sns.boxplot(data=metric_df, x="state_abbr", y="est")
plt.title(f"Distribution of {metric} Rates by State")
plt.ylabel("Estimated Rate (%)")
plt.xlabel("State")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
