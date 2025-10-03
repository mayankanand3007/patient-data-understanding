import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Health_Data_Cleaned.csv')

print(df.head())

print(df.columns)

df = df[['metric_name', 'data_period', 'est', 'lci', 'uci',
       'geo_name', 'state_abbr', 'period_type', 'source_name']]

print(df.shape)

print(df.head())

print(df.info())

print(df.describe())

# 1. Top and Bottom Cities by Obesity Rate
obesity = df[df['metric_name'] == 'Obesity']
top_cities = obesity.sort_values(by='est', ascending=False).head(10)
bottom_cities = obesity.sort_values(by='est', ascending=True).head(10)


sns.barplot(data=top_cities.head(10), x='est', y='geo_name', hue='geo_name', palette='flare', legend=False)

sns.barplot(data=bottom_cities.head(10), x='est', y='geo_name', hue='geo_name', palette="YlOrBr", legend=False)

# 2. Correlation: Mental Distress vs Diabetes
mental = df[df['metric_name'] == 'Frequent Mental Distress'][['geo_name', 'est']].rename(columns={'est': 'mental_distress'})
diabetes = df[df['metric_name'] == 'Diabetes'][['geo_name', 'est']].rename(columns={'est': 'diabetes'})

merged = pd.merge(mental, diabetes, on='geo_name')
correlation = merged[['mental_distress', 'diabetes']].corr()

sns.heatmap(correlation)

# 3. Firearm Suicides vs Binge Drinking or Mental Distress
firearm = df[df['metric_name'] == 'Firearm Suicides'][['geo_name', 'est']].rename(columns={'est': 'firearm_suicide'})
binge = df[df['metric_name'] == 'Binge Drinking'][['geo_name', 'est']].rename(columns={'est': 'binge_drinking'})
mental = mental  # from previous

merged = firearm.merge(binge, on='geo_name').merge(mental, on='geo_name')
correlation_matrix = merged.corr()

sns.heatmap(correlation_matrix)

# Filter for frequent mental distress
distress_state_avg = df[df['metric_name'] == 'Frequent Mental Distress']

# Group by state and average
state_distress = distress_state_avg.groupby('state_abbr')['est'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=state_distress, x='state_abbr', y='est', hue='est', legend=False, palette='viridis')
plt.title('Average Rate of Frequent Mental Distress by State')
plt.xlabel('State')
plt.ylabel('Frequent Mental Distress Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


premature = df[df['metric_name'] == 'Premature Deaths (All Causes)']
state_avg = premature.groupby('state_abbr')['est'].mean().reset_index()
national_avg = premature['est'].mean()

state_avg['Above_National'] = state_avg['est'] > national_avg

sns.barplot(data=state_avg, x='state_abbr', y='est', hue='Above_National', dodge=False, palette='Set2')

dental = df[df['metric_name'] == 'Dental Care'][['geo_name', 'est']].rename(columns={'est': 'dental_care'})
physical = df[df['metric_name'] == 'Frequent Physical Distress'][['geo_name', 'est']].rename(columns={'est': 'physical_distress'})

merged = dental.merge(physical, on='geo_name')
sns.scatterplot(data=merged, x='dental_care', y='physical_distress')
sns.regplot(data=merged, x='dental_care', y='physical_distress', scatter=False, color='red')

comparison = df.groupby(['metric_name', 'source_name'])['est'].mean().reset_index()
sns.barplot(data=comparison, x='est', y='metric_name', hue='source_name')

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


# Filter for Premature Deaths metric
premature_deaths = df[
    (df['metric_name'] == 'Premature Deaths (All Causes)')
]

# Sort and get top 15 cities
top_cities = premature_deaths.sort_values(by='est', ascending=False).head(15)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=top_cities, x='est', y='geo_name', hue='geo_name', legend=False, palette='Reds_r')
plt.title('Top 15 Cities by Premature Death Rates')
plt.xlabel('Premature Death Rate (per 100,000)')
plt.ylabel('City')
plt.tight_layout()
plt.show()


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