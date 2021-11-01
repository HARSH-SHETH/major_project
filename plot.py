import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")

fig = plt.figure(figsize=(10, 10))
fig.suptitle("Hop Count Distributions: Malicious vs Benign Webpages")
dataset = pd.read_csv('dataset.csv')[['hopCount','label']]
dataset['label'] = dataset['label'].apply(lambda x:'Malicious' if x == 'bad' else 'Benign')
good = dataset[dataset.label=='Benign'].value_counts().reset_index(name='count')
bad = dataset[dataset.label=='Malicious'].value_counts().reset_index(name='count')
combined = good.append(bad).reset_index()
# KDE Plot
ax = plt.subplot(221)
sns.kdeplot(x='hopCount', hue='label', data=combined)
ax.set_xlabel("KDE Plot")
ax = plt.subplot(222)
sns.violinplot(x='label', y='hopCount', data=dataset)
ax.set_xlabel("Violin Plot")
ax = plt.subplot(223)
sns.barplot(x='label', y='hopCount', data=combined)
ax.set_xlabel("Box Plot")
ax = plt.subplot(224)
sns.histplot(dataset, x='hopCount', y='hopCount', hue='label', ax=ax)
ax.set_xlabel("Bivariate Plot")
# plt.show()
fig.savefig("HopCountDistribution.svg")
