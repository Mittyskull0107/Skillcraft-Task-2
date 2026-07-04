
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('Mall_Customers.csv')

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, wcss, 'bo-', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Optimal k')
plt.grid(True)

plt.tight_layout()
plt.show()

optimal_k = 5
print(f"\nOptimal number of clusters: {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = cluster_labels




cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(
    cluster_centers_original,
    columns=['Annual Income (k$)', 'Spending Score (1-100)']
)
cluster_centers_df['Cluster'] = range(optimal_k)
print("\nCluster Centers (Original Scale):")
print(cluster_centers_df)


cluster_stats = df.groupby('Cluster').agg({
    'Annual Income (k$)': ['mean', 'min', 'max', 'count'],
    'Spending Score (1-100)': ['mean', 'min', 'max'],
    'Age': ['mean', 'min', 'max']
}).round(2)
print("\nCluster Statistics:")
print(cluster_stats)


gender_dist = pd.crosstab(df['Cluster'], df['Gender'], normalize='index') * 100
print("\nGender Distribution (%):")
print(gender_dist.round(1))

plt.figure(figsize=(12, 8))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'],
                cluster_data['Spending Score (1-100)'],
                c=colors[i],
                label=f'Cluster {i} (n={len(cluster_data)})',
                s=100, alpha=0.7, edgecolors='black', linewidth=0.5)

plt.scatter(cluster_centers_df['Annual Income (k$)'],
            cluster_centers_df['Spending Score (1-100)'],
            c='black', s=300, marker='X', label='Centroids')

plt.xlabel('Annual Income (k$)', fontsize=12)
plt.ylabel('Spending Score (1-100)', fontsize=12)
plt.title('Customer Segmentation using K-Means Clustering', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i in range(optimal_k):
    sns.kdeplot(data=df[df['Cluster'] == i]['Annual Income (k$)'],
                label=f'Cluster {i}', ax=axes[0, 0])
axes[0, 0].set_title('Annual Income Distribution by Cluster')
axes[0, 0].legend()


for i in range(optimal_k):
    sns.kdeplot(data=df[df['Cluster'] == i]['Spending Score (1-100)'],
                label=f'Cluster {i}', ax=axes[0, 1])
axes[0, 1].set_title('Spending Score Distribution by Cluster')
axes[0, 1].legend()

for i in range(optimal_k):
    sns.kdeplot(data=df[df['Cluster'] == i]['Age'],
                label=f'Cluster {i}', ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Cluster')
axes[1, 0].legend()


df['Cluster'].value_counts().sort_index().plot(kind='bar', ax=axes[1, 1], color=colors)
axes[1, 1].set_title('Number of Customers per Cluster')
axes[1, 1].set_xlabel('Cluster')
axes[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION INSIGHTS")
print("=" * 60)

cluster_names = {
    0: "High Income, Low Spenders (Careful)",
    1: "High Income, High Spenders (Premium)",
    2: "Low Income, Low Spenders (Budget)",
    3: "Medium Income, Medium Spenders (Average)",
    4: "Low Income, High Spenders (Aspirational)"
}

for i in range(optimal_k):
    cluster_data = df[df['Cluster'] == i]
    avg_income = cluster_data['Annual Income (k$)'].mean()
    avg_spending = cluster_data['Spending Score (1-100)'].mean()
    avg_age = cluster_data['Age'].mean()
    size = len(cluster_data)

    print(f"\nCluster {i}: {cluster_names.get(i, '')}")
    print(f"  - Size: {size} customers ({size / len(df) * 100:.1f}%)")
    print(f"  - Average Income: ${avg_income:.1f}k")
    print(f"  - Average Spending Score: {avg_spending:.1f}/100")
    print(f"  - Average Age: {avg_age:.1f} years")


    if i == 0:
        print("   trategy: Offer loyalty programs and bundled deals to increase spending")
    elif i == 1:
        print("  Strategy: Focus on premium products and personalized offers")
    elif i == 2:
        print("  Strategy: Provide discounts and value-for-money products")
    elif i == 3:
        print("  Strategy: Encourage upgrades with cross-selling and product recommendations")
    elif i == 4:
        print("  Strategy: Offer installment plans and entry-level luxury products")




df.to_csv('customer_segments.csv', index=False)
print("\n Clustered data saved to 'customer_segments.csv'")


cluster_centers_df.to_csv('cluster_centers.csv', index=False)
print(" Cluster centers saved to 'cluster_centers.csv'")