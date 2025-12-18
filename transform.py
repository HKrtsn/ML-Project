import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split

df=pd.read_csv('Project_description_and_data/claims_train.csv')
df_test=pd.read_csv('Project_description_and_data/claims_test.csv')


def transform_log_scale(df):
    df = df.copy()
    df['VehAge_1']=df['VehAge'] +1                   
    df['VehAge_log']=np.log(df['VehAge_1'])
    df['DrivAge_log']=np.log(df['DrivAge'])
    df['Density_log']=np.log(df['Density'])
    df.drop(['VehAge', 'VehAge_1', 'DrivAge', 'Density'], axis=1, inplace=True)
    return df

log_transformed_data = transform_log_scale(df)
log_transformed_test_data = transform_log_scale(df_test)

X = log_transformed_data.drop(columns='ClaimNb')
Y = log_transformed_data['ClaimNb']

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

categorical_features = ['VehBrand', 'VehGas', 'Area', 'Region']
numerical_features = [col for col in X_train.columns if col not in categorical_features]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features) #does not contain Standard Scaler becase not all models need that, pls apply manually
    ],
    remainder='drop'
)

preprocessor.set_output(transform="pandas")
preprocessor.fit(X_train)
preprocessor.get_feature_names_out()

new_Xtrain = preprocessor.transform(X_train)
new_Xval = preprocessor.transform(X_val)

Y_test = log_transformed_test_data['ClaimNb']
X_test = preprocessor.transform(log_transformed_test_data)

def pca(df):
    df=df.copy()
    df = transform_log_scale(df)
    #df=df[df['ClaimNb']>1] 

    features = ['DrivAge_log', 'VehPower', 'VehAge_log', 'Density_log', 'BonusMalus']

    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=5)   # for 2D visualization
    X_pca = pca.fit_transform(X_scaled)

    df["PC1"] = X_pca[:, 0]
    df["PC2"] = X_pca[:, 1]
    df["PC3"] = X_pca[:, 2]
    df["PC4"] = X_pca[:, 3]
    df["PC5"] = X_pca[:, 4]

    explained = pca.explained_variance_ratio_ 
    print(f"PC1: {explained[0]:.2%}, PC2: {explained[1]:.2%}, PC3: {explained[2]:.2%}, PC4: {explained[3]:.2%}, PC5: {explained[4]:.2%},  Total: {explained[:5].sum():.2%}")

    loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2", "PC3", "PC4", "PC5"],
    index=features
    )
    print(loadings)
    return df

def visualisation(df):
    df_pca = pca(df)
    sns.scatterplot(
        x="PC1",
        y="PC2",
        hue="ClaimNb",
        data=df_pca,
        palette="viridis",
        alpha=0.7
    )
    plt.title("PCA Projection Colored by Claim Number")
    plt.show()

visualisation(df)


def cluster_dbscan(df):
    df=df.copy()
    df=pca(df)
    X = df[['PC1', 'PC2']].values
    db = DBSCAN(eps=0.19, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    unique_labels = set(labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))

    for k, col in zip(unique_labels, colors(np.arange(len(unique_labels)))):
        if k == -1:
            col = (0, 0, 0, 1)  # black for noise

        class_member_mask = (labels == k)
        xy = X[class_member_mask]
    
        plt.plot(xy[:, 0], xy[:, 1], 'o',
            markerfacecolor=col,
            markeredgecolor='k',
            markersize=3)

show_clusters = cluster_dbscan(df)
show_clusters
