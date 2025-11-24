import pickle
import numpy as np
import os
from sklearn.decomposition import PCA

# Preprocessing 
def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        X = batch[b'data']
        y = np.array(batch[b'labels'])
        return X, y
    
def load_cifar10(data_dir):
    X_all = []
    y_all = []
    
    # 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        X, y = load_batch(batch_path)
        X_all.append(X)
        y_all.append(y)
        
    X_train = np.concatenate(X_all)
    y_train = np.concatenate(y_all)
    
    # test batch
    X_test, y_test = load_batch(os.path.join(data_dir, "test_batch"))
    
    return X_train/255, y_train, X_test/255, y_test 

DATA_DIR = "cifar-10-batches-py"
TARGET_DIM = 200

# Use PCA to reduce the dimension to 200 features
def pca_reduction(X_train, X_test, target_dim=TARGET_DIM):
    
    pca = PCA(n_components=target_dim)
    pca.fit(X_train)
    
    X_train_reduced = pca.transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    print(f"PCA -- Original X_train shape: {X_train.shape}")
    print(f"PCA -- Reduced X_train shape: {X_train_reduced.shape}")
    print(f"PCA -- Reduced X_test shape: {X_test_reduced.shape}")
    
    return X_train_reduced, X_test_reduced

X_train, y_train, X_test, y_test = load_cifar10(DATA_DIR)
X_train_reduced_pca, X_test_reduced_pca = pca_reduction(X_train, X_test)