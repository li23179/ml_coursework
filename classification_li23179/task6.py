import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

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
X_train_reduced, X_test_reduced = pca_reduction(X_train, X_test)

# Use cross-validation on the training split to select appropriate hyperparameters 
# (e.g. depth, split criteria). 
parameters = {"max_depth": [5, 10, 15, 20],
              "criterion": ["gini", "entropy"]}

#  Initialise a single decision tree classifier on the CIFAR–10 training set. 
model = DecisionTreeClassifier(random_state=42)

# Initialise GridSearch with 5-fold Cross Validation
clf = GridSearchCV(model, parameters, cv=5, scoring="accuracy", verbose=2)
clf.fit(X_train_reduced, y_train)

best_params = clf.best_params_
best_score = clf.best_score_

print(f"Best 5-fold Cross-Validation Score: {best_score: .3f}")
print(f"Best Hyperparameter: {best_params}")

# Train an ensemble classifier (AdaBoost)
num_models = 200
sample_size = 2000

sample_weights = np.ones(X_train_reduced.shape[0]) / X_train_reduced.shape[0]

alphas = []
models = []

for _ in range(num_models):
    # Sample with replacement from the training set. 
    # Each sample should contain sample_size data points chosen at random.
    sample_index = np.random.choice(X_train_reduced.shape[0], size=sample_size)
    X_train_sample, y_train_sample = X_train_reduced[sample_index], y_train[sample_index]

    # Use the best hyperparameters for the model
    model = DecisionTreeClassifier(
        max_depth=best_params["max_depth"],
        criterion=best_params["criterion"],
        random_state=1
    )
    
    # fit the model with sample weight
    model.fit(X_train_sample, y_train_sample, sample_weight=sample_weights[sample_index])
    
    error = 1-model.score(X_train_sample, y_train_sample, sample_weight=sample_weights[sample_index])
    
    # calculate alpha and append it to alphas
    alpha = np.log((1-error)/error)
    alphas.append(alpha)
    
    # update sample weights for incorrect prediction
    incorrect = model.predict(X_train_sample) != y_train_sample
    sample_weights[sample_index] *= np.exp(alpha*incorrect)
    
    sample_weights /= np.sum(sample_weights)
    
    models.append(model)
    
def boosting_predict(test_data, models, alphas):
    votes = np.zeros((test_data.shape[0], len(models)))
    combined_predictions = np.zeros(test_data.shape[0], dtype=int)
    
    for index, m in enumerate(models):
        # obtain predictions from each model and store it in votes
        votes[:, index] = m.predict(test_data)
        
    for test_points in range(len(votes)):
        # determine the class with the most votes for each test point and store it in combined_predictions
        combined_predictions[test_points] = np.bincount(np.int64(votes[test_points]), alphas).argmax()
        
    return combined_predictions

prediction = boosting_predict(X_test_reduced, models, alphas)
accuracy = np.count_nonzero(prediction==np.int64(y_test)) / y_test.shape[0]
print(f"Test Accuracy: {accuracy: .3f}")

incorrect_indices = np.where(prediction != y_test)[0]
print(f"Number of misclassified images: {len(incorrect_indices)} out of {X_test_reduced.shape[0]}")

# class label in CIFAR
classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# select the first 5 misclassified test image
five_incorrect = incorrect_indices[:5]

fig, ax = plt.subplots(1, 5, figsize=(10, 3))

for i, index in enumerate(five_incorrect):
    
    # 1) reshape to (channels, height, width)
    img = X_test[index].reshape(3, 32, 32)

    # 2) transpose to (height, width, channels) for matplotlib
    img = img.transpose(1, 2, 0)
    
    true_class = classes[y_test[index]]
    pred_class = classes[prediction[index]]
    
    ax[i].imshow(img, interpolation='nearest')
    ax[i].set_title(f"True: {true_class}\nPredict: {pred_class}")
    ax[i].axis("off")

plt.tight_layout()
plt.savefig("classification_images/adaboost_misclassifions.png", dpi=300)
plt.show()