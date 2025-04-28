import numpy as np
import pandas as pd
np.random.seed(42)
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import concurrent.futures
import time

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*, value=None): #value will only be passed to leaf nodes
        # Parameters after ,*, are keyword-arguments only, has to be addressed as value=42, and not (...,42)
        self.feature = feature
        self.threshold = threshold # The split value for the feature
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split # Minimum number of samples required to split an internal node, prevents overfitting
        self.max_depth = max_depth
        self.n_features = n_features 
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X,y)

    # Recursive function to grow the tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Check if we have reached the stopping criteria
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_labels == 1):


            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        subset_features = np.random.choice(n_features, self.n_features, replace=False) # Only use a subset of unique features

        # Find the best split
        best_thresh, best_feature = self._best_split(X, y, subset_features)
        
        # Create the left and right subtrees

        left_idxs, right_idxs = self._split(X[:,best_feature],best_thresh)

        if len(left_idxs)  == 0 or len(right_idxs) == 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_thresh, left, right)


    # Finding the best splits and thresholds 
    def _best_split(self, X, y, subset_features):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feature in subset_features:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Calculate the information gain
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature
                    split_thresh = threshold
        return split_thresh, split_idx


    def _information_gain(self,y,X_column, threshold):
        # IG = E(parent) - [weighted average] * E(children)

        #parent entropy

        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if(len(left_idxs) == 0 or len(right_idxs) == 0):
            return 0

        #calculate the avg. weighted entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l/n)*e_l + (n_r/n)*e_r

        #calculate the IG

        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column<=split_thresh).flatten() # gives the index of the arguments that fulfills the args. flatten() flattens the list of lists into a list
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs
    def _entropy(self,y):
        # E = - Sum(p(X)*log_2(p(X)))
        hist = np.bincount(y) #creates a historgram [instances_of_0, instances_of_1, ...]
        ps = hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0]) 

    def _most_common_label(self, y):
        # Return the most common label in y
        counter = Counter(y) # REPLACE
        most_common = counter.most_common(1)[0][0]
        return most_common

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)

    
    def predict(self, X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    
    # from sklearn.base import BaseEstimator, ClassifierMixin
# BaseEstimator, ClassifierMixin
class RandomForest():
    def __init__(self, n_trees=200, max_depth=10, min_samples_split=3, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []


    def fit(self, X,y):

        n_tot_features = X.shape[1]
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_tot_features))
        else:
            self.n_features = min(self.n_features, n_tot_features) 

        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                         n_features=self.n_features)
            X_sample, y_sample =self._bootstrap_samples(X,y)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)
        return self

    def _bootstrap_samples(self,X,y):
        n_samples, _ = X.shape
        idxs = np.random.choice(n_samples, n_samples, replace=True) # sampling with replacement
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        # Return the most common label in y
        counter = Counter(y) # REPLACE
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees]) # [[pred_1st_sample_by_1st_tree, pred_2st_sample_by_1st_tree],
                                                                        #   [pred_1st_sample_by_2nd_tree],[pred_2nd_sample_by_2nd_tree],
                                                                        #   [...], ...]
        #[[all predictions of first sample], [all predictions of second sample], [...], ...]

        tree_predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(pred) for pred in tree_predictions])
        return predictions
    
    def feature_importance(self, feature_names=None):
        """
        Calculate feature importance based on how often features are used for splitting
        
        Parameters:
        -----------
        feature_names : list, optional
            List of feature names, if None, will use indices
            
        Returns:
        --------
        DataFrame with feature importances
        """
        feature_counts = {}
        
        def count_feature_usage(node, counts):
            if node is None or node.is_leaf_node():
                return
            counts[node.feature] = counts.get(node.feature, 0) + 1
            count_feature_usage(node.left, counts)
            count_feature_usage(node.right, counts)
        
        # Count feature usage across all trees
        for tree in self.trees:
            counts = {}
            count_feature_usage(tree.root, counts)
            
            # Normalize counts for each tree
            total = sum(counts.values()) or 1
            for feature, count in counts.items():
                feature_counts[feature] = feature_counts.get(feature, 0) + count / total
        
        # Average across all trees
        n_trees = len(self.trees)
        importances = {feature: count / n_trees for feature, count in feature_counts.items()}
        
        # Create DataFrame
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'Feature': [feature_names[feature] for feature in importances.keys()],
                'Importance': list(importances.values())
            })
        else:
            importance_df = pd.DataFrame({
                'Feature': list(importances.keys()),
                'Importance': list(importances.values())
            })
        
        return importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)



# Add confusion matrix function
def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """Plot confusion matrix for the classifier results"""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Print per-class accuracy
    class_acc = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for i, acc in enumerate(class_acc):
        class_name = class_names[i] if class_names is not None else i
        print(f"Class {class_name}: {acc:.4f}")
    
    return cm

# Feature selection using correlation analysis
def select_features(X, y, features, threshold=0.7, top_n=None):
    """Select features based on correlation with target and between features"""
    # Create DataFrame with features and target
    data = pd.DataFrame(X, columns=features)
    data['target'] = y
    
    # Calculate correlation with target
    target_corr = data.corr()['target'].drop('target').abs()
    sorted_corr = target_corr.sort_values(ascending=False)
    
    if top_n:
        # Return top N features by correlation
        return sorted_corr.index[:top_n].tolist()
    else:
        # Select features with correlation above threshold
        selected = sorted_corr[sorted_corr > threshold].index.tolist()
        
        # Remove highly correlated features
        feature_corr = data[selected].corr().abs()
        features_to_drop = set()
        
        for i in range(len(selected)):
            if selected[i] in features_to_drop:
                continue
                
            for j in range(i+1, len(selected)):
                if selected[j] in features_to_drop:
                    continue
                    
                if feature_corr.iloc[i, j] > threshold:
                    # Keep the one with higher correlation to target
                    if target_corr[selected[i]] < target_corr[selected[j]]:
                        features_to_drop.add(selected[i])
                    else:
                        features_to_drop.add(selected[j])
        
        return [f for f in selected if f not in features_to_drop]

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

# Parallelized cross-validation function
def cross_validate_rf(X, y, n_folds=5, **params):
    """Perform cross-validation for RandomForest"""
    fold_size = len(X) // n_folds
    
    def evaluate_fold(i):
        # Create validation fold
        val_indices = list(range(i * fold_size, (i + 1) * fold_size))
        train_indices = [j for j in range(len(X)) if j not in val_indices]
        
        X_train_fold, y_train_fold = X[train_indices], y[train_indices]
        X_val_fold, y_val_fold = X[val_indices], y[val_indices]
        
        # Train model
        model = RandomForest(**params)
        model.fit(X_train_fold, y_train_fold)
        
        # Evaluate
        predictions = model.predict(X_val_fold)
        acc = accuracy(y_val_fold, predictions)
        return acc
    
    # Run folds in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        accuracies = list(executor.map(evaluate_fold, range(n_folds)))
    
    return np.mean(accuracies)

class ImprovedRandomForest(RandomForest):
    def __init__(self, n_trees=200, max_depth=10, min_samples_split=3, n_features=None):
        super().__init__(n_trees, max_depth, min_samples_split, n_features)
        self.oob_score_ = None
    
    def fit(self, X, y):
        """Fit with out-of-bag error estimation"""
        n_samples = X.shape[0]
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.trees = []
        
        # For OOB score calculation
        oob_predictions = np.zeros((n_samples, len(np.unique(y))))
        n_predictions = np.zeros(n_samples)
        
        for i in range(self.n_trees):
            # Bootstrap sample
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            oob_indices = np.array([i for i in range(n_samples) if i not in sample_indices])
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    n_features=self.n_features)
            tree.fit(X[sample_indices], y[sample_indices])
            self.trees.append(tree)
            
            # OOB prediction for this tree
            if len(oob_indices) > 0:
                oob_pred = tree.predict(X[oob_indices])
                
                # Count predictions
                for j, idx in enumerate(oob_indices):
                    oob_predictions[idx, oob_pred[j]] += 1
                    n_predictions[idx] += 1
        
        # Calculate OOB score
        oob_score = 0
        n_valid = 0
        
        for i in range(n_samples):
            if n_predictions[i] > 0:
                pred_class = np.argmax(oob_predictions[i])
                if pred_class == y[i]:
                    oob_score += 1
                n_valid += 1
        
        self.oob_score_ = oob_score / n_valid if n_valid > 0 else 0
        return self

# Load and prepare data
data = pd.read_csv('data/GenreClassData_30s.txt', sep='\t')
data["TrackID"] = range(len(data))

# Split the data into training and testing sets
train = data[data['Type'] == 'Train']
test = data[data['Type'] == 'Test']

# All features list
all_features = [
    'zero_cross_rate_mean','zero_cross_rate_std','rmse_mean','rmse_var',
    'spectral_centroid_mean','spectral_centroid_var','spectral_bandwidth_mean','spectral_bandwidth_var',
    'spectral_rolloff_mean','spectral_rolloff_var','spectral_contrast_mean','spectral_contrast_var',
    'spectral_flatness_mean','spectral_flatness_var',
    'chroma_stft_1_mean','chroma_stft_2_mean','chroma_stft_3_mean','chroma_stft_4_mean',
    'chroma_stft_5_mean','chroma_stft_6_mean','chroma_stft_7_mean','chroma_stft_8_mean',
    'chroma_stft_9_mean','chroma_stft_10_mean','chroma_stft_11_mean','chroma_stft_12_mean',
    'chroma_stft_1_std','chroma_stft_2_std','chroma_stft_3_std','chroma_stft_4_std',
    'chroma_stft_5_std','chroma_stft_6_std','chroma_stft_7_std','chroma_stft_8_std',
    'chroma_stft_9_std','chroma_stft_10_std','chroma_stft_11_std','chroma_stft_12_std',
    'tempo',
    'mfcc_1_mean','mfcc_2_mean','mfcc_3_mean','mfcc_4_mean','mfcc_5_mean','mfcc_6_mean',
    'mfcc_7_mean','mfcc_8_mean','mfcc_9_mean','mfcc_10_mean','mfcc_11_mean','mfcc_12_mean',
    'mfcc_1_std','mfcc_2_std','mfcc_3_std','mfcc_4_std','mfcc_5_std','mfcc_6_std',
    'mfcc_7_std','mfcc_8_std','mfcc_9_std','mfcc_10_std','mfcc_11_std','mfcc_12_std'
]

features = all_features
targets = ['GenreID'] 

# Prepare data for training
train = train.copy()
test = test.copy()

X_train, y_train = train[features], train[targets]
X_test, y_test = test[features], test[targets]

X_train_np = X_train.to_numpy()      
y_train_np = y_train.to_numpy().ravel()  # flatten from (n,1) to (n,)

X_test_np = X_test.to_numpy()
y_test_np = y_test.to_numpy().ravel()

# Get genre names for better visualization
genre_names = data['Genre'].unique()
genre_id_to_name = dict(zip(data['GenreID'].unique(), genre_names))

# Train initial model
print("\nTraining initial model...")
clf = RandomForest()
clf.fit(X_train_np, y_train_np)

predictions = clf.predict(X_test_np)
acc = accuracy(y_test_np, predictions)
print(f"Initial accuracy: {acc:.4f}")

# Feature importance
importance = clf.feature_importance(features)
print("\nTop 10 most important features:")
print(importance.head(10))

# Feature selection
print("\nPerforming feature selection...")
selected_features = select_features(X_train_np, y_train_np, features, threshold=0.3, top_n=20)
print(f"Selected {len(selected_features)} features out of {len(features)}")
print("Selected features:", selected_features)

# Extract selected features
X_train_selected = X_train[selected_features].to_numpy()
X_test_selected = X_test[selected_features].to_numpy()

# Train with selected features
print("\nTraining with selected features...")
clf_selected = RandomForest(n_trees=100, max_depth=10, min_samples_split=3)
clf_selected.fit(X_train_selected, y_train_np)
selected_predictions = clf_selected.predict(X_test_selected)
selected_acc = accuracy(y_test_np, selected_predictions)
print(f"Accuracy with selected features: {selected_acc:.4f}")

# Evaluate initial model with confusion matrix
print("\nEvaluating initial model...")
cm = plot_confusion_matrix(y_test_np, predictions, 
                        class_names=[genre_id_to_name[i] for i in sorted(genre_id_to_name.keys())])

# Evaluate selected features model
print("\nEvaluating model with selected features...")
cm_selected = plot_confusion_matrix(y_test_np, selected_predictions, 
                                class_names=[genre_id_to_name[i] for i in sorted(genre_id_to_name.keys())])

# Function to evaluate a single parameter combination
def evaluate_params(params_tuple):
    n_trees, max_depth, min_samples_split, n_features = params_tuple
    params = {
        'n_trees': n_trees,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'n_features': n_features
    }
    
    start_time = time.time()
    cv_acc = cross_validate_rf(X_train_selected, y_train_np, n_folds=3, **params)
    elapsed = time.time() - start_time
    
    return params, cv_acc, elapsed

# Hyperparameter tuning with parallel processing
print("\nPerforming hyperparameter tuning with selected features...")
print("Using parallel processing with 10 workers")
best_acc = 0
best_params = {}

# Fix the parameter grid definition
param_grid = {
    'n_trees': [50, 100],
    'max_depth': [5, 10],
    'min_samples_split': [2, 5],
    'n_features': [int(np.sqrt(len(selected_features))), len(selected_features) // 2]
}

# Create parameter combinations
param_combinations = []
for n_trees in param_grid['n_trees']:
    for max_depth in param_grid['max_depth']:
        for min_samples_split in param_grid['min_samples_split']:
            for n_features in param_grid['n_features']:
                param_combinations.append((n_trees, max_depth, min_samples_split, n_features))

print(f"Testing {len(param_combinations)} parameter combinations with 3-fold CV")

# Run parameter evaluation in parallel
results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    future_to_params = {executor.submit(evaluate_params, params): params for params in param_combinations}
    
    for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
        params_tuple = future_to_params[future]
        try:
            params, cv_acc, elapsed = future.result()
            results.append((params, cv_acc))
            print(f"Combination {i+1}/{len(param_combinations)}: {params}, CV Accuracy: {cv_acc:.4f}, Time: {elapsed:.2f}s")
            
            if cv_acc > best_acc:
                best_acc = cv_acc
                best_params = params.copy()
        except Exception as e:
            print(f"Error evaluating {params_tuple}: {e}")

print(f"\nBest hyperparameters with selected features: {best_params}")
print(f"Best CV accuracy with selected features: {best_acc:.4f}")

# Train improved model with best hyperparameters
print("\nTraining improved model with best hyperparameters...")
improved_clf = ImprovedRandomForest(**best_params)
improved_clf.fit(X_train_selected, y_train_np)
print(f"Out-of-bag accuracy estimate: {improved_clf.oob_score_:.4f}")

# Final evaluation
improved_predictions = improved_clf.predict(X_test_selected)
improved_acc = accuracy(y_test_np, improved_predictions)
print(f"Final test accuracy with improved model: {improved_acc:.4f}")

# Evaluate improved model
print("\nEvaluating improved model...")
cm_improved = plot_confusion_matrix(y_test_np, improved_predictions, 
                                class_names=[genre_id_to_name[i] for i in sorted(genre_id_to_name.keys())])

# Feature importance of the final model
importance = improved_clf.feature_importance(selected_features)
print("\nFeature importance in the final model:")
print(importance)