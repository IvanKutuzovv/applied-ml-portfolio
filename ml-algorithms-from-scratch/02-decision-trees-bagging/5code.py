import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]

    distinct_values_mask = feature_vector[1:] != feature_vector[:-1]
    
    thresholds = (feature_vector[1:][distinct_values_mask] + feature_vector[:-1][distinct_values_mask]) / 2.0
    
    if len(thresholds) == 0:
        return np.array([]), np.array([]), None, None

    classes = np.unique(target_vector)
    target_ohe = (target_vector[:, None] == classes[None, :]).astype(int)
    cumsum_counts = np.cumsum(target_ohe, axis=0)
    
    left_counts = cumsum_counts[:-1][distinct_values_mask]
    
    total_counts = cumsum_counts[-1]
    
    right_counts = total_counts - left_counts
    
    left_sizes = left_counts.sum(axis=1) 
    right_sizes = right_counts.sum(axis=1) 
    
    n_samples = target_vector.shape[0]

    gini_left = 1 - np.sum(left_counts ** 2, axis=1) / (left_sizes ** 2)
    
    gini_right = 1 - np.sum(right_counts ** 2, axis=1) / (right_sizes ** 2)
    
    ginis = - (left_sizes / n_samples * gini_left + right_sizes / n_samples * gini_right)
    
    best_idx = np.argmax(ginis)
    threshold_best = thresholds[best_idx]
    gini_best = ginis[best_idx]
    
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return
            
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                
                counts = Counter(sub_X[:, feature])
                most_common_class = Counter(sub_y).most_common(1)[0][0]
                
                clicks = Counter(sub_X[sub_y == most_common_class, feature]) # Спасибо Альбина 
                
                ratio = {}
                for key, current_count in counts.items():
                    current_click = clicks.get(key, 0)
                    ratio[key] = current_click / current_count
                
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) < 2:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            
            if gini is not None and (gini_best is None or gini > gini_best):
                
                curr_split = feature_vector < threshold
                if self._min_samples_leaf is not None:
                    if np.sum(curr_split) < self._min_samples_leaf or \
                       np.sum(~curr_split) < self._min_samples_leaf:
                        continue

                feature_best = feature
                gini_best = gini
                split = curr_split
                
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = set(threshold_best) 
        else:
            raise ValueError
            
        node["left_child"], node["right_child"] = {}, {}
        
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
        
        feature_idx = node["feature_split"]
        feature_val = x[feature_idx]
        
        go_left = False
        if "threshold" in node: 
            if feature_val < node["threshold"]:
                go_left = True
        elif "categories_split" in node: 
            if feature_val in node["categories_split"]:
                go_left = True
        
        if go_left:
            return self._predict_node(x, node["left_child"])
        else:
            return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)