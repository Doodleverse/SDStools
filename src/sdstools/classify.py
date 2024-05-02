
# classify time-series

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier


def classify_shorelines_randomforest(X, y):
    "Super basic hypothetical example of classifying shorelines X classified as y using a simple RF"
    #Random Forest classifier
    clf=RandomForestClassifier(random_state = 42, class_weight="balanced", criterion = 'gini', max_depth = 3, max_features = 'auto', n_estimators = 500)

    k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    output = cross_validate(clf, X, y, cv=k_fold, scoring = 'roc_auc', return_estimator =True)
    return output