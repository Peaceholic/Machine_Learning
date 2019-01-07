import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

X, y = make_blobs(n_samples=(4000, 500), centers=2, cluster_std=[7.0, 2], random_state=22)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

svc = SVC(gamma=.05).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train, y_train)

precision, recall, thresholds = precision_recall_curve(y_test, svc.decision_function(X_test))
precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, rf.predict_proba(X_test)[:, 1])

close_zero = np.argmin(np.abs(thresholds))
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))

plt.plot(precision[close_zero], recall[close_zero], 'o', markersize=10,
         label='svc: threshold 0', fillstyle='none', c='k', mew=2)
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], '^', c='k', markersize=10,
         label='rf: threshold 0.5', fillstyle="none", mew=2)

plt.plot(precision, recall, label="svc")
plt.plot(precision_rf, recall_rf, label="rf")

plt.xlabel("Precision")
plt.ylabel("Recall")

ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))

print("RandomForest Average Precision: {:.3f}".format(ap_rf))
print("SVC Average Precision: {:.3f}".format(ap_svc))

plt.show()
