import pandas as pd
import numpy as np
df = pd.read_csv('E:\\heart-disease.csv')
x = df.drop('target', axis=1)
y = df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state=0)
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

classifier = SVC(kernel='linear', random_state=0)  
classifier.fit(x_train, y_train)
y_pred1 = classifier.predict(x_test)

from sklearn.metrics import precision_recall_fscore_support as score
svc_acc = accuracy_score(y_test,y_pred1)
svc_pre_sc, svc_recall_sc, fscore, support = score(y_test, y_pred1)
svc_pre_sc = np.mean(svc_pre_sc)
svc_recall_sc = np.mean(svc_recall_sc)

from sklearn.neural_network import MLPClassifier
mdl = MLPClassifier(hidden_layer_sizes=(60,40),max_iter=500,random_state=0,max_fun=10000,verbose=True)
mdl.fit(x_train,y_train)
y_pred2 = mdl.predict(x_test)

mlp_acc = accuracy_score(y_test,y_pred2)
mlp_pre_sc, mlp_recall_sc, fscore, support = score(y_test, y_pred2)
mlp_pre_sc = np.mean(mlp_pre_sc)
mlp_recall_sc = np.mean(mlp_recall_sc)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(x_train, y_train)
y_pred3 = classifier.predict(x_test)

dt_acc = accuracy_score(y_test,y_pred3)
dt_pre_sc, dt_recall_sc, fscore, support = score(y_test, y_pred3)
dt_pre_sc = np.mean(dt_pre_sc)
dt_recall_sc = np.mean(dt_recall_sc)

import matplotlib.pyplot as plt
metrics = ['Accuracy', 'Recall', 'Precision']
data = np.array([[svc_acc, svc_recall_sc, svc_pre_sc],
                 [dt_acc, dt_recall_sc, dt_pre_sc],
                 [mlp_acc, mlp_recall_sc, mlp_pre_sc]])
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.2
ax.bar(x - width, data[0], width, label='SVC')
ax.bar(x, data[1], width, label='Decision Tree')
ax.bar(x + width, data[2], width, label='MLP')
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Algorithms Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
plt.tight_layout()
plt.show()