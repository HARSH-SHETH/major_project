import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import export_graphviz

data = pd.read_csv("tmp.csv")
data['label'] = data['label'].apply(lambda x: (True if x == 'good' else False))
data['who_is'] = data['who_is'].apply(lambda x: (True if x == 'complete' else False))
data['https'] = data['https'].apply(lambda x: (True if x == 'yes' else False))
data = data[['url_len','ip_add','geo_loc','tld','who_is','https','js_len','js_obf_len','content','label']]
ipEncoder = LabelEncoder()
geoEncoder = LabelEncoder()
tldEncoder = LabelEncoder()
data['ip_add'] = ipEncoder.fit_transform(data['ip_add'])
data['geo_loc'] = geoEncoder.fit_transform(data['geo_loc'])
data['tld'] = tldEncoder.fit_transform(data['tld'])
print(data.head())

print(data.size)
x = data[['url_len','ip_add','geo_loc','tld','who_is','https','js_len','js_obf_len','content']]
y = data['label']

xtrain = x[:7000]
ytrain = y[:7000]
xtest = x[7000:10000]
ytest = y[7000:10000]

rfClf = RandomForestClassifier()
gbClf = GradientBoostingClassifier()
rfClf.fit(xtrain,ytrain)
gbClf.fit(xtrain,ytrain)

for idx,estimator in enumerate(rfClf.estimators_):
    export_graphviz(estimator, out_file=f'rf/RF{idx}.dot',feature_names=x.columns, filled=True, rounded=True)
    os.system(f"dot -Tsvg rf/RF{idx}.dot -o rf/RF{idx}.svg")
os.system("rm rf/*.dot")

for idx,estimator in enumerate(gbClf.estimators_):
    export_graphviz(estimator[0], out_file=f'gb/GB{idx}.dot',feature_names=x.columns, filled=True, rounded=True)
    os.system(f"dot -Tsvg gb/GB{idx}.dot -o gb/gb{idx}.svg")
os.system(f"rm gb/*.dot")

ypred_rf = rfClf.predict(xtest)
ypred_gb = gbClf.predict(xtest)
print("Accuracy of Random Forest: ", accuracy_score(ytest, ypred_rf))
print("Accuracy of Gradient Boosting: ", accuracy_score(ytest, ypred_gb))
