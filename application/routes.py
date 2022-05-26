from application import app
from flask import Flask, render_template, request

# plotting
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import pandas as pd 
import json 
import plotly 
import plotly.express as px

# label encoder
from sklearn.preprocessing import LabelEncoder

# feature selection
from sklearn.ensemble import ExtraTreesClassifier

# SMOTE
from imblearn.over_sampling import ADASYN

# scale data
from sklearn.preprocessing import StandardScaler

# splitting data
from sklearn.model_selection import train_test_split

# modelling
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# evaluation
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

# pickle
import pickle

df = pd.read_csv('./heart-failure.csv')
df_encode = pd.read_csv('./df_encode.csv')
final_df = pd.read_csv('./final_df.csv')

# Scaler data
scaler = StandardScaler()
X = scaler.fit_transform(final_df)
y = df_encode['target']

resample = ADASYN(sampling_strategy='all', random_state=42)
X, y = resample.fit_resample(X,y)

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modelling Logistic
lr = LogisticRegression()
lr.fit(x_train, y_train)
pred_cv1 = lr.predict(x_test)
score_lr = accuracy_score(y_test, pred_cv1)

# Modelling SVM
svm = SVC(probability=True)
svm.fit(x_train, y_train)
pred_cv2 = svm.predict(x_test)
score_svm = accuracy_score(y_test, pred_cv2)

# Modelling Tree
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
pred_cv3 = tree.predict(x_test)
score_tree = accuracy_score(y_test, pred_cv3)

# Modelling Naive Bayes
nb = GaussianNB()
nb.fit(x_train, y_train)
pred_cv4 = nb.predict(x_test)
score_nb = accuracy_score(y_test, pred_cv4)

# Modelling Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
pred_cv5 = rf.predict(x_test)
score_rf = accuracy_score(y_test, pred_cv5)

# Modelling KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
pred_cv6 = knn.predict(x_test)
score_knn = accuracy_score(y_test, pred_cv6)

MAPE = mean_absolute_percentage_error(y_test, pred_cv5)
MAE = mean_absolute_error(y_test, pred_cv5)
RMSE = sqrt(mean_squared_error(y_test, pred_cv5))

# Logistic
y_pred_logistic = lr.predict_proba(x_test)[:,1]
logistic_fpr, logistic_tpr, threshold_svm = metrics.roc_curve(y_test, y_pred_logistic)
logistic_auc = metrics.auc(logistic_fpr, logistic_tpr)

# svm
y_pred_svm = svm.predict_proba(x_test)[:,1]
svm_fpr, svm_tpr, threshold = metrics.roc_curve(y_test, y_pred_svm)
svm_auc = metrics.auc(svm_fpr, svm_tpr)

# tree
y_pred_tree = tree.predict_proba(x_test)[:,1]
tree_fpr, tree_tpr, threshold_tree = metrics.roc_curve(y_test, y_pred_tree)
tree_auc = metrics.auc(tree_fpr, tree_tpr)

# naive bayes
y_pred_nb = nb.predict_proba(x_test)[:,1]
nb_fpr, nb_tpr, threshold_nb = metrics.roc_curve(y_test, y_pred_nb)
nb_auc = metrics.auc(nb_fpr, nb_tpr)

# random forest
y_pred_rf = rf.predict_proba(x_test)[:,1]
rf_fpr, rf_tpr, threshold_rf = metrics.roc_curve(y_test, y_pred_rf)
rf_auc = metrics.auc(rf_fpr, rf_tpr)

# random forest
y_pred_knn = knn.predict_proba(x_test)[:,1]
knn_fpr, knn_tpr, threshold_knn = metrics.roc_curve(y_test, y_pred_knn)
knn_auc = metrics.auc(knn_fpr, knn_tpr)

# Tentang Dataset
@app.route("/home")
@app.route("/")
def home():
    df.sort_values('target', inplace=True)
    colors = ['navy', 'tomato']
    fig = px.line(df, x='time', title='Dead Distribution', color='target', text='age',
                  markers="o", color_discrete_sequence=colors, template="plotly_white",
                  animation_frame='age', animation_group='target')
    fig.update_layout(height=620)
    graph1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('home.html', graph1=graph1)

# Data Distribution
@app.route("/data")
def data():
    # graph 1
    feature = 'age'
    age = create_plot(feature)
    return render_template('data.html', plot=age)

def create_plot(feature):
    df.sort_values('target', inplace=True)
    if feature == 'age':
        age = df['age']
        group_labels = ['age']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=age, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Age Distribution')

    elif feature == 'anaemia':
        anaemia = df['anaemia']
        group_labels = ['anaemia']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=anaemia, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Anaemia Distribution')

    elif feature == 'creatinine_phosphokinase':
        creatinine_phosphokinase = df['creatinine_phosphokinase']
        group_labels = ['creatinine_phosphokinase']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=creatinine_phosphokinase, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Creatinine Phosphokinase Distribution')

    elif feature == 'diabetes':
        diabetes = df['diabetes']
        group_labels = ['diabetes']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=diabetes, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Diabetes Distribution')

    elif feature == 'ejection_fraction':
        ejection_fraction = df['ejection_fraction']
        group_labels = ['ejection_fraction']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=ejection_fraction, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Ejection Fraction Distribution')

    elif feature == 'high_blood_pressure':
        high_blood_pressure = df['high_blood_pressure']
        group_labels = ['high_blood_pressure']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=high_blood_pressure, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='High Blood Pressure Distribution')

    elif feature == 'platelets':
        platelets = df['platelets']
        group_labels = ['platelets']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=platelets, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Platelets Distribution')

    elif feature == 'serum_creatinine':
        serum_creatinine = df['serum_creatinine']
        group_labels = ['serum_creatinine']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=serum_creatinine, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Serum_creatinine Distribution')
    
    elif feature == 'serum_sodium':
        serum_sodium = df['serum_sodium']
        group_labels = ['serum_sodium']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=serum_sodium, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Serum_sodium Distribution')

    elif feature == 'sex':
        sex = df['sex']
        group_labels = ['sex']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=sex, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Sex Distribution')

    elif feature == 'smoking':
        smoking= df['smoking']
        group_labels = ['smoking']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=smoking, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Smoking Distribution')

    else:
        time = df['time']
        group_labels = ['time']
        colors = ['navy', 'tomato']
        target=df['target']
        data = px.histogram(x=time, color=target, color_discrete_sequence=colors, template="plotly_white")
        data.update_layout(bargap=0.1, title_text='Time Distribution')

    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/plotting', methods=['GET', 'POST'])
def change_features():
    feature = request.args['selected']
    graphJSON= create_plot(feature)

    return graphJSON

@app.route("/fitur_im")
def fitur_im():
    # graph 2
    X = df_encode.drop(['time','target'], axis=1)
    y = df_encode['target']
    model = ExtraTreesClassifier()
    model.fit(X,y)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.sort_values(inplace=True)
    colors = ['navy']
    feature_selection =  px.bar(feat_importances, orientation='h', color_discrete_sequence=colors, template="plotly_white")
    feature_selection.update_layout(title_text='Urutan Fitur yang Sangat Berpengaruh Terhadap Target')
    graph_fs = json.dumps(feature_selection, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('fitur_im.html', graph_fs=graph_fs)

@app.route("/fitur_im_2")
def fitur_im_2():

    return render_template('fitur_im_2.html')

# Modelling
@app.route("/model")
def model():
    fitur = 'rf'
    rf = buat_plot(fitur)

    return render_template('model.html', score_lr=score_lr, score_svm=score_svm, score_tree=score_tree, 
                            score_nb=score_nb, score_rf=score_rf, score_knn=score_knn,
                             MAPE=MAPE, MAE=MAE, RMSE=RMSE, plot=rf)

def buat_plot(fitur):
    if fitur == 'logistic':
        colors = ['tomato']
        roc = px.area(x=logistic_fpr, y=logistic_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC Logistic Regression')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"Logistic Regression AUC={logistic_auc:.2f}", showarrow=False)
    elif fitur == 'svm':
        colors = ['tomato']
        roc = px.area(x=svm_fpr, y=svm_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC SVM')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"SVM AUC={svm_auc:.2f}", showarrow=False)
    elif fitur == 'tree':
        colors = ['tomato']
        roc = px.area(x=tree_fpr, y=tree_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC Decision Tree')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"Decisition Tree AUC={tree_auc:.2f}", showarrow=False)
    elif fitur == 'nb':
        colors = ['tomato']
        roc = px.area(x=nb_fpr, y=nb_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC Naive Bayes')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"Naive Bayes AUC={nb_auc:.2f}", showarrow=False)
    elif fitur == 'rf':
        colors = ['tomato']
        roc = px.area(x=rf_fpr, y=rf_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC Random Forest')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"Random Forest AUC={rf_auc:.2f}", showarrow=False)
    elif fitur == 'knn':
        colors = ['tomato']
        roc = px.area(x=knn_fpr, y=knn_tpr, color_discrete_sequence=colors, template="plotly_white")
        roc.update_layout(title_text='Kurva ROC KNN')
        roc.add_annotation(x=0.5, y=0.5,
                            text=f"KNN AUC={knn_auc:.2f}", showarrow=False)
    
    rocJSON = json.dumps(roc, cls=plotly.utils.PlotlyJSONEncoder)
    return rocJSON

@app.route('/chart', methods=['GET', 'POST'])
def ubah_fitur():
    fitur = request.args['terpilih']
    rocJSON= buat_plot(fitur)

    return rocJSON
    
@app.route("/pred", methods=['GET','POST'])
def pred():
    if request.method == "POST":
        ef = request.form["ef"]
        sc = request.form["sc"]
        umur = request.form["umur"]
        ss = request.form["ss"]
        cp = request.form["cp"]
        pl = request.form["pl"]

        pred_list = pd.DataFrame([[ef, sc, umur, ss, cp, pl]])
        sample = np.array(pred_list)
        sample_re = sample.reshape(1,-1)
        prediction = rf.predict(sample_re)

        output = {
            1 : "Ya, ada kemungkinan pasien ini meninggal akibat gagal jantung.",
            0 : "Tidak ada kemungkinan."
        }

        return render_template('predict.html',prediction=output[prediction[0]])

    return render_template('predict.html')

