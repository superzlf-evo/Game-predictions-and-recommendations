import eli5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings as wn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import itertools
import pdpbox
from eli5.sklearn import PermutationImportance
# from eli5.sklearn import PermutationImportance
from sklearn.metrics import roc_curve, auc, mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
wn.filterwarnings("ignore")
from pdpbox import info_plots,get_dataset,pdp,get_dataset
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from pdpbox import pdp,get_dataset,info_plots
df=pd.read_csv('train1.csv')

##特征两两相关性分析
df.corr()
plt.figure(figsize=(15,15))
# sns.heatmap(df.corr(),annot=True,fmt=".lf",square=True)
#annot=True:把数字写在图标上，fmt=".1f：保留一位小数，square=True：图是方形的
sns.heatmap(df.corr(),annot=True,fmt=".1f",square=True)
# plt.show()
plt.savefig('pic/所有特征之间关系heatmap.png', dpi=300, bbox_inches='tight')

df1=df[['deaths','dmgtoturrets','assists','largestkillingspree','kills']]
df1.corr()
plt.figure(figsize=(15,15))
sns.heatmap(df1.corr(),annot=True,fmt=".1f",square=True)
# plt.show()
plt.savefig('pic/重要特征关系直接关系heatmap.png', dpi=300, bbox_inches='tight')

df=pd.get_dummies(df)
X=df.drop('win',axis=1)
y=df['win']



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

model=RandomForestClassifier(max_depth=5,n_estimators=100,random_state=5)
model.fit(X_train,y_train)
#将特征值转化为字符串
feature_names=X_train.columns
y_train_str=y_train.astype("str")
y_train_str[y_train_str=='0']='no win'
y_train_str[y_train_str=='1']='win'
y_train_str=y_train_str.values

feature_names=X_test.columns
feature_importances=model.feature_importances_
indices=np.argsort(feature_importances)[::-1]

y_pred=model.predict(X_test)
y_pred_proba=model.predict_proba(X_test)
base_features=df.columns.values.tolist()
base_features.remove("win")

#选取'deaths','dmgtoturrets','assists','largestkillingspree','kills'

#deaths分析
fig,axes,summary_df=info_plots.target_plot(df=df,feature='deaths',feature_name='deaths',target=['win'])
plt.savefig('pic/deathsPDP图.png', dpi=300, bbox_inches='tight')
feat_name='deaths'
nick_name='deaths'
pdp_dist=pdp.pdp_isolate(model=model,dataset=X_test,model_features=base_features,feature=feat_name)
fig,axes=pdp.pdp_plot(pdp_dist,nick_name)
plt.savefig('pic/deathsICE图.png', dpi=300, bbox_inches='tight')

#dmgtoturrets分析
fig,axes,summary_df=info_plots.target_plot(df=df,feature='dmgtoturrets',feature_name='dmgtoturrets',target=['win'])
plt.savefig('pic/dmgtoturretsPDP图.png', dpi=300, bbox_inches='tight')
feat_name='dmgtoturrets'
nick_name='dmgtoturrets'
pdp_dist=pdp.pdp_isolate(model=model,dataset=X_test,model_features=base_features,feature=feat_name)
fig,axes=pdp.pdp_plot(pdp_dist,nick_name)
plt.savefig('pic/dmgtoturretsICE图.png', dpi=300, bbox_inches='tight')

#assists分析
fig,axes,summary_df=info_plots.target_plot(df=df,feature='assists',feature_name='assists',target=['win'])
plt.savefig('pic/assistsPDP图.png', dpi=300, bbox_inches='tight')
feat_name='assists'
nick_name='assists'
pdp_dist=pdp.pdp_isolate(model=model,dataset=X_test,model_features=base_features,feature=feat_name)
fig,axes=pdp.pdp_plot(pdp_dist,nick_name)
plt.savefig('pic/assistsICE图.png', dpi=300, bbox_inches='tight')

#largestkillingspree分析
fig,axes,summary_df=info_plots.target_plot(df=df,feature='largestkillingspree',feature_name='largestkillingspree',target=['win'])
plt.savefig('pic/largestkillingspreePDP图.png', dpi=300, bbox_inches='tight')
feat_name='largestkillingspree'
nick_name='largestkillingspree'
pdp_dist=pdp.pdp_isolate(model=model,dataset=X_test,model_features=base_features,feature=feat_name)
fig,axes=pdp.pdp_plot(pdp_dist,nick_name)
plt.savefig('pic/largestkillingspreeICE图.png', dpi=300, bbox_inches='tight')

#kills分析
fig,axes,summary_df=info_plots.target_plot(df=df,feature='kills',feature_name='kills',target=['win'])
plt.savefig('pic/killsPDP图.png', dpi=300, bbox_inches='tight')
feat_name='kills'
nick_name='kills'
pdp_dist=pdp.pdp_isolate(model=model,dataset=X_test,model_features=base_features,feature=feat_name)
fig,axes=pdp.pdp_plot(pdp_dist,nick_name)
plt.savefig('pic/killsICE图.png', dpi=300, bbox_inches='tight')




#kills和largestkillingspree对结果影响PDP图
feat_name1="kills"
nick_name1="kills"
feat_name2="largestkillingspree"
nick_name2="largestkillingspree"
inter1=pdp.pdp_interact(model=model,dataset=X_test,model_features=base_features,features=[feat_name1,feat_name2])
fig,axes=pdp.pdp_interact_plot(inter1,[nick_name1,nick_name2],plot_type="grid",x_quantile=True,plot_pdp=True)
plt.savefig('pic/kills和largestkillingspree对结果影响PDP图.png', dpi=300, bbox_inches='tight')

#kills和dmgtoturrets对结果影响PDP图
feat_name1="kills"
nick_name1="kills"
feat_name2="dmgtoturrets"
nick_name2="dmgtoturrets"
inter1=pdp.pdp_interact(model=model,dataset=X_test,model_features=base_features,features=[feat_name1,feat_name2])
fig,axes=pdp.pdp_interact_plot(inter1,[nick_name1,nick_name2],plot_type="grid",x_quantile=True,plot_pdp=True)
plt.savefig('pic/kills和dmgtoturrets对结果影响PDP图.png', dpi=300, bbox_inches='tight')

#largestkillingspree和dmgtoturrets对结果影响PDP图
feat_name1="largestkillingspree"
nick_name1="largestkillingspree"
feat_name2="dmgtoturrets"
nick_name2="dmgtoturrets"
inter1=pdp.pdp_interact(model=model,dataset=X_test,model_features=base_features,features=[feat_name1,feat_name2])
fig,axes=pdp.pdp_interact_plot(inter1,[nick_name1,nick_name2],plot_type="grid",x_quantile=True,plot_pdp=True)
plt.savefig('pic/largestkillingspree和dmgtoturrets对结果影响PDP图.png', dpi=300, bbox_inches='tight')