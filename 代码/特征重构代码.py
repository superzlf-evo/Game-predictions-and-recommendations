#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
id：玩家记录id
win：是否胜利，标签变量
kills：击杀次数
deaths：死亡次数
assists：助攻次数
largestkillingspree：最大 killing spree（游戏术语，意味大杀特杀。当你连续杀死三个对方英雄而中途没有死亡时）
largestmultikill：最大mult ikill（游戏术语，短时间内多重击杀）
longesttimespentliving：最长存活时间
doublekills：doublekills次数
triplekills：doublekills次数
quadrakills：quadrakills次数
pentakills：pentakills次数
totdmgdealt：总伤害
magicdmgdealt：魔法伤害
physicaldmgdealt：物理伤害
truedmgdealt：真实伤害
largestcrit：最大暴击伤害
totdmgtochamp：对对方玩家的伤害
magicdmgtochamp：对对方玩家的魔法伤害
physdmgtochamp：对对方玩家的物理伤害
truedmgtochamp：对对方玩家的真实伤害
totheal：治疗量
totunitshealed：痊愈的总单位
dmgtoturrets：对炮塔的伤害
timecc：法控时间
totdmgtaken：承受的伤害
magicdmgtaken：承受的魔法伤害
physdmgtaken：承受的物理伤害
truedmgtaken：承受的真实伤害
wardsplaced：侦查守卫放置次数
wardskilled：侦查守卫摧毁次数
firstblood：是否为firstblood 测试集中label字段win为空，需要选手预测。
"""


# In[1]:


import pandas as pd

file_path = 'C:\\Users\\17436\\Desktop\\AIWin\\train.csv'
df = pd.read_csv(file_path)

df.head()


# In[2]:


from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# 原数据中相关特征
top_lane_features = ['kills', 'deaths']

mid_lane_features = ['kills', 'assists', 'magicdmgdealt', 'magicdmgtochamp']

bot_lane_features = ['wardsplaced', 'wardskilled', 'totdmgtochamp']

jungle_features = ['totdmgtochamp', 'timecc', 'assists', 'wardsplaced', 'wardskilled', 'largestkillingspree']

# 归一化
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[top_lane_features + mid_lane_features + bot_lane_features + jungle_features] = scaler.fit_transform(
    df[top_lane_features + mid_lane_features + bot_lane_features + jungle_features]
)

# 新特征的计算
df_normalized['TopLane'] = df_normalized[top_lane_features].sum(axis=1)
df_normalized['MidLane'] = df_normalized[mid_lane_features].sum(axis=1)
df_normalized['BotLane'] = df_normalized[bot_lane_features].sum(axis=1)
df_normalized['Jungle'] = df_normalized[jungle_features].sum(axis=1)
df_normalized['CooperationAbility'] = (df['kills'] + df['assists']) / df['kills']
df_normalized['CooperationAbility'].replace([np.inf, -np.inf], 0, inplace=True)

# 创建新特征DataFrame
new_features = ['TopLane', 'MidLane', 'BotLane', 'Jungle', 'CooperationAbility']
new_df = df_normalized[new_features]

new_df.head()


# In[3]:


import numpy as np

# 协作能力定义
df_normalized['CooperationAbility'] = (df['kills'] + df['assists']) / df['kills']
df_normalized['CooperationAbility'].replace([np.inf, -np.inf], 0, inplace=True)

# 创建新特征DataFrame
new_features = ['TopLane', 'MidLane', 'BotLane', 'Jungle', 'CooperationAbility']
new_df = df_normalized[new_features]

new_df.head()


# In[4]:


# 将原数据集中的win列添加到新的DataFrame中
new_df_with_win = new_df.copy()
new_df_with_win['win'] = df['win']

new_df_with_win.head()


# In[9]:


# 保存
new_file_path_with_win = 'C:\\Users\\17436\\Desktop\\AIWin\\processed_train.csv'
new_df_with_win.to_csv(new_file_path_with_win, index=False)

new_file_path_with_win


# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, classification_report

data = pd.read_csv('C:\\Users\\17436\\Desktop\\AIWin\\processed_train.csv')

# 选择特征列
features = ['TopLane', 'MidLane', 'BotLane', 'Jungle', 'CooperationAbility']

# 创建特征矩阵 X 和目标变量 y
X = data[features]
y = data['win']

X.fillna(0, inplace=True)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 建立随机森林分类模型
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test)

# 计算模型评估指标
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印三线表形式的模型评估结果
report = classification_report(y_test, y_pred, target_names=['loss', 'win'], digits=4)

print("准确率：", accuracy)
print("召回率：", recall)
print("F1值：", f1)
print("\n分类报告：\n", report)


# In[ ]:





# In[8]:


# 读取数据
file_path = 'C:\\Users\\17436\\Desktop\\AIWin\\train1.csv'
data = pd.read_csv(file_path)

# 向数据集中添加新特征
data['EfficiencyIndex'] = (data['kills'] + data['assists']) / data['deaths'].replace(0, 10)
data['TeamContributionRate'] = data['assists'] / data['team_total_assists']
data['DamageContributionRate'] = data['totdmgtochamp'] / data['totdmgdealt']


# In[9]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


# In[10]:


# 检查数据中是否存在NaN和无穷大值
has_nan = np.isnan(data)
has_inf = np.isinf(data)

if np.any(has_nan) or np.any(has_inf):
    # 处理NaN和无穷大值，例如替换为合适的值或删除相关行/列
    data[has_nan] = 0  # 将NaN替换为0
    data[has_inf] = np.nan  # 将无穷大替换为NaN

# 转换数据类型为float32
data = data.astype(np.float32)

# 选择特征和目标变量
X = data[['EfficiencyIndex', 'TeamContributionRate', 'DamageContributionRate']]
y = data['win']

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练 GradientBoostingClassifier 模型
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# 预测并评估模型
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印评估指标
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


# In[13]:


# 检查数据中是否存在NaN和无穷大值
has_nan = np.isnan(data)
has_inf = np.isinf(data)

if np.any(has_nan) or np.any(has_inf):
    # 处理NaN和无穷大值，例如替换为合适的值或删除相关行/列
    data[has_nan] = 0  # 将NaN替换为0
    data[has_inf] = np.nan  # 将无穷大替换为NaN

# 转换数据类型为float32
data = data.astype(np.float32)

data['EfficiencyIndex'] = (data['kills'] + data['assists']) / data['deaths'].replace(0, 10)
data['TeamContributionRate'] = data['assists'] / data['team_total_assists']
data['DamageContributionRate'] = data['totdmgtochamp'] / data['totdmgdealt']
data['Survivability'] = data['longesttimespentliving'] / data['Total game time']
data['VisionContribution'] = (data['wardsplaced'] + data['wardskilled']) / data['Total game time']

# 检查数据中是否存在NaN和无穷大值
has_nan = np.isnan(data)
has_inf = np.isinf(data)

if np.any(has_nan) or np.any(has_inf):
    # 处理NaN和无穷大值，例如替换为合适的值或删除相关行/列
    data[has_nan] = 0  # 将NaN替换为0
    data[has_inf] = np.nan  # 将无穷大替换为NaN

# 转换数据类型为float32
data = data.astype(np.float32)

# 选择特征和目标变量
features = ['EfficiencyIndex', 'TeamContributionRate', 'DamageContributionRate', 'VisionContribution', 'Survivability']
X = data[features]
y = data['win']

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型并评估
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印带有所有特征的模型评估指标
# 计算模型评估指标
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 打印带有所有特征的模型评估指标
print('模型评估（所有特征）:')
print(classification_report(y_test, y_pred, target_names=['loss', 'win']))

# 消融实验
for feature in features:
    X_ablation = X_train.drop(columns=[feature])
    model.fit(X_ablation, y_train)
    y_pred_ablation = model.predict(X_test.drop(columns=[feature]))
    precision_ablation = precision_score(y_test, y_pred_ablation)
    recall_ablation = recall_score(y_test, y_pred_ablation)
    f1_ablation = f1_score(y_test, y_pred_ablation)

    print(f'移除特征 {feature} 后的模型评估:')
    print(f'Precision: {precision_ablation}')
    print(f'Recall: {recall_ablation}')
    print(f'F1 Score: {f1_ablation}\n')

# 特征重要性
feature_importance = model.feature_importances_
print('特征重要性:')
for feature, importance in zip(features, feature_importance):
    print(f'{feature}: {importance}')


# In[ ]:





# In[15]:


data = {
    'Model': ['All Features', 'Without EfficiencyIndex', 'Without TeamContributionRate', 'Without DamageContributionRate', 'Without VisionContribution', "Without Survivability"],
    'Precision': [0.794433186315413, 0.6414822439526505, 0.7893087236635905, 0.7955374768200036, 0.7887660069848661, 0.7900976290097629],
    'Recall': [0.7367751920844618, 0.7109612514509978, 0.7889005582886518, 0.7751920844618871, 0.7871317229561661, 0.75153391188989],
    'F1 Score': [0.685571131602793, 0.7987463622117752, 0.7947860583734768, 0.7972454721048065, 0.7621515779774245, 0.770333437207853
]
}

df_evaluation = pd.DataFrame(data)

df_evaluation


# In[7]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from six import iteritems

df_evaluation = pd.DataFrame({
    'Model': [
        'All Features', 
        'Without EfficiencyIndex', 
        'Without TeamContributionRate', 
        'Without DamageContributionRate', 
        'Without VisionContribution'
    ],
    'Precision': [
        0.794433186315413, 
        0.6344989744359397, 
        0.7984449400277259, 
        0.7961886498471864, 
        0.79545591152251
    ],
    'Recall': [
        0.7367751920844618, 
        0.7010668288099055, 
        0.7322425515449671, 
        0.7343983196064341, 
        0.7315239621911448
    ],
    'F1 Score': [
        0.7645186268605352, 
        0.6661239495798318, 
        0.7639121157949368, 
        0.7640462361262866, 
        0.7621515779774245
    ]
})

# df_evaluation = pd.DataFrame({
#     'Model': [
#         'All Features', 
#         'Without EfficiencyIndex', 
#         'Without TeamContributionRate', 
#         'Without DamageContributionRate', 
#         'Without VisionContribution'
#     ],
#     'Precision': [
#         0.7631, 
#         0.6344, 
#         0.7984, 
#         0.7861, 
#         0.7954
#     ],
#     'Recall': [
#         0.7628 , 
#         0.7010, 
#         0.7312, 
#         0.7143, 
#         0.7115
#     ],
#     'F1 Score': [
#         0.7628, 
#         0.6661, 
#         0.7639, 
#         0.7521, 
#         0.7621
#     ]
# })

def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors)])
    return ax

# ax = render_mpl_table(df_evaluation, header_columns=0, col_width=2.5, font_size=7)  # 设置字体大小为10


def render_mpl_table(data, col_width=3.5, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height]))
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    # 设置输出图像的分辨率（dpi）
    fig.set_dpi(300)

    return ax

ax = render_mpl_table(df_evaluation, col_width=2.5, font_size=7)


# In[18]:


df_evaluation = pd.DataFrame({
    'Model': [
        'All Features', 
        'Without EfficiencyIndex', 
        'Without TeamContributionRate', 
        'Without DamageContributionRate', 
        'Without VisionContribution'
    ],
    'Precision': [
        0.794433186315413, 
        0.6344989744359397, 
        0.7984449400277259, 
        0.7961886498471864, 
        0.79545591152251
    ],
    'Recall': [
        0.7367751920844618, 
        0.7010668288099055, 
        0.7322425515449671, 
        0.7343983196064341, 
        0.7315239621911448
    ],
    'F1 Score': [
        0.7645186268605352, 
        0.6661239495798318, 
        0.7639121157949368, 
        0.7640462361262866, 
        0.7621515779774245
    ]
})


plt.style.use('seaborn-whitegrid')

fig, ax = plt.subplots(figsize=(12, 8))

pos = np.arange(len(df_evaluation['Model']))
bar_width = 0.25

precision_bars = ax.bar(pos - bar_width, df_evaluation['Precision'], bar_width, label='Precision', alpha=0.8, color='#1f77b4')
recall_bars = ax.bar(pos, df_evaluation['Recall'], bar_width, label='Recall', alpha=0.8, color='#ff7f0e')
f1_score_bars = ax.bar(pos + bar_width, df_evaluation['F1 Score'], bar_width, label='F1 Score', alpha=0.8, color='#2ca02c')

ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Model Evaluation Metrics', fontsize=18)
ax.set_xticks(pos)
ax.set_xticklabels(df_evaluation['Model'], fontsize=9)

# 定义柱状图添加标签
def autolabel(bars):

    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(round(height, 4)),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(precision_bars)
autolabel(recall_bars)
autolabel(f1_score_bars)

# 将图例移到绘图外
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 调整布局
fig.tight_layout()

plt.show()


# In[5]:


models = ['All Features', 'Without EfficiencyIndex', 'Without TeamContributionRate', 'Without DamageContributionRate', 'Without VisionContribution']
precision = [0.7944, 0.6345, 0.7984, 0.7962, 0.7955]
recall = [0.7368, 0.7011, 0.7322, 0.7344, 0.7315]
f1_score = [0.7645, 0.6661, 0.7639, 0.7640, 0.7622]

# 特征重要性数据
features = ['EfficiencyIndex', 'TeamContributionRate', 'DamageContributionRate', 'VisionContribution']
importance = [0.9958, 0.1541, 0.1001, 0.1000]

# 创建第一个子图 - 模型评估数据的折线图
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制模型评估数据的折线图
color = 'tab:blue'
ax1.set_xlabel('Models', fontsize=12)  # 修改x轴标签字体大小
ax1.set_ylabel('Precision / Recall / F1 Score', color=color)
ax1.plot(models, precision, marker='o', label='Precision', color='lightblue', alpha=0.7)
ax1.plot(models, recall, marker='o', label='Recall', color='lightgreen', alpha=0.7)
ax1.plot(models, f1_score, marker='o', label='F1 Score', color='lightcoral', alpha=0.7)
ax1.tick_params(axis='y', labelcolor=color)

# 设置x轴标签的字体大小
ax1.tick_params(axis='x', labelsize=7)  # 修改x轴标签字体大小为10

# 添加图例
ax1.legend(loc='lower right', fontsize=10)

# 创建第二个子图  特征重要性数据的折线图
fig, ax2 = plt.subplots(figsize=(10, 6))

# 绘制特征重要性数据的折线图
color = 'tab:red'
ax2.set_xlabel('Features', fontsize=12)  # 修改x轴标签字体大小
ax2.set_ylabel('Feature Importance', color=color)
ax2.plot(features, importance, marker='s', linestyle='--', label='Feature Importance', color='red', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

# 添加网格线
plt.grid(True)

# 添加图例
ax2.legend(loc='upper right', fontsize=10)

# 添加标题
fig.suptitle('Model Evaluation and Feature Importance', fontsize=14)

# 调整布局
fig.tight_layout(rect=[0, 0, 1, 0.9])

plt.show()


# In[ ]:





# In[3]:


import matplotlib.pyplot as plt

feature_importances = [0.33950955703654095, 0.31573052177254823, 0.026931313819451404, 0.2388470971853273, 0.049197508943507544, 0.00018575980079048727, 0.0014190814148361962, 0.007489186387552001, 0.020063840359495117, 0.0006261332799509516]

features = ["dmgtoturrets", "deaths", "largestkillingspree", "assists", "kills", "largestmultikill", "doublekills", "totdmgtochamp", "totdmgdealt", "triplekills"]

fig, ax2 = plt.subplots(figsize=(10, 6))

# 绘制特征重要性折线图
color = 'tab:red'
ax2.set_xlabel('Features', fontsize=12)
ax2.set_ylabel('Feature Importance', color=color)
ax2.plot(features, feature_importances, marker='s', linestyle='--', label='Feature Importance', color='red', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

# 添加网格线
plt.grid(True)

# 添加图例
ax2.legend(loc='upper right', fontsize=10)

# 添加标题
fig.suptitle('Feature Importance', fontsize=14)

# 旋转x轴标签以避免重叠
plt.xticks(rotation=45)

plt.show()


# In[ ]:




