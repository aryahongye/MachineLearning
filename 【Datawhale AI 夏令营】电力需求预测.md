## 电力需求预测挑战赛(讯飞）2024.7.1——8.15

[赛事链接](https://challenge.xfyun.cn/h5/detail?type=electricity-demand&ch=dw24_uGS8Gs)


<details>
<summary>赛题剖析</summary>

  
### 【训练时序预测模型助力电力需求预测】

电力需求的准确预测对于电网的稳定运行、能源的有效管理以及可再生能源的整合至关重要。

本赛题是一个典型的时间序列问题

时间序列问题是指对按时间顺序排列的数据点进行分析和预测的问题，往往用来做未来的趋势预测。比如，基于历史股票每天的股价，预测未来股票的价格走向。

```markdown
常见的时间序列场景有：
1. 金融领域：股票价格预测、利率变动、汇率预测等。
2. 气象领域：温度、降水量、风速等气候指标的预测。
3. 销售预测：产品或服务的未来销售额预测。
4. 库存管理：预测库存需求，优化库存水平。
5. 能源领域：电力需求预测、石油价格预测等。
6. 医疗领域：疾病爆发趋势预测、医疗资源需求预测。

时间序列问题的数据往往有如下特点：
1. 时间依赖性：数据点之间存在时间上的连续性和依赖性。
2. 非平稳性：数据的统计特性（如均值、方差）随时间变化。
3. 季节性：数据表现出周期性的模式，如年度、月度或周度。
4. 趋势：数据随时间推移呈现长期上升或下降的趋势。
5. 周期性：数据可能存在非固定周期的波动。
6. 随机波动：数据可能受到随机事件的影响，表现出不确定性。
```

```markdown
规律：
1.周期性变化：数据中存在明显的周期性波动。目标值在一定时间间隔内会出现上升和下降的趋势，这可能与某些周期性事件或规律相关。
2.波动性：数据点之间的波动较大，目标值在不同时间点之间有显著的变化。这表明目标值可能受到多种因素的影响，而这些因素在不同时间点表现出不同的强度。
3.突发性峰值：图表中存在一些突发性的峰值，例如在大约第 100、200 和 450 单位时间左右，这些峰值明显高于周围的数据点。这可能表示在这些时间点上发生了一些特殊事件或情况，导致目标值急剧上升。
4.总体趋势：尽管存在波动和突发性峰值，从总体上看，目标值在整个时间段内有一定的变化趋势。例如，在前期（0 到 200 单位时间）目标值较低，然后在中期（200 到 400 单位时间）逐渐增加，最后在后期（400 到 500 单位时间）再次波动下降。
```

时间序列预测问题可以通过多种建模方法来解决，包括传统的时间序列模型、机器学习模型和深度学习模型。

以下是这三种方法的建模思路、优缺点对比：

```markdown
传统时间序列模型
- 基于时间序列数据的统计特性，如自相关性、季节性等。
- 使用ARIMA、季节性ARIMA（SARIMA）、指数平滑等模型。
- 通过识别数据的趋势和季节性成分来构建模型。
- 模型结构简单，易于理解和解释。
- 计算效率高，适合于数据量较小的问题。
- 直接针对时间序列数据设计，能够很好地处理数据的季节性和趋势。
- 对于非线性模式和复杂的时间序列数据，预测能力有限。
- 需要手动进行参数选择和模型调整。
- 对数据的平稳性有严格要求，非平稳数据需要差分等预处理。

机器学习模型
- 将时间序列数据转换为监督学习问题，使用历史数据作为特征，未来值作为标签。
- 使用决策树、随机森林、梯度提升树等模型。
- 通过特征工程来提取时间序列数据中的有用信息。
- 能够处理非线性关系和复杂的数据模式。
- 通过特征工程可以引入额外的解释性变量。
- 模型选择多样，可以进行模型融合以提高预测性能。
- 对于时间序列数据的内在时间结构和季节性可能不够敏感。
- 需要大量的特征工程工作。
- 模型的解释性可能不如传统时间序列模型。

深度学习模型
- 使用循环神经网络（RNN）、长短期记忆网络（LSTM）或一维卷积神经网络（1D-CNN）等模型。
- 能够捕捉时间序列数据中的长期依赖关系。
- 通过训练大量的参数来学习数据的复杂模式。
- 能够处理非常复杂的数据模式和长期依赖关系。
- 适用于大量数据，可以自动提取特征。
- 模型的灵活性和适应性强。
- 需要大量的数据和计算资源。
- 模型训练和调优可能比较复杂和耗时。
- 模型的解释性较差，难以理解预测结果的原因。

对比总结
- 适用性：传统模型适合数据量较小、模式简单的问题；机器学习模型适合中等复杂度的问题，可以引入额外变量；深度学习模型适合数据量大、模式复杂的任务。
- 解释性：传统时间序列模型通常具有较好的解释性；机器学习模型的解释性取决于特征工程；深度学习模型的解释性通常较差。
- 计算资源：传统模型计算效率最高；机器学习模型次之；深度学习模型通常需要最多的计算资源。
- 预测能力：深度学习模型在捕捉复杂模式方面具有优势，但需要大量数据支持；传统和机器学习模型在数据量较小或模式较简单时可能更有效。
在实际应用中，选择哪种模型取决于具体问题的需求、数据的特性以及可用的计算资源。有时，结合多种方法的混合模型可以提供更好的预测性能。
```
</details>

### 赛题任务
给定多个房屋对应电力消耗历史N天的相关序列数据等信息，预测房屋对应电力的消耗。


### 赛题数据简介
赛题数据由训练集和测试集组成，为了保证比赛的公平性，将每日日期进行脱敏，用1-N进行标识。

即1为数据集最近一天，其中1-10为测试集数据。


数据集由字段id（房屋id）、 dt（日标识）、type（房屋类型）、target（实际电力消耗）组成。


<details>
<summary>7.13版本1：基于经验模型，使用均值作为结果数据</summary>summary>
### 7.13版本1：基于经验模型，使用均值作为结果数据

#### 代码

```python
# 1. 导入需要用到的相关库
# 导入 pandas 库，用于数据处理和分析
import pandas as pd
# 导入 numpy 库，用于科学计算和多维数组操作
import numpy as np

# 2. 读取训练集和测试集
# 使用 read_csv() 函数从文件中读取训练集数据，文件名为 'train.csv'
train = pd.read_csv('./data/data283931/train.csv')
# 使用 read_csv() 函数从文件中读取测试集数据，文件名为 'train.csv'
test = pd.read_csv('./data/data283931/test.csv')

# 3. 计算训练数据最近11-20单位时间内对应id的目标均值
target_mean = train[train['dt']<=20].groupby(['id'])['target'].mean().reset_index()
```

##### 代码分解
```markdown
选择时间范围：
train[train['dt'] <= 20]
选择 dt 小于或等于 20 的所有数据行。dt 表示时间单位。

分组：
.groupby(['id'])
在选择了时间范围内的数据上按 id 进行分组。每个 id 表示一个独立的实体或记录的标识符。

计算目标均值：
['target'].mean()
对每个分组（即每个 id）计算 target 列的均值。

重置索引：
.reset_index()
重置分组结果的索引，使得返回的结果是一个新的 DataFrame，其中包含 id 和对应的 target 均值。
```

```python
# 4. 将target_mean作为测试集结果进行合并
test = test.merge(target_mean, on=['id'], how='left')

# 5. 保存结果文件到本地
test[['id','dt','target']].to_csv('submit.csv', index=None)
```

##### 代码分解

```markdown
目标：将 target_mean 与 test 数据集合并，使得 test 数据集包含每个 id 对应的目标均值 target。

merge 方法：
on=['id']：指定以 id 列作为键进行合并，即以 id 列为基础，将 target_mean 中的 target 列合并到 test 中。
how='left'：指定使用左连接（left join）进行合并。这意味着所有来自 test DataFrame 的行都将保留，如果在 target_mean 中找不到匹配的 id，则对应的 target 值将为 NaN。
```
</details>


### 7.15版本2：LightGBM,特征工程

```markdown
- 使用数据集绘制柱状图和折线图，
- 使用时间序列数据构建历史平移特征和窗口统计特征，
- 使用lightgbm模型进行训练并预测。
```

#### 思路
机器学习中的一个经典理论是：**数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限。**

回归预测问题的思路：
- 使用机器学习模型，如LightGBM、XGBoost
- 使用深度学习模型（神经网络等）进行实践，在模型的搭建上就比较复杂，需要自己构建模型结构，对于数值数据需要进行标准化处理；


使用机器学习方法有哪几个步骤？
- 探索性数据分析
- 数据预处理
- 提取特征
- 切分训练集与验证集
- 训练模型
- 预测结果

**GBDT**
- 机器学习中一个长盛不衰的模型
- 主要思想：利用弱分类器（决策树）迭代训练以得到最优模型
- 优点：训练效果好、不易过拟合
- 在工业界应用广泛：通常被用于多分类、点击率预测、搜索排序等任务
- 在各种数据挖掘竞赛中也是致命武器，据统计Kaggle上的比赛有一半以上的冠军方案都是基于GBDT

**LightGBM**
- Light Gradient Boosting Machine是一个实现GBDT算法的框架，支持高效率的并行训练，并且具有更快的训练速度、更低的内存消耗、更好的准确率、支持分布式可以快速处理海量数据等优点。
- 包括随机森林和逻辑回归等模型。通常应用于二分类、多分类和排序等场景。
  
例如：在个性化商品推荐场景中，通常需要做点击预估模型。使用用户过往的行为（点击、曝光未点击、购买等）作为训练数据，来预测用户点击或购买的概率。根据用户行为和用户属性提取一些特征，包括：
- 类别特征（Categorical Feature）：字符串类型，如性别（男/女）。
- 物品类型：服饰、玩具和电子等。
- 数值特征（Numrical Feature）：整型或浮点型，如用户活跃度或商品价格等。


#### 代码
```python
#导入模块
import numpy as np
import pandas as pd
import lightgbm as lgb
‘’‘
mean_squared_log_error：均方对数误差，用于评估预测值与真实值之间的对数差异平方的平均值。
mean_absolute_error：平均绝对误差，用于评估预测值与真实值之间的绝对差异的平均值。
mean_squared_error：均方误差，用于评估预测值与真实值之间的差异平方的平均值。
’‘’
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
# tqdm是一个用于显示进度条的库
import tqdm
# sys模块提供了一些函数和变量，用于与Python解释器进行交互。
import sys
# os模块提供了一些函数，用于与操作系统进行交互。
import os
# gc模块提供了一个接口来控制垃圾回收（Garbage Collection）。gc.collect()：强制进行垃圾回收，以释放内存。
import gc
# argparse模块用于解析命令行参数。它可以让你定义需要的参数，并自动生成帮助和使用信息。
import argparse
# warnings模块用于控制Python程序中的警告消息。
import warnings
# 通过warnings.filterwarnings('ignore')，你可以忽略所有警告消息，使它们不显示在控制台中。这在你希望保持输出简洁时特别有用。
warnings.filterwarnings('ignore')


# 读取训练数据和测试数据
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

# 不同type类型对应target的柱状图
import matplotlib.pyplot as plt
# 不同type类型对应target的柱状图
type_target_df = train.groupby('type')['target'].mean().reset_index()
‘’‘
.groupby('type')：按 'type' 列中的值对 DataFrame 进行分组。'type' 中的每个唯一值都会创建一个单独的组。
['target'].mean()：对由 groupby 创建的每个组，计算 'target' 值的均值。
.reset_index(): 这是为了重置索引，使 'type' 成为普通列而不是索引。
结果是一个新的 DataFrame type_target_df，其中包含两列：
'type'：原始 'type' 列中的唯一值。
'target'：每种 'type' 的 'target' 值的均值。
’‘’
plt.figure(figsize=(8, 4))
plt.bar(type_target_df['type'], type_target_df['target'], color=['blue', 'green'])
‘’‘
创建柱状图，其中：
type_target_df['type']：提供 x 轴的值（唯一的 'type' 值）。
type_target_df['target']：提供 y 轴的值（每种 'type' 的平均 'target' 值）。
color=['blue', 'green']：指定柱状图的颜色。如果 'type' 的种类超过两个，颜色会交替使用。
’‘’
plt.xlabel('Type')
plt.ylabel('Average Target Value')
plt.title('Bar Chart of Target by Type')
plt.show()
```
![不同type类型对应target的柱状图](/image/image2.png)

```python
# id为00037f39cf的按dt为序列关于target的折线图
# 得到一个只包含id为'00037f39cf'的行的DataFrame
specific_id_df = train[train['id'] == '00037f39cf']
plt.figure(figsize=(10, 5))
plt.plot(specific_id_df['dt'], specific_id_df['target'], marker='o', linestyle='-')
plt.xlabel('DateTime')
plt.ylabel('Target Value')
plt.title("Line Chart of Target for ID '00037f39cf'")
plt.show()
```

![id为00037f39cf的按dt为序列关于target的折线图](/image/image1.png)


#### 特征工程

- 历史平移特征：通过历史平移获取上个阶段的信息；如下图所示，可以将d-1时间的信息给到d时间，d时间信息给到d+1时间，这样就实现了平移一个单位的特征构建。
 ![历史平移特征](/image/image3.png)

- 窗口统计特征：窗口统计可以构建不同的窗口大小，然后基于窗口范围进统计均值、最大值、最小值、中位数、方差的信息，可以反映最近阶段数据的变化情况。如下图所示，可以将d时刻之前的三个时间单位的信息进行统计构建特征给d时刻。
 ![窗口统计特征](/image/image4.png)

```python
# 合并训练数据和测试数据，并进行排序
# axis=0按行方向合并，忽略原来的索引（ignore_index=True）
data = pd.concat([test, train], axis=0, ignore_index=True)
# 降序排列（ascending=False），并重置索引（reset_index(drop=True)）
data = data.sort_values(['id','dt'], ascending=False).reset_index(drop=True)

# 历史平移
for i in range(10,30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)
‘’‘
data.groupby(['id'])['target'].shift(i)：对每个id分组，然后将target列向后平移i个位置。
data[f'last{i}_target']：将平移后的结果存储在新的列中，列名格式为last{i}_target。
’‘’
    
# 窗口统计
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3

# 进行数据切分
train = data[data.target.notnull()].reset_index(drop=True)
test = data[data.target.isnull()].reset_index(drop=True)
‘’‘
data[data.target.notnull()]：从data中提取target不为空的行，生成新的训练数据train。
data[data.target.isnull()]：从data中提取target为空的行，生成新的测试数据test。
reset_index(drop=True)：重置索引，忽略原来的索引。
’‘’

# 确定输入特征
train_cols = [f for f in data.columns if f not in ['id','target']]
# 从data的列名中移除id和target，生成输入特征的列表train_cols。
```


#### 模型训练与测试集预测
这里选择使用Lightgbm模型，也是通常作为数据挖掘比赛的基线模型，在不需要过程调参的情况的也能得到比较稳定的分数。
另外需要注意的训练集和验证集的构建：因为数据存在时序关系，所以需要严格按照时序进行切分，
- 这里选择原始给出训练数据集中dt为30之后的数据作为训练数据，之前的数据作为验证数据，
- 这样保证了数据不存在穿越问题（不使用未来数据预测历史数据）。

```python
详细解释def time_model(lgb, train_df, test_df, cols):
    # 训练集和验证集切分
    trn_x, trn_y = train_df[train_df.dt>=31][cols], train_df[train_df.dt>=31]['target']
    val_x, val_y = train_df[train_df.dt<=30][cols], train_df[train_df.dt<=30]['target']
    # 构建模型输入数据
    train_matrix = lgb.Dataset(trn_x, label=trn_y)
    valid_matrix = lgb.Dataset(val_x, label=val_y)
    # lightgbm参数
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'min_child_weight': 5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.05,
        'seed': 2024,
        'nthread' : 16,
        'verbose' : -1,
    }
    # 训练模型
    model = lgb.train(lgb_params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix], 
                      categorical_feature=[], verbose_eval=500, early_stopping_rounds=500)
'''
使用 lgb.train 方法训练模型，传入定义的参数和数据集。
50000：最多训练 50000 轮
valid_sets：提供训练和验证集用于模型评估
categorical_feature：没有提供分类特征
verbose_eval=500：每 500 轮输出一次日志
early_stopping_rounds=500：在验证集上 500 轮没有提升时提前停止训练
'''
    # 验证集和测试集结果预测
    # num_iteration=model.best_iteration 确保使用最佳迭代次数。
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_df[cols], num_iteration=model.best_iteration)
    # 离线分数评估
    score = mean_squared_error(val_pred, val_y)
    print(score)
       
    return val_pred, test_pred
    
lgb_oof, lgb_test = time_model(lgb, train, test, train_cols)

# 保存结果文件到本地
test['target'] = lgb_test
test[['id','dt','target']].to_csv('submit.csv', index=None)
```

若安装的lightgbm版本为4.4.0，则上段代码修改为

```python
def time_model(lgb, train_df, test_df, cols):
    # 训练集和验证集切分
    trn_x, trn_y = train_df[train_df.dt>=31][cols], train_df[train_df.dt>=31]['target']
    val_x, val_y = train_df[train_df.dt<=30][cols], train_df[train_df.dt<=30]['target']
    
    # 构建模型输入数据
    train_matrix = lgb.Dataset(trn_x, label=trn_y)
    valid_matrix = lgb.Dataset(val_x, label=val_y)
    
    # LightGBM 参数
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'min_child_weight': 5,
        'num_leaves': 32,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.05,
        'seed': 2024,
        'nthread': 16,
        'verbose': -1,
        'early_stopping_round': 500,
        'verbose_eval': 500
    }
    
    # 训练模型
    model = lgb.train(
        lgb_params,
        train_matrix,
        num_boost_round=50000,
        valid_sets=[train_matrix, valid_matrix]
    )
    
    # 验证集和测试集结果预测
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_df[cols], num_iteration=model.best_iteration)
    
    # 离线分数评估
    score = mean_squared_error(val_pred, val_y)
    print(score)
    
    return val_pred, test_pred

# 示例函数调用
lgb_oof, lgb_test = time_model(lgb, train, test, train_cols)

# 保存结果文件到本地
test['target'] = lgb_test
test[['id', 'dt', 'target']].to_csv('submit.csv', index=None)
```
