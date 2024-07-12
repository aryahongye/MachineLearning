## 电力需求预测挑战赛(讯飞）2024.7.1——8.15

[赛事链接](https://challenge.xfyun.cn/h5/detail?type=electricity-demand&ch=dw24_uGS8Gs)


### 【训练时序预测模型助力电力需求预测】

电力需求的准确预测对于电网的稳定运行、能源的有效管理以及可再生能源的整合至关重要。


### 赛题任务
给定多个房屋对应电力消耗历史N天的相关序列数据等信息，预测房屋对应电力的消耗。


### 赛题数据简介
赛题数据由训练集和测试集组成，为了保证比赛的公平性，将每日日期进行脱敏，用1-N进行标识。

即1为数据集最近一天，其中1-10为测试集数据。


数据集由字段id（房屋id）、 dt（日标识）、type（房屋类型）、target（实际电力消耗）组成。


### 代码

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

#### 代码分解
```markdown
{
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
}
```

```python
# 4. 将target_mean作为测试集结果进行合并
test = test.merge(target_mean, on=['id'], how='left')

# 5. 保存结果文件到本地
test[['id','dt','target']].to_csv('submit.csv', index=None)
```

#### 代码分解

```markdown
{
目标：将 target_mean 与 test 数据集合并，使得 test 数据集包含每个 id 对应的目标均值 target。

merge 方法：
on=['id']：指定以 id 列作为键进行合并，即以 id 列为基础，将 target_mean 中的 target 列合并到 test 中。
how='left'：指定使用左连接（left join）进行合并。这意味着所有来自 test DataFrame 的行都将保留，如果在 target_mean 中找不到匹配的 id，则对应的 target 值将为 NaN。
}
```
