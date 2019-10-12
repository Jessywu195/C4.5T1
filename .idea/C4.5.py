from math import log
import operator
#创建数据集
def createDataSet():
    '''
    Create a data set
    :return:
    '''
    #特征标签
    featureLabels = ['weather','humidity','israin']
    #数据集
    dataSet = [
            ['sunny','70','yes','play'],
            ['sunny','90','yes','ntpl'],
            ['sunny','85','no','ntpl'],
            ['sunny','95','no','ntpl'],
            ['sunny','70','no','play'],
            ['cloudy','78','no','play'],
            ['cloudy','65','yes','play'],
            ['cloudy','75','no','play'],
            ['rainy','80','yes','ntpl'],
            ['rainy','70','yes','ntpl'],
            ['rainy','80','no','play'],
            ['rainy','80','no','play'],
            ['rainy','96','no','play']
    ]
    return dataSet,featureLabels


# 计算分类标签的信息熵
def calShannonEnt(dataSet):
    '''
    :param dataSet:
    :return:Entropy value of each feature
    function:Calculating Shannon Entropy
    '''
    # 为分类类目（是否去玩）创建字典，统计两类别数量
    labelCounts = {}
    vectorNum = len(dataSet)  # 计算向量数量
    for featureVector in dataSet:
        currentLabel = featureVector[-1]  # 获取分类
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:
        p = float(labelCounts[key] / vectorNum)   # 计算概率
        shannonEnt -= p * log(p, 2)    # 计算类别标签的香农熵
    return shannonEnt


# 根据特征标签的不同值划分数据集
def splitDataSet(dataSet, featureIndex, featureValue):
    '''
    :param dataSet:
    :param featureIndex:
    :param featureValue:
    :return: subDataSet
    '''
    subDataSet = []
    for vector in dataSet:     # 取数据
        if vector[featureIndex] == featureValue:   # 划分特征标签值等于value的向量
            subDataSetVector = vector[:featureIndex]
            subDataSetVector.extend(vector[featureIndex+1:])
            subDataSet.append(subDataSetVector)
    return subDataSet


# 计算最大信息增益率，按最大信息增益率划分数据集
def chooseBestFeatureToSplit(dataSet):
    '''
    :param dataSet:
    :return: best splitting feature
    function:Calculate the maximum information gain rate and
    divide the data set according to the maximum information gain rate.
    '''
    numOfFeature = len(dataSet[0]) - 1 # 特征个数（减去类别）
    baseEnt = calShannonEnt(dataSet)   # 类别的信息熵H(D)
    bestInfoGainRate = 0.0             # 最大信息增益率
    bestFeatureIndex = -1              # 使信息增益率最大的特征标签索引，设为负数

    for featureIndex in range(numOfFeature): # 遍历特征标签
        featureValueCol = [vector[featureIndex] for vector in dataSet]   # 得到某个特征值的一整列数据
        uniqueFeatureValues = set(featureValueCol)   # 获取特征标签的不同值，set：集合
        conditionEnt = 0.0
        splitInfo = 0.0
        for featureValue in uniqueFeatureValues:   # 取特征的值
           subDataSet = splitDataSet(dataSet,featureIndex,featureValue)  # 划分数据集
           p = len(subDataSet) / float(len(dataSet))        # 计算value的概率
           conditionEnt += p * calShannonEnt(subDataSet)    # 条件熵H(X|Y)
           splitInfo += -p * log(p,2)                       # 属性的信息熵H(X)
           infoGain = baseEnt - conditionEnt    # 信息增益H(D)-H(X|Y)
           if splitInfo == 0:
               splitInfo = 1
        infoGainRate = infoGain / splitInfo # 信息增益率
        if infoGainRate > bestInfoGainRate:
            bestInfoGainRate = infoGainRate
            bestFeatureIndex = featureIndex
    return bestFeatureIndex


# 分类不纯时，统计特征标签的分类值的最大次数
def majorCount(classValues):
    '''
    :param classValues:类别标签的值
    :return:
    当特征标签已经都被用完了，分类任然不纯，
    则用概率最大（次数最多）的classValue作为判定的类别
    '''
    classCount = {}
    for classValue in classValues:
        if classValue not in classCount.keys():
            classCount[classValue] = 0
        else:
            classCount[classValue] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


# 构建决策树
def createTree(dataSet,featureLabels):
    '''
    :param dataSet:classValues
    :param featureLabels:
    :return:
    function:Create a decision tree
    '''
    classValues = [vector[-1] for vector in dataSet]   # 取最后一列的数据（分类标签的值）
    if classValues.count(classValues[0]) == len(classValues): # 只有一种类别，则返回当前类别
        return classValues[0]
    if len(dataSet[0]) == 1:    # 已使用完全部feature任无法决策时，返回次数最多的类别
        return majorCount(classValues)

    # 按照信息增益率最高选取分类特征属性
    bestFeatureIndex = chooseBestFeatureToSplit(dataSet)    # 返回分类特征的数组索引
    bestFeatureLabel = featureLabels[bestFeatureIndex]             # 取出bestFeatureLabel

    print(bestFeatureLabel)
    Tree = {bestFeatureLabel:{}}    # 构建树的字典
    del(featureLabels[bestFeatureIndex])   # 从特征标签中删除该属性
    featureValues = [vector[bestFeatureIndex] for vector in dataSet]    # 取出每个bestFeature的所有值
    uniqueFeatureValues = set(featureValues)    # 使每个特征标签的值唯一
    for featureValue in uniqueFeatureValues:
        subLabels = featureLabels[:]
        # 构建数据的自己和，并进行递归
        subDataSet = splitDataSet(dataSet, bestFeatureIndex, featureValue)
        Tree[bestFeatureLabel][featureValue] = createTree(subDataSet, subLabels)
    return Tree


def main():
    dataSet,featureLabels = createDataSet()
    labelsTmp = featureLabels[:]  # 拷贝，createTree会改变labels
    desicionTree = createTree(dataSet, labelsTmp)
    print('desicionTree:\n', desicionTree)


if __name__ == '__main__':
    main()