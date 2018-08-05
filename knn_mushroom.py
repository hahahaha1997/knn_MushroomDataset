import operator
from numpy import *
def LoadData(filename):
    with open(filename,'r')as file:
        dataset=[line.strip().split(' ') for line in file.readlines()]
    dataset=[[int(column) if column.isdigit() else column for column in row ]for row in dataset]#将数据集中所有的字符数据转换为数字
    length = len(dataset)

    splitLoc = (int)(length / 5 * 4)
    TrainData=dataset[0:splitLoc]#数据集中的前五分之四作为训练集
    TestData=dataset[splitLoc+1:-1]#数据集的后五分之一作为测试集

    Trainlabel = [row[0] for row in TrainData]#获取第一行的标签
    TrainData = [row[1:-1] for row in TrainData]#取出第一行的标签
    Testlabel = [row[0]for row in TestData]
    TestData = [row[1:-1]for row in TestData]#同上

    return array(TrainData),array(TestData),Trainlabel,Testlabel#借助Numpy.array类型来实现数据集的操作简单化

def knn(k,TrainData,TestData,TrainLabel,TestLabel):
    num = 0#记录当前这是测试第几个TestVec
    true = 0#记录标签正确的个数

    DataNum = TrainData.shape[0]
    for TestVec in TestData:

        diff = tile(TestVec,(DataNum,1))-TrainData
        diffMat = diff ** 2
        distance = sqrt(abs(diffMat.sum(axis=1)))#计算矩阵的距离，不通过点对点计算，而是通过向量计算，axis=1指示这是一个行向量

        distance_sort = distance.argsort()#对计算过的，当前的测试vector和TrainData每个向量的距离进行排序

        count={}#一个统计正确与否的字典，N个键值对，N是数据集中的分类个数，键值对为   分类-分类的统计

        for i in range(k):#对于已经排序的距离的前k个值，统计每种分类的个数
            label = TrainLabel[distance_sort[i]]
            count[label] = count.get(label,0)+1

        count_sort = sorted(Trans(count),key = operator.itemgetter(1),reverse=True)#对统计的分类的个数按照键值对的value值来进行排序

        if count_sort[0][0] ==TestLabel[num]:#如果最多的那个分类和当前测试的向量的分类相同，即正确的进行了一次分类，则统计数据true加一
            true = true +1
        num+=1

    #计算分类的正确率
    print("K:" + str(k))
    print("正确率：" + str(true / len(TestData)))

def Trans(dic:dict):
    keys = dic.keys()
    values = dic.values()
    lst = [(key, value) for key, value in zip(keys, values)]
    return lst

TrainData,TestData,TrainLabel,TestLabel = LoadData(r'C:\Users\yang\Desktop\mushroom.dat')
k=[1,2,5,10]
for item in k:#测试k=1,2,5,10的情况
    knn(item,TrainData,TestData,TrainLabel,TestLabel)