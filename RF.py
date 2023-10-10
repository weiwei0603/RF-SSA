import os
os.system('cls' if os.name == 'nt' else 'clear')
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as scio
import warnings
from sklearn.ensemble import RandomForestRegressor
import SSA as SSA
warnings.filterwarnings("ignore")
#定义适应函数，以测试集和训练集的绝对误差和为适应度值
def fun(X):
    #训练随机森林分类器
    N_estimators = int(X[0]) #随机森林个数
    Max_features = int(X[1]) #最大特征数
    Model=RandomForestRegressor(n_estimators=N_estimators,max_features=Max_features, max_depth=None,min_samples_split=2, bootstrap=True,random_state=0)
    Model.fit(P_train,T_train)
    PredictTrain=Model.predict(P_train)
    PredictTest=Model.predict(P_test)
    MSETrain= np.sqrt(np.sum((PredictTrain - T_train[:,0])**2))/T_train.size#计算MSE
    MSETest=np.sqrt(np.sum((PredictTest - T_test[:,0])**2))/T_test.size#计算MSE
    output = MSETrain+MSETest
    return output
#读取数据,输入数据为1维的数据，输出数据为1维的数据
path='dataaa.mat'
data = scio.loadmat(path)
# 特征数据和标签数据
features = data['features']
labels = data['labels']
# 划分数据为训练集和测试集
P_train = features[:540, :]
T_train = labels[:540]
P_test = features[540:, :]
T_test = labels[540:]
n_features = P_train.shape[1]  
#设置麻雀参数
pop = 10 #种群数量
MaxIter = 30 #最大迭代次数
dim = 2 #维度
lb = np.matrix([[1],[1]]) #下边界
ub = np.matrix([[20],[n_features]])#上边界
fobj = fun
GbestScore,GbestPositon,Curve = SSA.SSA(pop,dim,lb,ub,MaxIter,fobj) 
print('最优适应度值：',GbestScore)
print('N_estimators最优解：',int(GbestPositon[0,0]))
print('Max_features最优解：',int(GbestPositon[0,1]))
#利用最终优化的结果计算分类正确率等信息
#利用最优参数训练随机森林
#利用最终优化的结果计算分类正确率等信息
#利用最优参数训练随机森林
N_estimators = int(GbestPositon[0,0]) #随机森林个数
Max_features = int(GbestPositon[0,1]) #最大特征数
n_features= P_train.shape[1]#特征数
ModelSSA=RandomForestRegressor(n_estimators=N_estimators,max_features=Max_features, max_depth=None,min_samples_split=2, bootstrap=True,random_state=0)
ModelSSA.fit(P_train,T_train)
PredictTrainSSA=ModelSSA.predict(P_train)
PredictTestSSA=ModelSSA.predict(P_test)
MSETrainSSA= np.sqrt(np.sum((PredictTrainSSA - T_train[:,0])**2))/T_train.size#计算RMSE
MSETestSSA=np.sqrt(np.sum((PredictTestSSA - T_test[:,0])**2))/T_test.size#计算RMSE
y_meantrainSSA = np.mean(T_train[:,0])
R2TrainSSA=1-np.sum((PredictTrainSSA - T_train[:,0])**2)/np.sum((y_meantrainSSA - T_train[:,0])**2)#计算R2
y_meantestSSA = np.mean(T_test[:,0])
R2TestSSA=1-np.sum((PredictTestSSA - T_test[:,0])**2)/np.sum((y_meantestSSA - T_test[:,0])**2)#计算R2
MAPETrainSSA = np.sum(np.abs((PredictTrainSSA - T_train[:, 0]) / T_train[:, 0])) / T_train.shape[0] * 100#计算MAPE
MAPETestSSA=np.sum(np.abs((PredictTestSSA - T_test[:,0])/T_test[:,0]))/T_test.shape[0] * 100#计算MAPE
accuracyTrainSSA=(1-(np.sum(np.abs(PredictTrainSSA - T_train[:,0]))/np.sum(T_train[:,0])))*100
accuracyTestSSA=(1-(np.sum(np.abs(PredictTestSSA - T_test[:,0]))/np.sum(T_test[:,0])))*100
print("RF-SSA训练集MSE：" +str(MSETrainSSA) )
print("RF-SSA测试集MSE："+str(MSETestSSA) )
print("RF-SSA总MSE："+str(MSETestSSA+MSETrainSSA) )
print("RF-SSA训练集R2：" +str(R2TrainSSA) )
print("RF-SSA测试集R2："+str(R2TestSSA) )
print("RF-SSA训练集MAPE：" +str(MAPETrainSSA) )
print("RF-SSA测试集MAPE："+str(MAPETestSSA) )
print("RF-SSA训练集预测精度accuracy："+str(accuracyTrainSSA) )
print("RF-SSA测试集预测精度accuracy："+str(accuracyTestSSA) )
# 获取特征重要性
feature_importances = ModelSSA.feature_importances_
# 将特征重要性与特征索引对应
feature_indices = np.argsort(feature_importances)[::-1]
# 输出特征重要性排序
print("特征重要性排序：")
for i, idx in enumerate(feature_indices):
    print(f"特征 {i + 1}: 特征索引 {idx}, 重要性 {feature_importances[idx]}")
# 输出特征重要性排序
print("特征重要性排序：")
for i in range(len(feature_importances)):
    print(f"特征 {i + 1} (原始索引 {i}): 重要性 {feature_importances[i]}")
# 将训练数据集结果保存为 CSV 文件
train_results = np.column_stack((T_train, PredictTrainSSA, PredictTrain))
np.savetxt('train_results.csv', train_results, delimiter=',', header='True value, SSA-RF_results, RF_results', comments='')
np.savetxt('feature_importance.csv', feature_importances, delimiter=',', header='Feature Importance', comments='')
# 将测试数据集结果保存为 CSV 文件
test_results = np.column_stack((T_test, PredictTestSSA, PredictTest))
np.savetxt('test_results.csv', test_results, delimiter=',', header='True value, SSA-RF_results, RF_results', comments='')
#绘制适应度曲线
plt.figure(1)
plt.plot(Curve,'r-',linewidth=2)
plt.xlabel('Iteration',fontsize='medium')
plt.ylabel("Fitness",fontsize='medium')
plt.grid()
plt.title('SSA-RF',fontsize='large')
plt.show()
#绘制训练集结果
plt.figure(2)
plt.plot(T_train,'ro-',linewidth=1,label='True value')
plt.plot(PredictTrainSSA,'b*-',linewidth=1,label='SSA-RF_results')
plt.plot(PredictTrain,'g.-',linewidth=1,label='RF_results')
plt.xlabel('index',fontsize='medium')
plt.ylabel("type",fontsize='medium')
plt.grid()
plt.title('TrainDataSet Result',fontsize='large')
plt.legend()
plt.show()
#绘制测试集结果
plt.figure(3)
plt.plot(T_test,'ro-',linewidth=1,label='True value ')
plt.plot(PredictTestSSA,'b*-',linewidth=1,label='SSA-RF_results')
plt.plot(PredictTest,'g.-',linewidth=1,label='RF_results')
plt.xlabel('index',fontsize='medium')
plt.ylabel("type",fontsize='medium')
plt.grid()
plt.title('TestDataSet Result',fontsize='large')
plt.legend()
plt.show()
# 绘制特征重要性图
plt.figure(figsize=(8, 6))
plt.bar(range(n_features), feature_importances[feature_indices], align="center")
plt.xticks(range(n_features), feature_indices)
plt.xlabel("特征索引")
plt.ylabel("特征重要性")
plt.title("特征重要性排序")
plt.show()




