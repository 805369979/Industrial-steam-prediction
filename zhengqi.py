import numpy as np
from sklearn.decomposition import PCA
from sklearn import datasets,linear_model
def Load_Exdata(filename):
    data = []
    with open(filename, 'r') as f:
        count = 0
        for line in f.readlines():
            count += 1
            if count ==1:
                continue
            line = line.split('\t')
            #current = [item for item in line]
            data.append(line)
    return data

data = Load_Exdata('zhengqi_train.txt')
data = np.array(data,dtype = np.float64)
X = data[:,:-1]
Y = data[:,-1]
#27 0.1421
#20 0.1465
#25
#26
#pca提取数据
pca = PCA(n_components= 26)
pca.fit(X)
ccc = []
sum = 0
for i in range(len(pca.explained_variance_ratio_)):
    sum = sum + pca.explained_variance_ratio_[i]
    ccc.append(sum)
X_pca = pca.transform(X)


#训练数据
regr = linear_model.LinearRegression()
regr.fit(X_pca,Y)

#预测
test = Load_Exdata('zhengqi_test.txt')
test = np.array(test,dtype=np.float64)
test_pca = pca.transform(test)

y_test = regr.predict(test_pca)
np.savetxt("result.txt", y_test)
print(y_test)