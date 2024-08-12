
import operator
from collections import Counter

class KNN:
    def __init__(self,k):
        self.k=k

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training Done")

    def predict(self,X_test):
        # self.X_test = X_test
        print("Predicting")
        distance={}
        counter=1

        for i in self.X_train:
            distance[counter]=((X_test[0][0]-i[0])**2 + (X_test[0][1]-i[1])**2)**1/2
            counter=counter+1
        distance=sorted(distance.items(), key=operator.itemgetter(1))
      
        return self.classify(distance=distance[:self.k])   # passinf first k point

    def classify(self,distance):
        label=[]

        for i in distance:
            label.append(self.y_train[i[0]])
        print(label)
        return( Counter(label).most_common()[0][0])
