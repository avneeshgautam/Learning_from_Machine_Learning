# import operator
# from collections import Counter

# class KNN:
#     def __init__(self, k):
#         self.k = k
    
#     # use to train the model
#     def fit(self,X_train,y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#         print("Trainind Done")
    

#     def predict(self,X_test):

#         distance={}
#         counter=1

#         for i in self.X_train:
#             distance[counter]=((X_test[0][0]-i[0])**2 + (X_test[0][1]-i[1])**2)**1/2
#             counter=counter+1
#         distance=sorted(distance.items(), key=operator.itemgetter(1))



 # def predict(self):
    #     X_test = self.X_train
    #     print("sdsdds")
    #     # we will find distance between X_train(single value) and X_train
    #     distance = {}
    #     counter = 1

    #     for i in self.X_train:
    #         # eculidian distance
    #         val1 = X_test[0][0] - i[0]
    #         val2 = X_test[0][1] - i[1] 

    #         distance[counter] = np.sprt(np.sqr(val1) + np.sqr(val2))
    #         counter +=1

    #     print(distance)



import operator
from collections import Counter

class KNN:
    def __init__(self,k):
        self.k=k

    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        print("Training Done")

    def prediction(self,X_test):
        # self.X_test = X_test
        print("working")
        distance={}
        counter=1

        for i in self.X_train:
            distance[counter]=((X_test[0][0]-i[0])**2 + (X_test[0][1]-i[1])**2)**1/2
            counter=counter+1
        distance=sorted(distance.items(), key=operator.itemgetter(1))
    
        self.classify(distance=distance[:self.k])

    def classify(self,distance):
        label=[]

        for i in distance:
            label.append(self.y_train[i[0]])

        return Counter(label).most_common()[0][0]
    
    # def testing(val):
    #     print("Testing",val)
