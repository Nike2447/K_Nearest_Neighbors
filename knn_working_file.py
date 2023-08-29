import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")

# preprocessing each label to fit a numeric value equivalent to them for example very good is mapped as 2 and bad as 0.
pre = preprocessing.LabelEncoder()
buying = pre.fit_transform(list(data["buying"]))
maint = pre.fit_transform(list(data["maint"]))
door = pre.fit_transform(list(data["door"]))
persons = pre.fit_transform(list(data["persons"]))
lug_boot = pre.fit_transform(list(data["lug_boot"]))
safety = pre.fit_transform(list(data["safety"]))
cls = pre.fit_transform(list(data["class"]))

predict = "class"

# making a master list containg all the values except the first column.
x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

#splitting the data into test and train data with test of sample size 10% of total.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y,test_size=0.1)

# running the knn classifier algorithm with value of k set as 9.
model = KNeighborsClassifier(n_neighbors=9)
#Fitting the training data to the model 
model.fit(x_train,y_train)

#Predicting model score using the test variables
accuracy = model.score(x_test,y_test)
print("Accuracy : ",round(accuracy*100,2),"%")

#storing the predicted values of the test in a variable
predicted = model.predict(x_test)
names = ["unacc","acc","good","vgood"]

#Uncomment below to see each predicted and actual value of the model 
#for i in range(len(predicted)):
#    print("Predicted: ",names[predicted[i]]," Actual: ",names[y_test[i]])