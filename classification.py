import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(r"D:\abdullah\abbbb python\aapandas\classification\titanic.csv")
df.drop_duplicates(inplace=True)

# ploting heatmap to see null values in the dataset
sns.heatmap(df.isnull() , cbar=False , yticklabels=False , cmap="cool")
# sns.boxplot(x="pclass",y="age",data=df)
plt.show()


# auto fill age values on the basis of median according to "pclass"
def imputation(cols):
    age = cols[0]
    pclass = cols[1]
    if pd.isnull(age):
        if pclass==1:
            return 37
        elif pclass==2:
            return 29
        else:
            return 23
    else:
        return age

df["age"] = df[["age","pclass"]].apply(imputation , axis=1)


# drop un-necessary rows
df.drop(["who","sibsp","parch","embarked","class","adult_male","deck","embark_town","alone","alive"],axis=1,inplace=True)


# now check heatmap after cleaning the data
sns.heatmap(df.isnull() , cbar=False , yticklabels=False , cmap="cool")
plt.show()


# changing float type into integer
o = ["age","fare"]
for item in o:
    df[item] = df[item].astype("int")


# set sex colums into numeric
df["sex"].replace({"male":"1","female":"0"},inplace=True)


# now separate target variable
x = df.drop("survived",axis=1)
y = df["survived"]


# splitting data into four parts
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.20,random_state=100)

# This model is Logistic Regression and showing accuracy : 71%
logistic = LogisticRegression()
logistic.fit(x_train,y_train)
predict = logistic.predict(x_test)
accuracy = round(accuracy_score(y_test , predict)*100,3)
print("Logistic Regression : ",accuracy,"%")

# This model is Random Forest Classifier and showing accuracy : 72%
random = RandomForestClassifier()
random.fit(x_train,y_train)
predict2 = random.predict(x_test)
accuracy2 = round(accuracy_score(y_test,predict2)*100,2)
print("Random Forest Classifier Accuracy",accuracy2,"%")

# This model is Decision Tree Classifier and showing accuracy : 70%
tree = DecisionTreeClassifier()
tree.fit(x_train,y_train)
predict3 = tree.predict(x_test)
accuracy3 = round(accuracy_score(y_test,predict3)*100,2)
print("Decision tree Classifier Accuracy",accuracy3,"%")

# This model is Gradient Boosting Classifier and showing accuracy : 74%
gradient = GradientBoostingClassifier()
gradient.fit(x_train,y_train)
predict4 = gradient.predict(x_test)
accuracy4 = round(accuracy_score(y_test,predict4)*100,2)
print("Gradient Boosting Classifier Accuracy : ",accuracy4,"%")


# Best model is of Gradient Boosting Classifier because it shows high accuracy during testing which is 74% than other models