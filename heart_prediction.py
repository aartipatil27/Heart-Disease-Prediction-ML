from pyexpat import model

import pandas as pd

df = pd.read_csv("heart.csv")

print(df.head())
print(df.info())


# check target column balanced or inbalanced 
print(df["target"].value_counts())

#featue (x) and target (y) dividing
x= df.drop("target",axis=1)
y=df["target"]

#checking values
print(x.shape)
print(y.shape)


#train_test split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(
x,y, test_size=0.2, random_state=42)


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# model import # train logistic regression model

from sklearn .linear_model import LogisticRegression
model=LogisticRegression(max_iter=5000)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))



# #create decison tree model

from sklearn.tree import  DecisionTreeClassifier
dt_model= DecisionTreeClassifier(random_state=42)

#train
dt_model.fit(x_train,y_train)
dt_pred=dt_model.predict(x_test)


#accuracy
from sklearn.metrics import accuracy_score
dt_accuracy= accuracy_score(y_test,dt_pred)
print("decision tree accuracy:",dt_accuracy)


##confusion matrixx
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,dt_pred))


##classisfication report
from sklearn. metrics import classification_report
print(classification_report(y_test,dt_pred))



 ###random forest create model
from sklearn .ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
     n_estimators=100,
     random_state=42
)


#train

rf_model.fit(x_train,y_train)


##predition

rf_pred =rf_model.predict(x_test)


# # # ##accuracy


from  sklearn. metrics import accuracy_score
rf_accuracy = accuracy_score(y_test,rf_pred)
print("randome forest accuracy:",rf_accuracy)


# # # ##confusion mateix

from sklearn. metrics import confusion_matrix
print(confusion_matrix(y_test,rf_pred))


# # # ##classification report

from sklearn.metrics import classification_report
print (classification_report(y_test,rf_pred))



# ## Adding new patient details


new_patient =[[52,1,2,130,245,0,1,165,0,1.0,2,0,2]]

prediction = model.predict(new_patient)

if prediction[0] ==1:
 print ("heart disease detected")

else:

 print("no heart disease")



# # ##probability
probability = model.predict_proba(new_patient)

print("Probability:", probability)



##graphs

# ##target distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
df["target"].value_counts().plot(kind="bar")
plt.title("Heart Disease Distribution")
plt.xlabel("Target (0 = No Disease, 1 = Disease)")
plt.ylabel("Count")
plt.show()


# # ##correlation heatmapp
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()


# import matplotlib.pyplot as plt

# # Model names
models = ["Logistic Regression", "Decision Tree", "Random Forest"]

# # Accuracy values
accuracy = [88.52, 75.41, 83.61]

# # Create bar chart
plt.figure(figsize=(8, 5))
plt.bar(models, accuracy)

# Add title and labels
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy (%)")

# Set y-axis limit
plt.ylim(0, 100)

# Display accuracy values on top of bars
for i, value in enumerate(accuracy):
    plt.text(i, value + 1, f"{value:.2f}%", ha='center')

plt.show()