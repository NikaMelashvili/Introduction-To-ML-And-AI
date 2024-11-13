from sklearn import preprocessing
import pandas as pd
from sklearn.naive_bayes import GaussianNB

weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny','Rainy','Sunny','Overcast','Overcast','Rainy']
#temp=['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play=['No','No','Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']


#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
wheather_encoded=le.fit_transform(weather)
print ('wheather_encoded=')
print (wheather_encoded)

# Converting string labels into numbers

#temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

print(label)
#print ("Temp:",temp_encoded)
print ("Play:",label)

#df_features = pd.DataFrame(columns=['weather', 'Temperature'])
df_features = pd.DataFrame(columns=['weather'])
df_features['weather']=wheather_encoded
#df_features['Temperature']=temp_encoded
print('df_features=')
print(df_features)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(df_features,label)


new_df=df_features
new_df['wstring']=weather
print(new_df)

df=new_df[new_df.wstring=='Overcast']
print(df)
numpy_df=df.to_numpy()
print(numpy_df)
print('encoded number=')
print(numpy_df[0,0])
#predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
predicted= model.predict([[numpy_df[0,0]]]) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)
print ("Predicted Value:",le.inverse_transform(predicted))
print ("Predicted Value:", model.predict_proba([[0]]))


##############################################################
label=le.fit(weather)
Overcast_label=le.transform(['Overcast'])
print(Overcast_label)
predicted= model.predict([Overcast_label])
print ("Predicted Value:", predicted)
print ("Predicted Value:",le.inverse_transform(predicted))
print ("Predicted Value:", model.predict_proba([[0]]))


#######################################
# Assigning features and label variables
weather=['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny','Sunny', 'Rainy','Sunny','Overcast','Overcast','Rainy']
temp =  ['Hot'  ,'Hot'  ,'Hot'     ,'Mild' ,'Cool' ,'Cool' ,'Cool'     ,'Mild','Cool','Mild','Mild','Mild','Hot','Mild']
play =  ['No',   'No',   'Yes','Yes','Yes','No','Yes','No','Yes','Yes','Yes','Yes','Yes','No']

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
weather_encoded=le.fit_transform(weather)
print("whether_encoded:")
print("whether:", weather_encoded)

# Converting string labels into numbers
temp_encoded=le.fit_transform(temp)
label=le.fit_transform(play)

print( "Temp:",temp_encoded)
print ("Play:",label)

#Combinig weather and temp into single list of tuples
features=list(zip(weather_encoded,temp_encoded))
print ("features: ", features)

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features,label)

#Predict Output
predicted= model.predict([[0,2]]) # 0:Overcast, 2:Mild
print ("Predicted Value:", predicted)

print(model.predict_proba([[0,2]]))
