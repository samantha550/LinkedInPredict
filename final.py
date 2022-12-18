import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import altair as alt
import streamlit as st
from PIL import Image



s = pd.read_csv('social_media_usage.csv')
s.ndim

def clean_sm(x):
    x = np.where (x == 1, 1,0)
    return x

ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"]>9, np.nan, s["income"]),
    "education": np.where(s["educ2"]>8, np.nan, s["educ2"]),
    "parent": clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female": np.where(s["gender"] == 2, 1,0),
    "age": np.where(s["age"] > 98, np.nan, s["age"])
    
})

ss = ss.dropna()

y = ss["sm_li"]
x = ss[["age", "education", "parent", "female","income","married"]]

x_train, x_test, y_train, y_test = train_test_split(
    x,y,
    stratify = y,
    test_size = 0.2,
    random_state = 987
)

lr = LogisticRegression(class_weight = 'balanced')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)


st.markdown(f"## :sparkles: Welcome :sparkles: ")
st.markdown(f"### The purpose of this app is to predict the chances of you being a LinkedIn User")
st.markdown(f"#### To get started please enter your information below:")
age = st.slider(label="Enter your age",  min_value=1, max_value=100,value=50)
inc = st.selectbox(f"Income Range", options = ["Less than $10,000", "10 to under $20,000","20 to under $30,000","30 to under $40,000","40 to under $50,000","50 to under $75,000"," 75 to under $100,000","100 to under $150,000","$150,000 or more"] )
deg= st.selectbox(f"Degree Level", options = ["Did not go to High School","High school incomplete","High school graduate or GED","Some college, no degree","Two-year degree - Associates","Four-year degree - Bachelor","Some postgraduate or professional schooling","Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD))"] )
mar = st.number_input("Married", "no", "yes")
par = st.number_input("Parent (0=no, 1=yes)", 0, 1)
gen = st.number_input("Gender (0=Male, 1=Female)", 0, 1)



if inc == "Less than $10,000":
     inc = 1
elif inc == "10 to under $20,000":
     inc = 2
elif inc =="20 to under $30,000":
    inc = 3
elif inc == "30 to under $40,000":
    inc = 4
elif inc == "40 to under $50,000":
    inc = 5
elif inc == "50 to under $75,000":
    inc = 6
elif inc == " 75 to under $100,000":
    inc = 7
elif inc == "100 to under $150,000":
    inc = 8
else:
    inc = 9

    
if deg == "Did not go to High School":
     deg = 1
elif deg == "High school incomplete":
     deg = 2
elif deg =="High school graduate or GED":
    deg = 3
elif deg == "Some college, no degree":
    deg = 4
elif deg == "Two-year degree - Associates":
    deg = 5
elif deg == "Four-year degree - Bachelor":
    deg = 6
elif deg == "Some postgraduate or professional schooling":
    deg = 7
else:
    deg = 8

if mar == "yes":
    mar = 1
else:
    mar=0
    
 
new_guess = pd.DataFrame({
    "age": age,
    "education": [deg],
    "parent": [par],
     "female": [gen],
    "income": [inc],
    "married": [mar],
   
    })

# Predict class, given input features
user_pred = lr.predict(new_guess)

probs = lr.predict_proba(new_guess)


st.write(f"The chances of you being a LinkedIn User are : {probs[0][1]}")

# prediction 
if  user_pred== 1:
    user_label = "a LinkedIn User :star2:"
else:
    user_label = "not a LinkedIn User"
    
if st.checkbox(f" :arrow_left: Click the check box to see our prediciton!"):
    st.write(f"## We predict you are: ### \n {user_label}")


