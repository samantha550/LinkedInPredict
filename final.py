import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import altair as alt
import streamlit as st
from PIL import Image



s = pd.read_csv('/Users/samsa/OneDrive/Desktop/Programming II/social_media_usage.csv')
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
st.markdown(f"### The purpose of this website is to predict the chances of you being a LinkedIn User")
st.markdown(f"#### To get started please enter your information below:")
age = st.slider(label="Enter your age",  min_value=1, max_value=100,value=50)
inc = st.selectbox(f"Income Range - see below for description of options", options = [1,2,3,4,5,6,7,8,9] )
st.text(f"(1 = Less than $10,000 | "
" 2 = 10 to under $20,000 | "
" 3 = 20 to under $30,000 |"
"4 = 30 to under $40,000 | "
" 5 = 40 to under $50,000 | "
" 6 = 50 to under $75,000 |"
"7 = 75 to under $100,000 | "
" 8 = 100 to under $150,000 | "
" 9 = $150,000 or more)")
#st.number_input("Income (low=1 to high=9)", 1, 9)
deg= st.selectbox(f"Degree Level - see below for description of options", options = [1,2,3,4,5,6,7,8] )
st.text(f"(1 = Did not go to High School |"
"2 = High school incomplete | "
"3 = High school graduate or GED |"
"4 = Some college, no degree | "
"5 = Two-year degree - Associates |"
"6 = Four-year degree - Bachelor |"
"7 = Some postgraduate or professional schooling |"
"8 = Postgraduate or professional degree (e.g., MA, MS, PhD, MD, JD))")
#deg = st.number_input("College degree (no=0 to yes=1)", 1, 9)
mar = st.number_input("Married (0=no, 1=yes)", 0, 1)
par = st.number_input("Parent (0=no, 1=yes)", 0, 1)
gen = st.number_input("Gender (0=Male, 1=Female)", 0, 1)

#st.write(f"#### Based on your responses as a {age} year old {gen_label}; who is {mar_label},{par_label}, a {deg_label}, and in a {inc_label} bracket: ")

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

# Generate probability of positive class (=1)

probs = lr.predict_proba(new_guess)

#user_guess = lr.predict(new_guess)

#probs = lr.predict_proba([new_guess])

# prediction 
if  user_pred== 1:
    user_label = "a LinkedIn User :star2:"
else:
    user_label = "not a LinkedIn User"
with st.sidebar:
    if st.checkbox(f" :arrow_left: Click the check box to see our prediciton!"):
        st.write(f"## We predict you are: ### \n {user_label}")


st.write(f"The chances of you being a LinkedIn User are : {probs[0][1]}")


#st.write(f" ## We predict that you are a {user_label}")

