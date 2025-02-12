import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay


def main():
    #documentation
    #st.markdown
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or pisonous? 🍄")
    st.sidebar.markdown("Are your mushrooms edible or pisonous? 🍄")
    
    #Loading the data
    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data
    
    #Split data
    @st.cache(persist = True)
    def split(df):
        y = df.type
        x = df.drop(columns = ['type'])
        x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    #Plot user selected evaluation metrics
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
            
            

    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisouous']
    st.sidebar.subheader("Choose Classifer")
    classifer = st.sidebar.selectbox(" Classifier", ("Support Vector Machie (SVM)", "Logitic Regression", "Random Forest"))
    
    
    if classifer == 'Support Vector Machie (SVM)':
        st.sidebar.subheader("Model Hyperparamters")
        C = st.sidebar.number_input("C (Regularization paramter)", 0.01 , 10.0, step= 0.01, key= 'C')
        kernel = st.sidebar.radio("kernel",("rbf","linear"), key ='kernel')
        gamma = st.sidebar.radio("Gamma(Kernel Coefficient",("scale", "auto"),key='gamma')
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data Set (Classification)")
        st.write(df)
            
    

if __name__ == '__main__':
    main()


