import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
st.title('IRIS Data Exploration')
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
if st.checkbox("Display IRIS Dataset"):
    st.markdown('#### IRIS Dataset')
    st.dataframe(df)


if st.checkbox("Show the average sepal length for each species"):
    st.markdown("#### Average Sepal Length Per Species")
    avg_sepal_length = df.groupby("species")["sepal_length"].mean()
    st.dataframe(avg_sepal_length)

st.subheader("Comparison of Two Features Using a Scatter Plot")
feature_1 = st.selectbox("Select the first feature:", df.columns[:-1])
feature_2 = st.selectbox("Select the second feature:", df.columns[:-1])
scatter_plot = px.scatter(df, x=feature_1, y=feature_2, color="species", hover_name="species")
st.plotly_chart(scatter_plot)

st.subheader('Filter data based on species')
selected_species = st.multiselect("select Species", df['species'].unique())

if selected_species:
    filtered_data = df[df['species'].isin(selected_species)]
    st.dataframe(filtered_data)
else:
    st.write("No species selected.")


if st.checkbox("Select pairplot for the selected species"):
    st.subheader("Pairplot for the Selected Species")
    if selected_species:
        sns.pairplot(filtered_data, hue="species")
    else:
        sns.pairplot(df, hue="species")
    st.pyplot()

st.subheader("Distribution of a Selected Feature")
selected_feature = st.selectbox("Select a feature to display its distribution:", df.columns[:-1])
hist_plot = px.histogram(df, x=selected_feature, color="species", nbins=30, marginal="box", hover_data=df.columns)
st.plotly_chart(hist_plot)
