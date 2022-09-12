# import dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from utils import load_model
import torch
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# load the iris dataset
@st.cache
def load_data():

    # load the iris dataset with sklearn
    dataset_iris = load_iris()

    # define inputs and output
    df_inputs = pd.DataFrame(dataset_iris.data, columns=dataset_iris.feature_names)
    df_output = pd.DataFrame(dataset_iris.target, columns=['variety'])

    return df_inputs, df_output

# display eda figure based on source dataset
@st.cache(allow_output_mutation=True)
def data_visualization(df_inputs, df_output):

    df = pd.concat([df_inputs, df_output['variety']], axis=1)
    eda = sns.pairplot(data=df, hue="variety", palette=['#0D0888', '#CB4779', '#F0F922'])

    return eda

# create a sidebar with sliders
def create_slider(df_inputs):

    # slidebars with min, max and mean (by default) values
    sepal_length = st.sidebar.slider(
        label='Sepal Length',
        min_value=float(df_inputs['sepal length (cm)'].min()),
        max_value=float(df_inputs['sepal length (cm)'].max()),
        value=float(round(df_inputs['sepal length (cm)'].mean(), 1)),
        step=0.1)

    sepal_width = st.sidebar.slider(
        label='Sepal Width',
        min_value=float(df_inputs['sepal width (cm)'].min()),
        max_value=float(df_inputs['sepal width (cm)'].max()),
        value=float(round(df_inputs['sepal width (cm)'].mean(), 1)),
        step=0.1)

    petal_length = st.sidebar.slider(
        label='Petal Length',
        min_value=float(df_inputs['petal length (cm)'].min()),
        max_value=float(df_inputs['petal length (cm)'].max()),
        value=float(round(df_inputs['petal length (cm)'].mean(), 1)),
        step=0.1)

    petal_width = st.sidebar.slider(
        label='Petal Width',
        min_value=float(df_inputs['petal width (cm)'].min()),
        max_value=float(df_inputs['petal width (cm)'].max()),
        value=float(round(df_inputs['petal width (cm)'].mean(), 1)),
        step=0.1)

    return sepal_length, sepal_width, petal_length, petal_width

# run a PCA
@st.cache
def run_pca():

    pca = PCA(2)
    X = df_inputs.iloc[:, :4]
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame(pca.transform(X))
    df_pca.columns = ['PC1', 'PC2']
    df_pca = pd.concat([df_pca, df_output['variety']], axis=1)

    return pca, df_pca

# dataframe with value >=0
def extract_positive_value(prediction):

    # f(prediction) = max(0, prediction)
    prediction_positive = []
    for p in prediction:
        if p < 0:
            p = 0
        prediction_positive.append(p)

    return pd.DataFrame({'Species': ['Setosa', 'Versicolor', 'Virginica'], 'Confidence': prediction_positive})

# display image
def display_img(species):

    # define the list of images
    list_img = ['setosa.png', 'versicolor.png', 'virginica.png']

    return Image.open(list_img[species])

# main
if __name__ == '__main__':

    # app title
    st.title('Iris Flower Classifier')
    st.markdown('Visualize the Iris dataset and predict the species of an Iris flower using sepal and petal measurements.')

    # load the input and output data
    df_inputs, df_output = load_data()

    # sidebar with slidebars
    st.sidebar.header('Input Features')
    sepal_length, sepal_width, petal_length, petal_width = create_slider(df_inputs)

    # inputs scaling
    X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X = torch.FloatTensor(X)

    # run input through the model
    prediction, species = load_model(X)
    df_pred = pd.DataFrame({'Species': ['Setosa', 'Versicolor', 'Virginica'], 'Confidence': prediction})

    # run pca
    pca, df_pca = run_pca()

    # create the PCA chart
    pca_fig = px.scatter(df_pca, x='PC1', y='PC2', color='variety', hover_name='variety')

    # user input
    datapoint = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    datapoint_pca = pca.transform(datapoint)

    # add user input to the PCA chart
    pca_fig.add_trace(go.Scatter(x=[datapoint_pca[0, 0]], y=[datapoint_pca[0, 1]], mode='markers', marker={'color': 'black', 'size': 12}, name='Datapoint'))

    # define prediction bar chart
    fig = px.bar(extract_positive_value(prediction), x='Species', y='Confidence', width=400, height=400, color='Species', color_discrete_sequence=['#0D0888', '#CB4779', '#F0F922'])

    st.write('### EDA on iris dataset')
    eda = data_visualization(df_inputs, df_output)
    st.pyplot(eda)
    st.write('Here it can be seen that the **setosa** `0` variety is easily separated from the other two (**versicolor** `1` and **virginica** `2`).')

    st.write('The following classification is obtained from a PyTorch model exported in a previous [notebook](https://github.com/ovh/ai-training-examples/blob/main/notebooks/getting-started/pytorch/notebook_classification_iris.ipynb).')
    st.write('### Principle Component Analysis')
    pca_fig

    # create two columns for the web app
    col1, col2 = st.columns([3, 1])

    # column 1 will be for the predictions
    with col1:
        st.write('### Predictions')
        fig

    # column 2 will be for the PCA
    with col2:
        st.write('### Iris species')
        if st.button('Show flower image'):
            st.image(display_img(species), width=300)
            st.write(df_pred.iloc[species, 0])
