import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Custom CSS
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    .sidebar .sidebar-content .sidebar-top {
        color: #333333;
    }
    .stButton>button {
        background-color: #4CAF50; /* Green */
        border: none;
        color: white;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_data(dataset_name):
    if dataset_name == "IRIS":
        data = datasets.load_iris()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    else:
        raise ValueError("Invalid dataset name")
    return data

def preprocess_data(data):
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test

def select_model(model_name):
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Neural Network":
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    elif model_name == "Naive Bayes":
        model = GaussianNB()
    else:
        raise ValueError("Invalid model name")
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    return model.predict(X_test)
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix_wrapper(model, X_test, y_test):
    # Predict the labels
    y_pred = model.predict(X_test)
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix using seaborn heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    st.pyplot(plt.gcf())  # Pass the current figure explicitly

def main():
    st.title("Streamlit Predictor")

    dataset_name = st.sidebar.selectbox("Select Dataset", ("IRIS", "Digits"))
    data = load_data(dataset_name)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    st.write(f"Loaded {dataset_name} dataset")
    st.write("Number of samples in training data:", X_train.shape[0])
    st.write("Number of samples in testing data:", X_test.shape[0])
    st.write("Number of unique classes in true labels:", len(set(y_test)))

    model_name = st.sidebar.selectbox("Select Model", ("Logistic Regression", "Neural Network", "Naive Bayes"))
    model = select_model(model_name)

    st.write(f"Selected {model_name} model")

    model = train_model(model, X_train, y_train)

    # Dynamically generate input fields based on the selected dataset
    st.sidebar.subheader("Enter Feature Values")
    feature_values = []
    if dataset_name == "IRIS":
        feature_ranges = [(min(feature), max(feature)) for feature in zip(*data.data)]
    elif dataset_name == "Digits":
        feature_ranges = [(0, 16)] * data.data.shape[1]  # Digits dataset features range from 0 to 16
    else:
        raise ValueError("Invalid dataset name")

    for i in range(len(data.feature_names)):
        feature_range = feature_ranges[i]
        feature_value = st.sidebar.number_input(data.feature_names[i], value=feature_range[0], min_value=feature_range[0], max_value=feature_range[1])
        st.sidebar.write(f"Min: {feature_range[0]}, Max: {feature_range[1]}")  # Display min and max values
        feature_values.append(feature_value)

    if st.sidebar.button("Make Predictions", key="make_predictions_button"):
        # Validate user inputs
        if len(feature_values) != X_test.shape[1]:
            st.error("Number of feature values entered does not match the dataset.")
        else:
            # Predict using user inputs
            user_input_array = np.array(feature_values).reshape(1, -1)
            y_pred = predict(model, user_input_array)
            st.write("Predicted Class:", y_pred[0])

            plot_confusion_matrix_wrapper(model, X_test, y_test)          

if __name__ == "__main__":
    main()