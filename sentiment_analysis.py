# Import packages
import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk #natural language toolkit
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from PIL import Image
import matplotlib.pyplot as plt


# AT Global level
# Page setup
st.set_page_config(page_title="Sentiment Analysis", page_icon="📈", layout="wide")

# Streamlit configuration
#st.set_option("deprecation.showPyplotGlobalUse", False)

# SSL context for NLTK downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Download required NLTK packages
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# Load Dataset
dataset = "/Users/gloriabotchway/PycharmProjects/PythonProject/Labelled_stories.txt"

with open(dataset, "r", encoding="UTF-8") as file:
    lines = file.readlines()

# Parse Dataset
parsed_data = []

for line in lines:
    parts = line.strip().split("\t")
    if len(parts) == 2:
        # FIRST = TEXT
        # SECOND = CLASS
        text, class_label = parts
        parsed_data.append([text, class_label])
    else:
        parsed_data.append([line.strip(),"Unknown"])

# Create Data Frame
df = pd.DataFrame(parsed_data, columns=["Text", "Class"])

# Data Cleaning
def clean_text(texts):
    cleaned = []
    for text in texts:
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)  # Remove HTML
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        cleaned.append(text)
    return cleaned

# Clean ONLY the Text column
df_clean = df.copy()
df_clean["Text"] = clean_text(df_clean["Text"])

# Tokenise each Text observation separately
df_clean["Tokens"] = df_clean["Text"].apply(word_tokenize)

# STOPWORDS
stop_words = set(stopwords.words("english"))

# REMOVE STOPWORDS
df_clean["Filtered Text"] = df_clean["Tokens"].apply(
    lambda tokens: [word for word in tokens
        if word.lower() not in stop_words
    ]
)

# CONVERT TOKENS BACK TO TEXT
df_clean["Preprocessed Text"] = df_clean["Filtered Text"].apply(lambda tokens: " ".join(tokens))

#Final dataset to used for building the model
final_data=df_clean[["Preprocessed Text", "Class"]]

# PAGE 1 — CORPORA VIEWER
def page1():
    st.subheader("Corpora Viewer")

    if st.checkbox("Raw Data: Display the original text data"):
        st.write(lines)

    if st.checkbox("Tabular representation of Text and Class"):
        st.dataframe(df, use_container_width=True)

    if st.checkbox("Click to upload file"):
        uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            uploaded_lines = (uploaded_file.read().decode("utf-8").splitlines())
            st.write(uploaded_lines)

    if st.checkbox("Preprocessed Data"):
        st.write(final_data)

# PAGE 2 — DATA PREPROCESSING
def page2():
    st.subheader("Text Preprocessing")

    # 1. Original Text and Class
    st.markdown("### 1. Original Text and Class")
    st.dataframe(df[["Text", "Class"]],use_container_width=True)

    # 2. Cleaned Text
    st.markdown("### 2. Cleaned Text")
    st.dataframe(df_clean[["Text", "Class"]],use_container_width=True)

    # 3. Tokenisation
    st.markdown("### 3. Tokenisation")
    st.write("Each Text observation is tokenised separately.")
    st.dataframe(
        df_clean[["Text", "Class", "Tokens"]],use_container_width=True)

    # 4. Stopwords
    #st.markdown("### 4. Stopwords")
    #st.write("Sample English stopwords:")
    #st.write(list(stop_words)[:100])

    # 5. Stopword Removal
    st.markdown("### 5. Stopword Removal")
    st.dataframe(df_clean[["Text", "Class", "Filtered Text"]], use_container_width=True)

    # 6. Final Preprocessed Text
    st.markdown(
        "### 6. Final Preprocessed Text")
    st.dataframe(df_clean[["Preprocessed Text", "Class"]],use_container_width=True)

# PAGE 3 — SENTIMENT ANALYSIS

def page3():
    st.header("Sentiment Analysis")

    # MODEL TRAINING
    # Separate predictor variable from class variable
    # X = text, y = class
    x_predictor = df_clean['Preprocessed Text']
    y_output = df_clean['Class']

    # Train-test split: 90% training and 10% testing
    X_train, X_test, y_train, y_test = train_test_split(x_predictor, y_output, test_size=0.1,random_state=40)

    # Feature extraction using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Naive Bayes classifier
    nb = MultinomialNB()
    nb.fit(X_train_tfidf, y_train)

    # Predict test results
    test_predict = nb.predict(X_test_tfidf)

    # MODEL EVALUATION
    accuracy = accuracy_score(y_test, test_predict) * 100
    st.write(f"### Model Accuracy: {accuracy:.2f}%")

    report = classification_report(y_test, test_predict)
    #st.write("### Classification Report")
    #st.text(report)


    # USER INPUT
    # Initialize session state
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""

    # Text input area
    user_input = st.text_area("Enter text here",key="user_input", height=150)
    st.write(f"You wrote: {len(user_input)} characters")

    # File uploader
    file_upload = st.file_uploader("Or upload a file 📄",type=["txt", "csv"])

    # FILE PROCESSING

    file_content = ""
    if file_upload is not None:
        try:
            file_content = file_upload.read().decode("utf-8")
            st.success("File content loaded successfully!")
        except Exception as e:
            st.error(f"Could not read the file: {str(e)}")

    # File content takes priority over text input
    final_input = file_content if file_content else user_input

    # Create a row with two button columns and an empty spacer
    col1, col2, col3 = st.columns([1, 1, 4])

    # Predict button
    with col1:
        predict_button = st.button(
            "Predict",
            use_container_width=True
        )

    # Clear button
    with col2:
        clear_button = st.button(
            "Clear Text",
            use_container_width=True
        )

    # Empty space
    with col3:
        st.empty()

    # CLEAR TEXT
    if clear_button:
        st.session_state.user_input = ""
        st.rerun()

    # PREDICTION
    if predict_button:
        if not final_input:
            st.warning("Please enter text or upload a file first!")
        else:
            try:
                # Convert user input into TF-IDF features
                user_input_tfidf = tfidf.transform([final_input])

                # Make prediction
                prediction = nb.predict(user_input_tfidf)

                # POSITIVE SENTIMENT
                if prediction[0] == "Positive":
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.write(
                            f"### Sentiment is: 😊 {prediction[0]}"
                        )

                    with result_col2:
                        image = Image.open("sabinus2.jpg")
                        st.image(image,width=250)


                # NEGATIVE SENTIMENT
                else:
                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.write(f"### Sentiment is: ☹️ {prediction[0]}" )

                    with result_col2:
                        image1 = Image.open( "sabinus.jpg")

                        st.image(image1, width=250)

            except Exception as e:
                st.error( f"Prediction failed: {str(e)}")


# PAGE 4 — LLAMA
def page4():

    st.subheader("Large Language Model with Llama")
    st.write("This section will contain the ""Llama-based Large Language Model.")

# SIDEBAR NAVIGATION
pages = {
    "Corpora Viewer": page1,
    "Data Preprocessing": page2,
    "Sentiment Analysis": page3,
    "Large Language Model with Llama": page4
}

select_page = st.sidebar.selectbox("Select a page",list(pages.keys()))
pages[select_page]()