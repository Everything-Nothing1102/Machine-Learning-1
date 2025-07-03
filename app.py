import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Personalized News Finder", page_icon="📰", layout="wide")

# Initialize session state
for key in ['data_loaded', 'models_trained', 'df', 'models']:
    if key not in st.session_state:
        st.session_state[key] = False if 'loaded' in key or 'trained' in key else None if key == 'df' else {}

# Load datasetdef load_dataset():
def load_dataset():
    url = "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/dataset.csv"
    try:
        df = pd.read_csv(url, encoding='ISO-8859-1')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '').str.lower()  
        st.write("📄 Columns in dataset:", df.columns.tolist())
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")


# Train models
def train_models():
    df = st.session_state.df

    if 'news' not in df.columns or 'type' not in df.columns:
        st.error("❌ Required columns 'news' and 'type' not found.")
        st.write("📄 Available columns:", df.columns.tolist())
        return

    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['news'])   
    y = df['type']                        

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

    st.session_state.models_trained = True
    st.session_state.models = {
        'Logistic Regression': report_lr,
        'K-Nearest Neighbors': report_knn
    }
    st.success("✅ Models trained successfully!")

# Dataset overview
def show_dataset_overview():
    if not st.session_state.data_loaded:
        st.warning("Load the dataset first.")
        return
    df = st.session_state.df
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.subheader("🗂️ Category Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, y='type', order=df['type'].value_counts().index, ax=ax)
    st.pyplot(fig)

# Model performance
def show_model_performance():
    if not st.session_state.models_trained:
        st.warning("Train the models first.")
        return
    st.subheader("📊 Model Performance")
    for name, report in st.session_state.models.items():
        st.markdown(f"#### {name}")
        st.dataframe(pd.DataFrame(report).transpose())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(user_article, df, tfidf_vectorizer, top_n=7):
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['news'])
    user_vec = tfidf_vectorizer.transform([user_article])
    cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

    top_indices = cosine_similarities.argsort()[-(top_n + 1):][::-1]  
    return df.iloc[top_indices[1:]]  


def show_recommendations():
    df = st.session_state.df

    if 'news' not in df.columns or 'type' not in df.columns:
        st.error("❌ Required columns ('news', 'type') not found.")
        return

    st.subheader("📝 Enter your news interest or custom query:")
    user_input = st.text_area("🔍 Type something like 'latest politics in UK' or 'technology trends'")

    if user_input.strip():
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(df['news'])
        user_vec = tfidf.transform([user_input])
        cosine_similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()

        top_indices = cosine_similarities.argsort()[::-1][:7]
        top_scores = cosine_similarities[top_indices]

        st.subheader("🔁 Top Matches Based on Your Input:")
        for i, idx in enumerate(top_indices):
            score_percent = round(top_scores[i] * 100, 2)
            st.markdown(f"**{i+1}. ({score_percent}% match)** {df.iloc[idx]['news'][:250]}...")
    else:
        st.info("⌨️ Enter a query above to get personalized article suggestions.")


# Home
def show_home():
    st.title("📰 Personalized News Finder")
    st.caption("AI-Powered News Classification and Recommendation System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("- Text classification with Logistic Regression and KNN\\n"
                    "- Interactive dataset exploration\\n"
                    "- Cosine similarity-based recommendations")
    with col2:
        st.markdown("### 🚀 Get Started")
        st.markdown("1. Load the dataset\\n"
                    "2. Train the models\\n"
                    "3. Evaluate model performance\\n"
                    "4. Get recommendations")
    st.markdown("### ⚡ Quick Actions")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("📂 Load Dataset"):
            load_dataset()
    with c2:
        if st.button("🤖 Train Models"):
            if st.session_state.data_loaded:
                train_models()
            else:
                st.warning("Please load the dataset first!")
    with c3:
        if st.button("📈 Show Performance"):
            if st.session_state.models_trained:
                show_model_performance()
            else:
                st.warning("Please train the models first!")

# Main
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Home", "Dataset Overview", "Model Performance", "Personalized Recommendations"
    ])
    if page == "Home":
        show_home()
    elif page == "Dataset Overview":
        show_dataset_overview()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Personalized Recommendations":
        show_recommendations()

if __name__ == "__main__":
    main()
