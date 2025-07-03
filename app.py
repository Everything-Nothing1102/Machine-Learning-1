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
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')  # Clean headers
        st.write("📄 COLUMN HEADERS:", df.columns.tolist())  # <--- Show actual columns
        st.session_state.df = df
        st.session_state.data_loaded = True
        st.success("✅ Dataset loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")


# Train models
 def train_models():
    df = st.session_state.df
    st.write("✅ Available columns:", df.columns.tolist())  # TEMP DEBUG
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    if 'Text' not in df.columns:
    st.error("❌ Column 'Text' not found in dataset.")
    st.write("Columns found:", df.columns.tolist())
    return
    X = tfidf.fit_transform(df['Text'])  # Now safe
    y = df['Category']                   # Capital C
    
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
    sns.countplot(data=df, y='Category', order=df['Category'].value_counts().index, ax=ax)
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

# Recommendations
def show_recommendations():
    if not st.session_state.data_loaded:
        st.warning("Load the dataset first.")
        return
    df = st.session_state.df
    st.subheader("🧠 Personalized News Recommendations")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Text'])
    st.markdown("**Select one or more articles you like:**")
    sample_articles = df.sample(10).reset_index(drop=True)
    article_choices = [f"{i+1}. {row['Text'][:80]}..." for i, row in sample_articles.iterrows()]
    selected = st.multiselect("Articles:", article_choices)
    if selected:
        selected_indices = [int(s.split('.')[0]) - 1 for s in selected]
        selected_vectors = X[sample_articles.iloc[selected_indices].index]
        similarity_scores = cosine_similarity(selected_vectors, X).mean(axis=0)
        top_indices = similarity_scores.argsort()[::-1]
        shown = set(sample_articles.iloc[selected_indices].index)
        recommendations = []
        for idx in top_indices:
            if idx not in shown:
                recommendations.append((idx, similarity_scores[idx]))
            if len(recommendations) >= 5:
                break
        st.markdown("### 🔍 Top Recommended Articles:")
        for idx, score in recommendations:
            st.markdown(f"**Score: {score:.2f}** — *{df.iloc[idx]['Category'].capitalize()}*")
            st.write(df.iloc[idx]['Text'])
            st.markdown("---")
    else:
        st.info("Select a few articles to receive personalized recommendations.")

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
