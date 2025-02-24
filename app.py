import streamlit as st
import joblib

# Load model and vectorizer
def load_model():
    try:
        vectorizer = joblib.load("vectorizer.jb")
        model = joblib.load("lr_model.jb")
        return vectorizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Make prediction
def make_prediction(vectorizer, model, input_text):
    transform_input = vectorizer.transform([input_text])
    prediction = model.predict(transform_input)
    return prediction

# Main app
def main():
    st.title("Fake News Detector")
    st.write("Enter a News Article below to check whether it is Fake or Real. ")
    
    vectorizer, model = load_model()
    
    input_text = st.text_area("News Article:", "")
    
    if st.button("Check "):
        if input_text.strip():
            prediction = make_prediction(vectorizer, model, input_text)
            if prediction[0] == 1:
                st.success("The News is Real! ")
            else:
                st.error("Oops !!! The News is Fake! ")
        else:
            st.warning("Please enter some text to Analyze. ")

if __name__ == "__main__":
    main()
