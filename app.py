import streamlit as st
import pickle
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = pickle.load(open('clf.pkl','rb'))
tfidf = pickle.load(open('tfidf.pkl','rb'))

def clean_resume(txt):
    cleanTxt = re.sub('http:\\S+\s',' ',txt)
    cleanTxt = re.sub('@\\S+',' ',cleanTxt)
    cleanTxt = re.sub('#\\S+',' ',cleanTxt)
    cleanTxt = re.sub('RT|cc', ' ', cleanTxt)
    cleanTxt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]',r' ', cleanTxt)
    cleanTxt = re.sub('\s+', ' ', cleanTxt)
    return cleanTxt

#web app
def main():
    st.title("Resume Screening App")
    uploaded_file=st.file_uploader('Upload Resume',type=['txt','pdf'])

    if uploaded_file is not None:
        try:
            resume_bytes = uploaded_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text=resume_bytes.decode('latin-1')


        cleaned_resume=clean_resume(resume_text)
        input_features=tfidf.transform([cleaned_resume])
        prediction_id=clf.predict(input_features)[0]
        st.write("Predicted Category:", prediction_id)    





# python main
if __name__=="__main__":
    main()
