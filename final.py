import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from htmlTemplates import css, bot_template, user_template

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    # contains raw text of pdfs in one variable, loop through the pdf objects and read them
    # read them take the contents of it and concatenate it to the variable here.

    text = ""
    for pdf in pdf_docs:
        # It creates the pdf object that has pages and we read from pages. So we need to loop through each page and add to text.
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, say "answer is not available in the context".
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain
    
def user_input(user_question):
    if "conversation" not in st.session_state:
        st.session_state.conversation = get_conversational_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = {}

    # Check if the current user question is already in the chat history
    if user_question not in st.session_state.chat_history:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        response = st.session_state.conversation({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        answer_text = response["output_text"]

        # Store the question and its corresponding answer in the chat history
        st.session_state.chat_history[user_question] = answer_text

    # Display chat history
    for question in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}", question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history[question]), unsafe_allow_html=True)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with the PDF", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    st.header("Chat with your Documents :books:")
    user_question = st.text_input("What do you want to know from your files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your files here", accept_multiple_files=True)
        
        if st.button("Collect Information"):
            with st.spinner("In progress..."):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #--> Takes the list of pdf files and going to return a single tezt string with all content
                # st.write(raw_text)    --> Just to see the text it appears on the sidebar

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)
                
                get_vector_store(text_chunks)
                st.success("DONE!")

if __name__ == '__main__':
    main()