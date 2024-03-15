import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set Google API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
        Please provide a detailed answer based on the given context. If the answer is not available in the provided information, simply state that it's not in the context. Avoid providing incorrect information.\n\n
        Context:\n {context}?\n
        Question:\n{question}\n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the vector store from the local file
    new_db = FAISS.load_local("faiss_index", embeddings)
    
    # Search for similar documents based on the user's question
    docs = new_db.similarity_search(user_question)

    # If no relevant documents are found, return "out of context" response
    if not docs:
        st.write("Reply: Out of context")
        return None
    
    # Otherwise, continue with the conversational chain
    chain = get_conversational_chain()

    # Generate response based on the provided context and user's question
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response


def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini Pro Model")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button",
                                     accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask me any question related to pdf"):
        # Display user message in chat message container
        st.chat_message("user").markdown(user_question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Get response from assistant
        response = user_input(user_question)
        
        if response:
            response_text = response["output_text"]
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(response_text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
