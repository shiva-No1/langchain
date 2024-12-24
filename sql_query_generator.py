import pandas as pd
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import streamlit as st

OPENAI_API_KEY = st.secrets ["API_KEY"]

openai_emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Function to Extract Data from Uploaded Excel File
def excel_file_extract(data):
    try:
        df = pd.read_excel(data)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

# Function to Embed Data into FAISS
def openai_embedding_faiss(df):
    documents = [str(row.to_dict()) for _, row in df.iterrows()]
    metadatas = [{"row_index": idx} for idx in df.index]
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(documents, openai_emb, metadatas=metadatas)
    st.success("Data successfully embedded into FAISS.")
    return vectorstore

# Function to Search Data in FAISS
def search_data_faiss(vectorstore, user_question):
    docs = vectorstore.similarity_search(user_question, k=10)
    return docs

# Function to Create a Chain for SQL Query Generation
def creating_chain():
    prompt_template = """
    In the given dataset, it contains metadata of the database.
    Generate the SQL query for the user question with the context given. Consider the first row of the Excel as column names. 
    Generate a very accurate SQL query and provide a small explanation of the query below.

    Context:
    {context}

    Question:
    {user_question}

    Answer:
    """
    model = ChatOpenAI(model="gpt-4", temperature=0.5, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "user_question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to Handle User Questions
def ask_question(vectorstore, user_question):
    docs = search_data_faiss(vectorstore, user_question)
    if not docs:
        return "No relevant documents found."
    
    converted_docs = [Document(page_content=doc.page_content) for doc in docs]
    chain = creating_chain()
    response = chain({"input_documents": converted_docs, "user_question": user_question}, return_only_outputs=True)

    output_text = response.get("output_text", "")
    if "Explanation:" in output_text:
        answer, explanation = output_text.split("Explanation:", 1)
        return f"**Answer:** {answer.strip()}\n\n**Explanation:** {explanation.strip()}"
    else:
        return f"**Output:** {output_text}"

def main():
    st.header("SQL Query Generator 📄")
    st.sidebar.header("Upload Your Excel File")

    # Initialize session state variables
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    # File upload and embedding section
    with st.sidebar:
        data = st.sidebar.file_uploader("Upload Excel file:", type=["xlsx"])
        if data and st.sidebar.button("Click to Process"):
            df = excel_file_extract(data)
            if df is not None:
                st.sidebar.success("Excel file loaded successfully.")
                with st.spinner("Embedding data into FAISS..."):
                    st.session_state.vectorstore = openai_embedding_faiss(df)
            else:
                st.error("Failed to process the Excel file.")

    # Initialize chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):  
            st.markdown(message["content"])

    # User question section
    if user_question := st.chat_input("Ask a question about your data..."):
        if st.session_state.vectorstore is None:
            st.error("No data has been processed yet. Please upload and process an Excel file first.")
            return

        # Append user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate and display assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = ask_question(st.session_state.vectorstore, user_question)
                if not answer:
                    answer = "I'm sorry, I couldn't find an answer to your question."
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.markdown(answer)

main()
