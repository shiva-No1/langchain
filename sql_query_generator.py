
import pandas as pd
from secret_file import API_KEY
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_function
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
import streamlit as st

OPENAI_API_KEY = st.secrets ("API_KEY")

openai_emb = embedding_function.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002"
)

# Function to Extract Data from Uploaded Excel File
def excel_file_extract(data):
    try:
        df = pd.read_excel(data)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

# Function to Embed Data into ChromaDB
def openai_embedding(df, batch_size=10):
    client = PersistentClient(path="chroma_vector_data/")
    collection = client.get_or_create_collection(
        name="excel_collection", embedding_function=openai_emb
    )
    
    total_rows = len(df)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = df.iloc[start:end]

        documents = [str(row.to_dict()) for i, row in batch.iterrows()]
        metadatas = [{"row_index": idx} for idx in batch.index]
        ids = [f"row_{idx}" for idx in batch.index]

        # Add data to ChromaDB collection
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
    
    st.success("Batch processing completed.")
    return collection

# Function to Search Data in ChromaDB
def search_data_chromadb(collection, user_question):
    results = collection.query(query_texts=[user_question], n_results=10)
    return results.get("documents", [[]])[0]

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
def ask_question(collection, user_question):
    docs = search_data_chromadb(collection, user_question)
    if not docs:
        return "No relevant documents found."
    
    converted_docs = [Document(page_content=i) for i in docs]
    chain = creating_chain()
    response = chain({"input_documents": converted_docs, "user_question": user_question}, return_only_outputs=True)

    output_text = response.get("output_text", "")
    if "Explanation:" in output_text:
        answer, explanation = output_text.split("Explanation:", 1)
        return f"**Answer:** {answer.strip()}\n\n**Explanation:** {explanation.strip()}"
    else:
        return f"**Output:** {output_text}"

# Main Function with Chat Interface
def main():
    st.header("SQL Query Generator")
    st.sidebar.header("Upload Your Excel File")

    # Initialize session state variables
    if "collection" not in st.session_state:
        st.session_state.collection = None

    # File upload and embedding section
    data = st.sidebar.file_uploader("Upload Excel file:", type=["xlsx"])
    if data and st.sidebar.button("Click to Process"):
        df = excel_file_extract(data)
        if df is not None:
            st.sidebar.success("Excel file loaded successfully.")
            with st.spinner("Embedding data into ChromaDB..."):
                st.session_state.collection = openai_embedding(df)
            st.success("Data embedded successfully.")
        else:
            st.error("Failed to process the Excel file.")

    # User question section
    if user_question := st.chat_input("Ask a question about your data..."):
        if st.session_state.collection is None:
            st.error("No data has been processed yet. Please upload and process an Excel file first.")
            return

        # Add user input to chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Generate Response
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                answer = ask_question(st.session_state.collection, user_question)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)

main()


