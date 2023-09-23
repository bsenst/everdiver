from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import ClarifaiEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Clarifai
import streamlit as st
import random
import time
import re

st.title("Welcome")

PAT = st.secrets.CLARIFAI_PAT

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource()
def get_document():
    loader = CSVLoader(file_path='./data-assets/enex-parsed.csv')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents

if st.secrets.ENV == "dev":
    documents = get_document()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore")
else:
    vectorstore = FAISS.load_local("vectorstore", embeddings)

retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

msgs = StreamlitChatMessageHistory(key="chat_history")

llm = Clarifai(
    pat=PAT,
    user_id="tiiuae",
    app_id="falcon",
    model_id="falcon-40b-instruct"
)

prompt_template = """
Don't try to make up an answer, if you don't know just say that you don't know.
Answer in the same language the question was asked.
Use only the following pieces of context to answer the question at the end.

{context}

Question: {question}
Answer:"""


PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff", # map_reduce, map_rerank, stuff, refine
    retriever = retriever, 
    chain_type_kwargs = {"prompt": PROMPT},
    return_source_documents = True,
    verbose = True
)

if len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message(
        "Please ask me anything about your evernote data!")

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Send a message"):
    st.chat_message("human").write(question)
    with st.chat_message("ai"):
        with st.spinner(text="Processing..."):
            try:
                response = qa_chain(question)
                st.markdown(response["result"])
                print(response["source_documents"])
            except Exception as e:
                print(e)
                st.error("Oops! Something went wrong. Try another Query!")
                st.stop()        
