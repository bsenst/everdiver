from langchain.chains import ConversationalRetrievalChain
from langchain.memory import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import ClarifaiEmbeddings
from langchain.vectorstores import FAISS
from st_pages import Page, show_pages
from langchain.llms import Clarifai
import streamlit as st
import random
import time

st.title("Welcome")

PAT = st.secrets.CLARIFAI_PAT

# embeddings = ClarifaiEmbeddings(
#     pat=PAT,
#     user_id="openai",
#     app_id="embed",
#     model_id="text-embedding-ada"
# )

from langchain.embeddings import FakeEmbeddings

embeddings = FakeEmbeddings(size=1352)

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
    vectorstore = FAISS.load_local("model", embeddings)

retriever = vectorstore.as_retriever(search_kwargs={'k': 1})

msgs = StreamlitChatMessageHistory(key="chat_history")

llm = Clarifai(
    pat=PAT,
    user_id="tiiuae",
    app_id="falcon",
    model_id="falcon-40b-instruct"
)

# memory = ConversationBufferWindowMemory(
#     chat_memory=msgs,
#     return_messages=True,
#     memory_key="chat_history",
#     k=6
# )

# conversation = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     retriever=retriever,
#     memory=memory,
#     verbose=True if st.secrets.ENV == "dev" else False
# )

from langchain import PromptTemplate, LLMChain

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

from langchain.chains import RetrievalQA

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
        with st.spinner(text="Thinking..."):
            try:
                response = qa_chain(question)
                st.markdown(response["result"])
            except Exception as e:
                error_string = str(e)
                pattern = r'"(.*?)"'
                match = re.findall(pattern, error_string)
                try:
                    error_desc = match[0]
                except:
                    error_desc = error_string
                response = "Oops! Something went wrong. Error: " + error_desc
                st.error(response)
                st.stop()        
