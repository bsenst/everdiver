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

with st.sidebar:
    clarifai_pat = st.text_input(
        "Clarifai PAT", type="password",
        help="The PAT is provided by default, but you can input your own.")
    "[Get your Clarifai PAT](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/)"

PAT = clarifai_pat if clarifai_pat else st.secrets.CLARIFAI_PAT

embeddings = ClarifaiEmbeddings(
    pat=PAT,
    user_id="openai",
    app_id="embed",
    model_id="text-embedding-ada"
)

@st.cache_resource()
def get_document():
    loader = CSVLoader(file_path='./data-assets/enex-parsed.csv')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)
    return documents

if st.secrets.ENV == "dev":
    documents = utils.get_document()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("vectorstore")
else:
    vectorstore = FAISS.load_local("model", embeddings)

retriever = vectorstore.as_retriever()

msgs = StreamlitChatMessageHistory(key="chat_history")

llm = Clarifai(
    pat=PAT,
    user_id="tiiuae",
    app_id="falcon",
    model_id="falcon-40b-instruct"
)

memory = ConversationBufferWindowMemory(
    chat_memory=msgs,
    return_messages=True,
    memory_key="chat_history",
    k=6
)

conversation = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    verbose=True if st.secrets.ENV == "dev" else False
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
                response = conversation.run(question)
            except Exception as e:
                st.error(e)
                st.stop()
        full_response = ""
        placeholder = st.empty()
        for chunk in response.split():
            full_response += chunk + " "
            time.sleep(random.uniform(0, 0.3))
            placeholder.markdown(full_response)
        placeholder.markdown(response)
