import streamlit as st
import PyPDF2
import os
import json
import os
import streamlit as st
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings import GradientEmbedding
from llama_index.llms import GradientBaseModelLLM
from llama_index.vector_stores import CassandraVectorStore
from copy import deepcopy
from tempfile import NamedTemporaryFile

os.environ['GRADIENT_ACCESS_TOKEN'] = "sevG6Rqb0ztaquM4xjr83SBNSYj91cux"
os.environ['GRADIENT_WORKSPACE_ID'] = "4de36c1f-5ee6-41da-8f95-9d2fb1ded33a_workspace"

st.set_page_config(page_title = "NyayMitra", page_icon="⚖︎")
st.header('Chat with your legal companion NyayMitra')

def create_datastax_connection():

    cloud_config = {
        'secure_connect_bundle': 'secure-connect-temp-db.zip'
    }

    with open("temp_db-token.json") as f:
        secrets = json.load(f)

    CLIENT_ID = secrets["clientId"]
    CLIENT_SECRET = secrets["secret"]

    auth_provider = PlainTextAuthProvider(CLIENT_ID, CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    astra_session = cluster.connect()
    return astra_session

# Function to read PDF files from a directory
def read_pdf_from_directory(directory):
    pdf_texts = []
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
                pdf_texts.append(text)
    return pdf_texts

# Define the local directory containing PDF files
pdf_directory = "Z:\exp\env_chat_bot"

st.subheader('Processing PDF Files from Local Directory')

if st.button('Start'):
    with st.spinner('Starting bot...'):
        pdf_texts = read_pdf_from_directory(pdf_directory)

        if pdf_texts:
            st.success(f"Processed {len(pdf_texts)} PDF file(s) from the directory.")
            if "conversation" not in st.session_state:
                st.session_state.conversation = None

            if "activate_chat" not in st.session_state:
                st.session_state.activate_chat = False

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"], avatar = message['avatar']):
                    st.markdown(message["content"])

            session=create_datastax_connection()

            llm = GradientBaseModelLLM(base_model_slug="llama2-7b-chat", max_tokens=400)

            embed_model = GradientEmbedding(
            gradient_access_token = os.environ["GRADIENT_ACCESS_TOKEN"],
            gradient_workspace_id = os.environ["GRADIENT_WORKSPACE_ID"],
            gradient_model_slug="bge-large")

            service_context = ServiceContext.from_defaults(
            llm = llm,
            embed_model = embed_model,
            chunk_size=256)

            set_global_service_context(service_context)

            reader = SimpleDirectoryReader("Z:\exp\env_chat_bot")
            # Get the current directory
            reader = os.path.dirname(os.path.abspath(__file__))

            # Load data from the directory
            documents = reader.load_data()
            index = VectorStoreIndex.from_documents(documents,
                                                            service_context=service_context) #This line creates an index from the loaded documents.
            query_engine = index.as_query_engine() #This line converts the index to a query engine.
            if "query_engine" not in st.session_state:
                st.session_state.query_engine = query_engine
            st.session_state.activate_chat = True

            if st.session_state.activate_chat == True:
                if prompt := st.chat_input("Ask your query"):
                    if prompt := st.chat_input("Ask your question from the PDF?"):
                        with st.chat_message("user", avatar = '👨🏻'):
                            st.markdown(prompt)
                        st.session_state.messages.append({"role": "user", 
                                              "avatar" :'👨🏻',
                                              "content": prompt})
                        
                        query_index_placeholder = st.session_state.query_engine
                        pdf_response = query_index_placeholder.query(prompt)
                        cleaned_response = pdf_response.response
                        with st.chat_message("assistant", avatar='🤖'):
                            st.markdown(cleaned_response)
                        st.session_state.messages.append({"role": "assistant", 
                                              "avatar" :'🤖',
                                              "content": cleaned_response})
                        

            
                   
            # Now you can use pdf_texts for further processing or display
        else:
            st.warning("No PDF files found in the directory.")
