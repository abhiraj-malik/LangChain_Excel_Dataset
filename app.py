import os
import streamlit as st
from apikey import apikey
from langchain.llms import OpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA

# your apikey
os.environ['OPENAI_API_KEY'] = 'your api key'

# creating app
st.title('🦜🔗 GPT Insurance Report')
prompt = st.text_input('plug in your prompt here')

llm = OpenAI(temperature=0.9)

# loading excel file
loader = CSVLoader(file_path='your file path')


# creating indices for talking to data
indexes = VectorstoreIndexCreator()
docsearch = indexes.from_loaders([loader])

# ques-answer chain for asking question
chain = RetrievalQA.from_chain_type(llm, chain_type='stuff',
                                    retriever=docsearch.vectorstore.as_retriever(), input_key="question")

if prompt:
    response = chain.run(prompt)
    st.write(response)

