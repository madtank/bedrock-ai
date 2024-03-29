# from aws_langchain.kendra import AmazonKendraRetriever #custom library
from langchain.retrievers import AmazonKendraRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.vectorstores import Chroma
import sys
import os
import argparse

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

#Globals
MAX_HISTORY_LENGTH = 5
region = "us-west-2"
llm = Bedrock(
    region_name=region,
    model_kwargs={"max_tokens_to_sample": 500,  # Increased from 300
                  "temperature": 1.0,  # Slightly lowered
                  "top_k": 250,
                  "top_p": 0.999,
                  "anthropic_version": "bedrock-2023-05-31"},
    model_id="anthropic.claude-v2"
)

def build_chain(persona, persona_description):
  kendra_index_id = os.environ.get('KENDRA_INDEX_ID', None)
  if kendra_index_id:
    retriever = AmazonKendraRetriever(index_id=kendra_index_id, top_k=5, region_name=region)
  else:
    # BedrockEmbeddings if not using Kendra
    bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1")

    # Your existing PDF loader and text splitter
    loader = PyPDFDirectoryLoader("./data/")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap  = 100,
    )
    docs = text_splitter.split_documents(documents)

    vectorstore_chroma = Chroma.from_documents(
        docs,
        bedrock_embeddings,
    )
    retriever = vectorstore_chroma.as_retriever()  # Assuming Chroma has an as_retriever() method
  
  print(f"Building chain for persona: {persona}")  # Debug print

  prompt_template = f"""{persona_description}
  Human: Here are some documents that may be relevant. Please analyze them and provide your best response based on their content. 
  Keep your response under 240 tokens. If the documents do not contain any relevant information, clearly state that.
  <documents>
  {{context}}
  </documents>
  Assistant: Based on the documents provided, I will analyze and respond to the best of my ability. 
  If the necessary information is not contained within these documents, I will explicitly mention it.
  Human: Respond to the following:
  {{question}}
  Assistant:
  """

  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  condense_qa_template = """{chat_history}
  Human:
  Given the previous conversation and a follow-up below, rephrase the follow-up to be standalone. Please keep your response under 240 tokens.

  Follow Up: {question}
  Standalone Response:

  Assistant:
  """
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

  qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT},
        verbose=False)
  
  # print(f"Prompt Template: {prompt_template}")  # for debug
  return qa

def run_chain(chain, prompt: str, history=[]):
  # print(f"Running chain with prompt: {prompt}, history: {history}")  # Debug print
  return chain({"question": prompt, "chat_history": history})

def get_claude_response_without_rag(prompt, memory, persona_description):
    from langchain.prompts import PromptTemplate
    from langchain.chains import ConversationChain

    # Use the passed-in memory object for the conversation
    conversation = ConversationChain(
        llm=llm, verbose=False, memory=memory
    )

    prompt_template = f"""{persona_description}
    Current conversation:
    {{history}}

    Human: {{input}}

    Assistant:
    """

    claude_prompt = PromptTemplate.from_template(prompt_template)

    conversation.prompt = claude_prompt
    response = conversation.predict(input=prompt)
    
    return response
