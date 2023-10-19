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

MAX_HISTORY_LENGTH = 5

def build_chain(persona):
  region = "us-west-2"
  llm = Bedrock(
      region_name=region,
      model_kwargs={"max_tokens_to_sample": 300, "temperature": 1, "top_k": 250, "top_p": 0.999, "anthropic_version": "bedrock-2023-05-31"},
      model_id="anthropic.claude-v2"
  )
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

  persona_prompt_modification = {
      'Friendly AI': "I'll be a friendly AI assistant.",
      'Dev': "I'll respond like a software developer.",
      'Guru': "I'll act as a yogi.",
      'Comedian': "I'll try to keep things funny."
  }

  prompt_template = f"""Human: The AI is talkative and provides specific details from its context but limits it to 240 tokens.
  If the AI does not know the answer to a question, it truthfully says it 
  does not know.
  
  Assistant: {persona_prompt_modification.get(persona, "I'll be a friendly AI assistant.")}

  Human: Here are a few documents in <documents> tags:
  <documents>
  {{context}}
  </documents>
  Based on the above documents, provide a detailed answer for, {{question}} 
  Answer "don't know" if not present in the document. 
  
  Assistant:
  """
  PROMPT = PromptTemplate(
      template=prompt_template, input_variables=["context", "question"]
  )
  condense_qa_template = """{chat_history}
  Human:
  Given the previous conversation and a follow up question below, rephrase the follow up question
  to be a standalone question.

  Follow Up Question: {question}
  Standalone Question:

  Assistant:"""
  standalone_question_prompt = PromptTemplate.from_template(condense_qa_template)

  qa = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever,
        condense_question_prompt=standalone_question_prompt, 
        return_source_documents=True, 
        combine_docs_chain_kwargs={"prompt":PROMPT},
        verbose=True)
  return qa

def run_chain(chain, prompt: str, history=[]):
  return chain({"question": prompt, "chat_history": history})


#### Debug from CLI Only ####

# Function to display settings menu and update configurations
def display_settings_menu():
  global use_document_retrieval  # Using global to modify the variable defined outside the function
  global current_persona  # Same as above
  
  while True:
    print(f"Current Settings - Use Document Retrieval: {use_document_retrieval}, Persona: {current_persona}")
    user_input = input("Press 'Enter' to continue, or type 'use_doc' to change document retrieval or 'persona' to change persona: ")
      
    if user_input == '':
      print("Continuing the chat session...")
      break
    
    if user_input == 'use_doc':
      print(f"Current setting for Use Document Retrieval is {use_document_retrieval}")
      new_setting = input("Type 't' for True or 'f' for False: ").lower()
      if new_setting in ['t', 'true']:
        use_document_retrieval = True
      elif new_setting in ['f', 'false']:
        use_document_retrieval = False
      else:
        print("Invalid input. No changes made.")
      print(f"Updated Use Document Retrieval to {use_document_retrieval}")
    
    elif user_input == 'persona':
      available_personas = ["Default", "Friendly", "Professional"]
      print(f"Available Personas: {available_personas}")
      selected_persona = input("Select a persona: ")
      if selected_persona in available_personas:
        current_persona = selected_persona
        print(f"Changed persona to {current_persona}")
      else:
        print("Invalid persona. No changes made.")
    else:
      print("Invalid input. Please try again.")    


# Initial default settings
use_document_retrieval = True
current_persona = "Default"

if __name__ == "__main__":
  chat_history = []  # Initialize chat_history outside the loop
  qa = None  # Initialize the chain variable

  # Display settings menu at startup
  print("Welcome! Let's set up your initial settings.")
  display_settings_menu()
  # Inform user they can change settings later
  print("Note: You can change these settings at any time by typing '!settings'.")
  
  print(bcolors.OKBLUE + "Hello! How can I help you?" + bcolors.ENDC)
  print(bcolors.OKCYAN + "Ask a question, start a New search: or CTRL-D to exit." + bcolors.ENDC)

  while True:  # Main loop for queries or other input
    query = input(">")
    if query.strip() == '!settings':
      # Open settings menu
      display_settings_menu()
      qa = None  # Invalidate the existing chain, so it gets rebuilt on next loop iteration
      continue
    elif query.strip() == 'exit':
      print("Exiting program...")
      break

    # Build the chain if it's not built yet (or if settings changed)
    if qa is None:
      if use_document_retrieval:
        qa = build_chain()  # Build chain for Kendra or document-based retrieval
      else:
        print('Build conversational chain here')
        # qa = build_conversational_chain()  # Build chain for conversational mode

    if query.strip().lower().startswith("new search:"):
      query = query.strip().lower().replace("new search:", "")
      chat_history = []
    elif len(chat_history) == MAX_HISTORY_LENGTH:
      chat_history.pop(0)

    # Run the chain based on settings
    result = run_chain(qa, query, chat_history)

    chat_history.append((query, result["answer"]))
    print(bcolors.OKGREEN + result['answer'] + bcolors.ENDC)

    if 'source_documents' in result:
      print(bcolors.OKGREEN + 'Sources:')
      for d in result['source_documents']:
        print(d.metadata['source'])
