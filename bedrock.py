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

MAX_HISTORY_LENGTH = 5
# Global Persona Mapping
PERSONA_PROMPT_MODIFICATION = {
    'Security Analyst': "You should delve into documents to extract and analyze security-related information, ensuring protocols are adhered to.",
    'Reviewer': "You should meticulously review documents, pointing out key information and potential areas of improvement.",
    'Document Summarizer': "You should distill lengthy documents into concise, essential summaries, making information easily digestible.",
    'Analytical Guru': "You should interpret documents, providing a thorough analysis while connecting the dots between content and broader implications.",
    'Communication Advisor': "You should assist in drafting, improving, and proofreading responses for emails or instant messages. Ensure the responses are articulate, accurate, and professionally composed while maintaining a tone that reflects my communication style.",
    'DevOps Engineer': "You should help with infrastructure as code tasks, primarily focusing on Terraform. Ensure that best practices are followed and that the code is efficient and secure.",
    'Python Developer': "You should assist with Python development tasks. This includes debugging, optimizing code, and ensuring that best coding practices are adhered to."
}

def build_chain(persona):
  region = "us-west-2"
  llm = Bedrock(
      region_name=region,
      model_kwargs={"max_tokens_to_sample": 300, 
                    "temperature": 1, "top_k": 250, "top_p": 0.999, 
                    "anthropic_version": "bedrock-2023-05-31"},
                     model_id="anthropic.claude-v1"
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
  
  # print(f"Building chain for persona: {persona}")  # Debug print

  prompt_template = f"""Human: The AI is {PERSONA_PROMPT_MODIFICATION.get(persona)}
  Assistant: Acknowledged, I'm your {persona}

  Human: Use the documents and/or your knowledge to answer.
  <documents>
  {{context}}
  </documents>
  Based on the above, provide a short answer for, {{question}}

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
        verbose=False)
  
  # print(f"Prompt Template: {prompt_template}")  # for debug
  return qa

def run_chain(chain, prompt: str, history=[]):
  # print(f"Running chain with prompt: {prompt}, history: {history}")  # Debug print
  return chain({"question": prompt, "chat_history": history})


#### Debug from CLI Only ####
#### Debug from CLI Only ####


# Function to display settings menu and update configurations
def display_settings_menu(current_settings):
  while True:
    print(f"Current Settings - Persona: {current_settings['current_persona']}")
    user_input = input("Press 'Enter' to continue or type 'persona' to change persona: ")
    
    if user_input == '':
      print("Continuing the chat session...")
      break
    
    elif user_input == 'persona':
      print(f"Available Personas: {list(PERSONA_PROMPT_MODIFICATION.keys())}")
      selected_persona = input("Select a persona: ")
      if selected_persona in PERSONA_PROMPT_MODIFICATION.keys():
        current_settings['current_persona'] = selected_persona
        print(f"Changed persona to {selected_persona}")
      else:
        print("Invalid persona. No changes made.")
    
    elif user_input == 'persona':
      available_personas = ['Security Analyst', 'Reviewer', 'Document Summarizer', 
                            'Analytical Guru', 'Communication Advisor', 'DevOps Engineer','Python Developer']
      print(f"Available Personas: {available_personas}")
      selected_persona = input("Select a persona: ")
      if selected_persona in available_personas:
        current_persona = selected_persona
        print(f"Changed persona to {current_persona}")
      else:
        print("Invalid persona. No changes made.")
    else:
      print("Invalid input. Please try again.")
  return current_settings

def main(initial_persona):
    current_settings = {'current_persona': initial_persona}
    chat_history = []  # Initialize chat_history outside the loop
    qa = None  # Initialize the chain variable

    # Display settings menu at startup
    print("Welcome! Let's set up your initial settings.")
    current_settings = display_settings_menu(current_settings)  # Update current_settings
    # ... (rest of your code remains the same)

    while True:  # Main loop for queries or other input
        query = input(">")
        if query.strip() == '!settings':
          # Open settings menu
          current_settings = display_settings_menu(current_settings)  # Update current_settings
          qa = None  # Invalidate the existing chain, so it gets rebuilt on next loop iteration
          continue
        elif query.strip() == 'exit':
          print("Exiting program...")
          break

        # Build the chain if it's not built yet (or if settings changed)
        if qa is None:
            qa = build_chain(current_settings['current_persona'])

        if query.strip().lower().startswith("new search:"):
          query = query.strip().lower().replace("new search:", "")
          chat_history = []
        elif len(chat_history) == MAX_HISTORY_LENGTH:
          chat_history.pop(0)

        # Run the chain based on settings
        result = run_chain(qa, query, chat_history)

        chat_history.append((query, result["answer"]))
        print(result['answer'])

        if 'source_documents' in result:
          print('Sources:')
          for d in result['source_documents']:
            print(d.metadata['source'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Choose a persona for the conversational model.')
    parser.add_argument('--persona', type=str, default='Default',
                        help='The persona to use for the conversational model')
    args = parser.parse_args()
    main(args.persona)