
import streamlit as st
import uuid
import sys
from langchain.memory import ConversationBufferMemory
import bedrock as bedrock  # Assuming bedrock is a module you've created

# Constants and session initialization
USER_ICON = "images/user-icon.png"
AI_ICON = "images/ai-icon.png"
MAX_HISTORY_LENGTH = 5
PROVIDER_MAP = {
    'bedrock': 'AWS Bedrock Claude V2'
}

# Sidebar for persona selection
# Sidebar for persona selection
with st.sidebar:
    st.title("Persona Selection")
    persona_list = ['Security Analyst', 'Reviewer', 'Document Summarizer', 'Analytical Guru', 'Communication Advisor', 'DevOps Engineer','Python Developer']
    st.session_state.persona = st.selectbox("Choose a Persona:", persona_list, key='persona_sidebar')  # Update the state directly
    
    # Toggle for RAG
    use_rag = st.toggle('Use RAG', value=False, key='use_rag')


# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state['user_id'] = str(uuid.uuid4())

if 'persona' not in st.session_state:
    st.session_state.persona = "Communication Advisor"

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if "questions" not in st.session_state:
    st.session_state.questions = []

if "answers" not in st.session_state:
    st.session_state.answers = []

if "input" not in st.session_state:
    st.session_state.input = ""

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

# Initialize or update the chain based on the persona.
def update_chain():
    print(f"Building chain for persona: {st.session_state.persona}")
    st.session_state['llm_chain'] = bedrock.build_chain(persona=st.session_state.persona)

# Check if 'llm_chain' needs to be updated or initialized
# Initialize chain for RAG only when the toggle is activated
if st.session_state.get('use_rag', False):
    if 'llm_chain' not in st.session_state or st.session_state.persona != st.session_state.get('last_persona', None):
        print(f"Changing persona from {st.session_state.get('last_persona', 'None')} to {st.session_state.persona}")
        update_chain()
        st.session_state['last_persona'] = st.session_state.persona
        if len(sys.argv) > 1:
            if sys.argv[1] == 'bedrock':
                print(f"Initializing or updating Bedrock chain with persona: {st.session_state.persona}")  # Debug print
                st.session_state['llm_app'] = bedrock
                st.session_state['llm_chain'] = bedrock.build_chain(persona=st.session_state.persona)
            else:
                raise Exception(f"Unsupported LLM: {sys.argv[1]}")
        else:
            raise Exception("Usage: streamlit run app.py <bedrock>")


def handle_rag_input(input, persona):
    # Check if 'llm_chain' needs to be updated or initialized
    if 'llm_chain' not in st.session_state or st.session_state.persona != st.session_state.get('last_persona', None):
        print(f"Changing persona from {st.session_state.get('last_persona', 'None')} to {st.session_state.persona}")
        update_chain()
        st.session_state['last_persona'] = st.session_state.persona
    print(f"Handling input: {st.session_state.input}")  # Debug print
    input = st.session_state.input
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)

    chat_history = st.session_state["chat_history"]
    if len(chat_history) == MAX_HISTORY_LENGTH:
        chat_history = chat_history[:-1]

    llm_chain = st.session_state['llm_chain']
    chain = st.session_state['llm_app']
    result = chain.run_chain(llm_chain, input, chat_history)
    print(f"Question: {input}, Answer: {result['answer']}")
    answer = result['answer']
    chat_history.append((input, answer))
    
    document_list = []
    if 'source_documents' in result:
        for d in result['source_documents']:
            if not (d.metadata['source'] in document_list):
                document_list.append((d.metadata['source']))

    st.session_state.answers.append({
        'answer': result,
        'sources': document_list,
        'id': len(st.session_state.questions)
    })
    st.session_state.input = ""
    print(f"Chat History: {st.session_state['chat_history']}")
    st.rerun()
    st.session_state.questions.append(question_with_id)
    # Additional RAG-specific logic
    st.write(f"RAG Function Executed for {persona} with input: {input}")


def handle_chatbot_input(input, persona):
    question_with_id = {
        'question': input,
        'id': len(st.session_state.questions)
    }
    st.session_state.questions.append(question_with_id)

    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(ai_prefix="Assistant")

    # Invoke get_claude_response_without_rag and get the response
    response = bedrock.get_claude_response_without_rag(input, st.session_state.conversation_memory, persona)

    # # Debugging
    # print(f"Type of Response: {type(response)}")
    # if isinstance(response, dict) and 'answer' in response:
    #     print(f"Answer exists in response: {response['answer']}")
    # elif isinstance(response, str):
    #     print(f"Response is a string: {response}")

    # Store the answer into session_state
    st.session_state.answers.append({
        'answer': response if isinstance(response, dict) else {'answer': response},
        'id': len(st.session_state.questions)
    })

    # # Additional Chatbot-specific logic
    # st.write(f"Chatbot Function Executed for {persona} with input: {input}")
    # st.write(f"Response from Claude: {response if isinstance(response, dict) else response}")

    # Assuming you have a list in session_state for storing chat messages
    st.session_state.chat_messages.append({
        'role': 'assistant',
        'message': response
    })


# Function to clear chat
def clear_chat():
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
    st.session_state["chat_history"] = []

# Top Bar and Clear Chat Button
def write_top_bar():
    col1, col2, col3 = st.columns([1,10,2])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        selected_provider = sys.argv[1]
        if selected_provider in PROVIDER_MAP:
            provider = PROVIDER_MAP[selected_provider]
        else:
            provider = selected_provider.capitalize()
        header = f"An AI App powered by Amazon Kendra and {provider}!"
        st.write(f"<h3 class='main-header'>{header}</h3>", unsafe_allow_html=True)
    with col3:
        clear = st.button("Clear Chat")
    return clear

clear = write_top_bar()
if clear:
    clear_chat()

# Handle user input based on the selected mode (RAG or Chatbot)
prompt = st.chat_input("You are talking to an AI, ask any question.")
if prompt:
    st.session_state.input = prompt
    if use_rag:
        handle_rag_input(st.session_state.input, st.session_state.persona)
    else:
        handle_chatbot_input(st.session_state.input, st.session_state.persona)


# The rest of your code for rendering chat and other UI elements can go here
if clear:
    st.session_state.questions = []
    st.session_state.answers = []
    st.session_state.input = ""
    st.session_state["chat_history"] = []

def write_user_message(md):
    col1, col2 = st.columns([1,12])
    
    with col1:
        st.image(USER_ICON, use_column_width='always')
    with col2:
        st.warning(md['question'])

def render_result(result):
    answer, sources = st.tabs(['Answer', 'Sources'])
    with answer:
        render_answer(result['answer'])
    with sources:
        if 'source_documents' in result:
            render_sources(result['source_documents'])
        else:
            render_sources([])

def render_answer(answer):
    col1, col2 = st.columns([1,12])
    with col1:
        st.image(AI_ICON, use_column_width='always')
    with col2:
        if isinstance(answer, dict):
            st.info(answer.get('answer', 'No answer found'))
        else:
            st.info(answer)


def render_sources(sources):
    col1, col2 = st.columns([1,12])
    with col2:
        with st.expander("Sources"):
            for s in sources:
                st.write(s)
    
#Each answer will have context of the question asked in order to associate the provided feedback with the respective question
def write_chat_message(md, q):
    chat = st.container()
    with chat:
        render_answer(md['answer'])
        if 'sources' in md:
            render_sources(md['sources'])
        else:
            st.write("No sources available.")

with st.container():
  for (q, a) in zip(st.session_state.questions, st.session_state.answers):
    write_user_message(q)
    write_chat_message(a, q)

st.markdown('---')

# Add empty space
for _ in range(2):
    st.write("")
