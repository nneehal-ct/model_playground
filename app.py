import streamlit as st
import aisuite as ai
import wandb, weave
import json
import os
import tiktoken
from together import Together 

from dotenv import load_dotenv
load_dotenv()

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_API_KEY")
os.environ['TOGETHER_API_KEY'] = os.getenv('TOGETHER_API_KEY')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_context' not in st.session_state:
    st.session_state.file_context = ""
if 'system_message' not in st.session_state:
    st.session_state.system_message = "You are a helpful assistant."

weave.init("SVx")

PROMPT_REFS = {
    "No pre-set prompts": None,
    "Step 1: FSM Overview": "weave:///caspia-technologies/SVx/object/step_1:dCLtZ2iDazxnHhbYQhey9gfc7UAzs6UpUPcpVLsVldM",
    "Step 2:": "<test>"
}

MODELS = {
    "HF-Qwen2.5-72B-Instruct": "huggingface:Qwen/Qwen2.5-72B-Instruct",
    "HF-Llama-3.3-70B-Instruct": "huggingface:meta-llama/Llama-3.3-70B-Instruct",
    "Together-Mistral-Small-24B-Instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
    "OpenAI-GPT-4o": "openai:gpt-4o",
    "Anthropic-Claude-3.5-Sonnet": "anthropic:claude-3-5-sonnet-20240620"
}

def process_file_content(file):
    """Process different file types and return formatted content"""
    content = file.read().decode()
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'json':
            json_content = json.loads(content)
            return f"File: {file.name} (JSON)\n{json.dumps(json_content, indent=2)}\n"
        elif file_extension in ['v', 'sv']:
            return f"File: {file.name} (Verilog/SystemVerilog)\n{content}\n"
        else:
            return f"File: {file.name} (Text)\n{content}\n"
    except json.JSONDecodeError:
        return f"File: {file.name} (Invalid JSON - treating as text)\n{content}\n"
    except Exception as e:
        return f"File: {file.name} (Error processing file: {str(e)})\n"

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += -1
    num_tokens += 2
    return num_tokens

def format_response(message, model, content, step=None, purpose=None):
    """Helper function to format response HTML"""
    if step and purpose:
        return f"""
        <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
            <strong>Assistant ({model}):</strong>
            <br><br>
            <strong>Step {step}</strong><br>
            <strong>Purpose:</strong> {purpose}<br>
            <strong>Response:</strong><br>{content}
        </div>
        """
    else:
        return f"""
        <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
            <strong>Assistant ({model}):</strong><br>{content}
        </div>
        """

def ask(message, sys_message="You are a helpful agent.", model="groq:llama-3.2-3b-preview"):
    try:
        client = ai.Client()
        
        context = f"{message}\n\nAdditional Context:\n{st.session_state.file_context}" if st.session_state.file_context else message
        
        messages = [
            {"role": "system", "content": sys_message},
            {"role": "user", "content": context}
        ]
        
        token_count = num_tokens_from_messages(messages)
        st.success(f"Input Token count: {token_count}")
        
        with st.expander("View Full Prompt"):
            st.text_area("Complete prompt sent to model:", 
                        value=json.dumps(messages, indent=2),
                        height=300)
        
        response_placeholder = st.empty()
        
        # Extract step and purpose if present
        step = message.split("<step>")[1].split("</step>")[0] if "<step>" in message else None
        purpose = message.split("<purpose>")[1].split("</purpose>")[0] if "<purpose>" in message else None
        
        # Handle different model types
        if model.startswith("mistralai/"):
            # Special handling for Together AI models
            client = Together()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )
        else:
            # Default handling for all other models (including Anthropic)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False
            )

        # Handle response uniformly for all models
        response_text = response.choices[0].message.content if hasattr(response.choices[0], 'message') else response.choices[0].text
        formatted_response = format_response(message, model, response_text, step, purpose)
        response_placeholder.markdown(formatted_response, unsafe_allow_html=True)
        return response_text
            
    except Exception as e:
        st.error(f"Error during API call: {str(e)}")
        return None


def load_prompt(url):
    if url is None:
        return None
    return weave.ref(url).get()

st.set_page_config(layout="wide", page_title="Caspia Model Playground", page_icon="🚀")

st.image("https://caspiatechnologies.com/wp-content/uploads/2024/05/cropped-Logo-Mark-Dark.png", width=200)
st.title("Caspia Model Playground")

with st.sidebar:
    st.title("Chat Settings")
    selected_model = st.selectbox("Choose AI Model", options=list(MODELS.keys()))
    selected_prompt_name = st.selectbox("Select Preset Prompt", options=list(PROMPT_REFS.keys()))
    
    if selected_prompt_name != "No pre-set prompts" and st.button("Finalize This Prompt"):
        prompt_text = load_prompt(PROMPT_REFS[selected_prompt_name])
        st.success(f"Prompt loaded: {selected_prompt_name}")

    st.subheader("Upload Context Files")
    uploaded_files = st.file_uploader(
        "Upload files (.v, .sv, .txt, .json)", 
        type=['v', 'sv', 'txt', 'json'], 
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("Process Files"):
        file_contents = []
        for file in uploaded_files:
            content = process_file_content(file)
            file_contents.append(content)
        st.session_state.file_context = "\n".join(file_contents)
        st.success(f"Processed {len(uploaded_files)} files")

st.subheader("Input")

system_input = st.text_area(
    "System Message",
    value=st.session_state.system_message,
    height=100,
    key="system_input"
)

user_input = st.text_area(
    "User Message",
    value=st.session_state.get('current_prompt', ''),
    height=200,
    key="user_input"
)

if selected_prompt_name != "No pre-set prompts" and st.button("Load Selected Prompt", key="load_prompt_button"):
    prompt_text = load_prompt(PROMPT_REFS[selected_prompt_name])
    if prompt_text and "<purpose>" in prompt_text.content and "<prompt>" in prompt_text.content:
        purpose = prompt_text.content.split("<purpose>")[1].split("</purpose>")[0]
        prompt = prompt_text.content.split("<prompt>")[1].split("</prompt>")[0]
        
        st.session_state.system_message = f"You are a helpful assistant. Your task is to {purpose}"
        st.session_state.current_prompt = prompt
        st.rerun()

if st.button("Ask LLM", key="send_button"):
    if user_input:
        response = ask(message=user_input, sys_message=system_input, model=MODELS[selected_model])

st.markdown("---")