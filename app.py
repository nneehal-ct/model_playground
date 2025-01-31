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

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'file_context' not in st.session_state:
    st.session_state.file_context = ""
# At the beginning of your script, add this to your session state initialization
if 'system_message' not in st.session_state:
    st.session_state.system_message = "You are a helpful assistant."

weave.init("SVx")

PROMPT_REFS = {
    "Step 1": "weave:///caspia-technologies/SVx/object/step_1:dCLtZ2iDazxnHhbYQhey9gfc7UAzs6UpUPcpVLsVldM",
    "Step 2": "<test>"
}

MODELS = {
    "OpenAI-GPT-4o": "openai:gpt-4o",
    "GROQ-Deepseek-Distil-Llama-3.3-70B": "groq:deepseek-r1-distill-llama-70b",
    "HF-Qwen2.5-72B-Instruct": "huggingface:Qwen/Qwen2.5-72B-Instruct",
    "Together-Mistral-Small-24B-Instruct": "mistralai/Mistral-Small-24B-Instruct-2501",
}

def process_file_content(file):
    """Process different file types and return formatted content"""
    content = file.read().decode()
    file_extension = file.name.split('.')[-1].lower()
    
    try:
        if file_extension == 'json':
            # Parse JSON and format it nicely
            json_content = json.loads(content)
            return f"File: {file.name} (JSON)\n{json.dumps(json_content, indent=2)}\n"
        elif file_extension in ['v', 'sv']:
            # Format Verilog/SystemVerilog files
            return f"File: {file.name} (Verilog/SystemVerilog)\n{content}\n"
        else:  # txt and other text files
            return f"File: {file.name} (Text)\n{content}\n"
    except json.JSONDecodeError:
        # Handle invalid JSON files
        return f"File: {file.name} (Invalid JSON - treating as text)\n{content}\n"
    except Exception as e:
        return f"File: {file.name} (Error processing file: {str(e)})\n"

def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens += -1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def ask(message, sys_message="You are a helpful agent.", model="groq:llama-3.2-3b-preview"):
    if model=="mistralai/Mistral-Small-24B-Instruct-2501":
        client = Together()
    else:
        client = ai.Client()
    
    context = f"{message}\n\nAdditional Context:\n{st.session_state.file_context}" if st.session_state.file_context else message
    
    messages = [
        {"role": "system", "content": sys_message},
        {"role": "user", "content": context}
    ]
    
    if "gpt" in model.lower():
        token_count = num_tokens_from_messages(messages)
        st.sidebar.write(f"Input tokens: {token_count}")
    
    with st.expander("View Full Prompt"):
        st.text_area("Complete prompt sent to model:", 
                    value=json.dumps(messages, indent=2),
                    height=300)
    
    # Disable streaming for Huggingface models
    use_streaming = "huggingface" not in model.lower()
    
    response = client.chat.completions.create(
        model=model, 
        messages=messages,
        stream=use_streaming
    )

    # Create a placeholder for the response
    response_placeholder = st.empty()
    
    # Handle streaming
    if hasattr(response, '__iter__'):
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'choices') and chunk.choices:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        # Update the formatted response in the placeholder
                        if "<step>" in message and "<purpose>" in message and "<prompt>" in message:
                            step = message.split("<step>")[1].split("</step>")[0]
                            purpose = message.split("<purpose>")[1].split("</purpose>")[0]
                            formatted_response = f"""
                            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                                <strong>Assistant ({model}):</strong>
                                <br><br>
                                <strong>Step {step}</strong><br>
                                <strong>Purpose:</strong> {purpose}<br>
                                <strong>Response:</strong><br>{full_response}▌
                            </div>
                            """
                        else:
                            formatted_response = f"""
                            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                                <strong>Assistant ({model}):</strong><br>{full_response}▌
                            </div>
                            """
                        response_placeholder.markdown(formatted_response, unsafe_allow_html=True)
        
        # Final update without the cursor
        if "<step>" in message and "<purpose>" in message and "<prompt>" in message:
            step = message.split("<step>")[1].split("</step>")[0]
            purpose = message.split("<purpose>")[1].split("</purpose>")[0]
            final_response = f"""
            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                <strong>Assistant ({model}):</strong>
                <br><br>
                <strong>Step {step}</strong><br>
                <strong>Purpose:</strong> {purpose}<br>
                <strong>Response:</strong><br>{full_response}
            </div>
            """
        else:
            final_response = f"""
            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                <strong>Assistant ({model}):</strong><br>{full_response}
            </div>
            """
        response_placeholder.markdown(final_response, unsafe_allow_html=True)
        return full_response
    else:
        # Non-streaming response
        response_text = response.choices[0].message.content
        if "<step>" in message and "<purpose>" in message and "<prompt>" in message:
            step = message.split("<step>")[1].split("</step>")[0]
            purpose = message.split("<purpose>")[1].split("</purpose>")[0]
            formatted_response = f"""
            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                <strong>Assistant ({model}):</strong>
                <br><br>
                <strong>Step {step}</strong><br>
                <strong>Purpose:</strong> {purpose}<br>
                <strong>Response:</strong><br>{response_text}
            </div>
            """
        else:
            formatted_response = f"""
            <div style='background-color: #00cc0020; padding: 10px; border-radius: 5px; margin-bottom: 10px'>
                <strong>Assistant ({model}):</strong><br>{response_text}
            </div>
            """
        response_placeholder.markdown(formatted_response, unsafe_allow_html=True)
        return response_text

def load_prompt(url):
    return weave.ref(url).get()

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Chat Settings")
    selected_model = st.selectbox("Choose AI Model", options=list(MODELS.keys()))
    selected_prompt_name = st.selectbox("Select Preset Prompt", options=list(PROMPT_REFS.keys()))
    
    if selected_prompt_name and st.button("Load Prompt"):
        prompt_text = load_prompt(PROMPT_REFS[selected_prompt_name])
        st.session_state.current_prompt = prompt_text.content

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
        
        # Show preview of processed files
        with st.expander("Preview processed files"):
            st.code(st.session_state.file_context)

# Main content area
st.subheader("Input")

# System Message Input
system_input = st.text_area(
    "System Message",
    value=st.session_state.system_message,
    height=100,
    key="system_input"
)

# User Message Input
user_input = st.text_area(
    "User Message",
    value=st.session_state.get('current_prompt', ''),
    height=200,
    key="user_input"
)

if selected_prompt_name and st.button("Load Prompt", key="load_prompt_button"):
    prompt_text = load_prompt(PROMPT_REFS[selected_prompt_name])
    if "<purpose>" in prompt_text.content and "<prompt>" in prompt_text.content:
        purpose = prompt_text.content.split("<purpose>")[1].split("</purpose>")[0]
        prompt = prompt_text.content.split("<prompt>")[1].split("</prompt>")[0]
        
        st.session_state.system_message = f"You are a helpful assistant. Your task is to {purpose}"
        st.session_state.current_prompt = prompt
        st.rerun()

if st.button("Send", key="send_button"):
    if user_input:
        # The ask function now handles all formatting and display
        response = ask(message=user_input, sys_message=system_input, model=MODELS[selected_model])

# Optional: Add a divider between input and response
st.markdown("---")