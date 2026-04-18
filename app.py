# Login related imports
from dotenv import load_dotenv
from huggingface_hub import login
import huggingface_hub
from datasets import load_dataset
from huggingface_hub import snapshot_download
import os
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_KEY")
## Check if user has accepted GAIA terms
try:
    huggingface_hub.login(token=hf_token)
    huggingface_hub.whoami()  # Verify HF authentication
except Exception as e:
    print("[KNOW ISSUE]", e)
    try:
        os.system("hf auth login")
    except Exception as e2:
        print("[MSG] hf package not installed probably.", e2)
    print("[ERROR] User not logged in to Hugging Face Hub. Please log) in to access GAIA benchmark.")
    exit(1)

import gradio as gr
import requests
import pandas as pd
from typing import Optional, List  # IMPROVEMENT #3: For file attachment handling
import tempfile
import shutil
# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
temp_dir = tempfile.mkdtemp()
# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
# My code starts here---------------------------------------------------------------------------------------------------

"""
This module serves to integrate the LLM with the tools and initiate the main application
1. Make the state class
-- prepare the system prompt important
2. Create the nodes and the LLM
3. Create the stategraph
"""

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools.retriever import create_retriever_tool
from supabase.client import Client, create_client
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)


from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import ToolNode  # imp shortcut to make the tool node
from langgraph.graph import StateGraph, START, END
from retriever import CustomRetriever  # custom import I made be cautious
# custom tools made
from tools import (
    CustomRetrieverTool,
    wiki_search,
    web_search,
    arxiv_search,
    execute_code_multilang,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    transcribe_audio,  # IMPROVEMENT #2: Audio transcription tool
    save_and_read_file,
    download_file_from_url,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    analyze_image,
    transform_image
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


### === Initialization code === ###
print("[INIT] Initializing...")
# GAIA Manual Mapping : as Huggingface endpoint is giving 404, thanks!
# Download GAIA dataset and build task_id → file_path mapping
print("[INIT] Downloading GAIA dataset...")
data_dir = snapshot_download("gaia-benchmark/GAIA", repo_type="dataset")
print("[DEBUG] data_dir :", data_dir)
ds = load_dataset(data_dir, "2023_level1", split="validation")
GAIA_FILE_MAP = {}  # Global mapping
for ex in ds:
    if ex["file_path"] and ex["file_name"]:
        full_path = os.path.join(data_dir, ex["file_path"])
        print("[DEBUG] full_path :", full_path)
        if os.path.exists(full_path):
            GAIA_FILE_MAP[ex["task_id"]] = full_path
print(f"[INIT] GAIA dataset ready. {len(GAIA_FILE_MAP)} tasks with files.")

## Initialize retriever (do this once at startup)
# retriever_instance = CustomRetriever("sample_data.csv")  # sample file for testing
# vector_retriever = retriever_instance.run()
# print("[INIT] Retriever ready.")
# retriever_tool = CustomRetrieverTool(retriever=vector_retriever)
## Initialize the parser
parser = JsonOutputParser()
print("[INIT] Output parser ready.")


### ===Custom Agent Code=== ###
# LLM
# llm_instance = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
ans = input("Which model to use? (1: Qwen3-32B, 2: Gemini Flash Lite) ")
if ans=="1":
    llm_instance = ChatGroq(model="qwen/qwen3-32b", temperature=0)
else:
    llm_instance = ChatGoogleGenerativeAI(model="models/gemini-3-pro-preview", temperature=0)
tools = [
    wiki_search,
    web_search,
    arxiv_search,
    execute_code_multilang,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    transcribe_audio,
    save_and_read_file,
    download_file_from_url,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    analyze_image,
    transform_image
]
llm_instance = llm_instance.bind_tools(tools)  # tool binded llm ready
print("[INIT] Tool binded LLM ready.")

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    file_paths: Optional[List[str]]

# Nodes
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_text = f.read()
def llm(state: AgentState) -> AgentState:
    """Node to send the query to the llm"""
    sys_prompt = system_prompt_text
    message = state["messages"]
    response = llm_instance.invoke([SystemMessage(content=sys_prompt)] + list(message))
    # Poor performance code need improvement later
    # parse json if final answer
    # Only parse if it's the final answer (no tool calls)
    # if not hasattr(response, "tool_calls") or not response.tool_calls:
    #     try:
    #         parsed = parser.parse(response.content)
    #         if "Final Answer" in parsed:
    #             return {"messages": [AIMessage(content=str(parsed))]}
    #     except Exception:
    #         pass  # Not JSON yet, continue workflow
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    """Node to decide if to continue to call tools or end the process and give the final answer"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return True
    return False

# (disabled) Retriever node code
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"
# )  #  dim=768
# supabase: Client = create_client(
#     os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
# )
# vector_store = SupabaseVectorStore(
#     client=supabase,
#     embedding=embeddings,
#     table_name="documents2",
#     query_name="match_documents_2",
# )
# create_retriever_tool = create_retriever_tool(
#     retriever=vector_store.as_retriever(),
#     name="Question Search",
#     description="A tool to retrieve similar questions from a vector store.",
# )



# StateGraph
graph = StateGraph(AgentState)
graph.add_node("llm_node", llm)
# graph.add_node("should_continue_node", should_continue) (not needed check)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)  # tool node using prebuild ToolNode code
# Edges
graph.add_edge(START, "llm_node")
graph.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        True: "tool_node",
        False: END
    }
)
graph.add_edge("tool_node", "llm_node")  # reconnection
app = graph.compile()

# # test
# response = app.invoke(
#     {"messages": [HumanMessage(content="At what time I was doing 'Gym session' according to my notes ")]})
# print("\n\nFinal Response from Agent:\n", response["messages"][-1].content[0]['text'])

# My code ends here(huggingface template code states)-----------------------------------------------------------------------------------------------------

class BasicAgent:
    def __init__(self):
        print("BasicAgent initialized. Custom run...")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:100]}...")
        try:
            response = app.invoke({"messages": [HumanMessage(content=question)]})
            last_message = response["messages"][-1]

            # Extract text from AIMessage content
            if hasattr(last_message, 'content'):
                if isinstance(last_message.content, list):
                    fixed_answer = last_message.content[0].get('text', str(last_message.content))
                else:
                    fixed_answer = str(last_message.content)
            else:
                fixed_answer = str(last_message)

            print(f"Agent returning answer: {fixed_answer[:500]}(truncated output ends)")
            return fixed_answer
        except Exception as e:
            print(f"Error invoking agent graph: {e}")
            return f"Error: {str(e)}"

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # # NEW: 2.5 Download associated files for each question
    # if temp_dir:
    #     print("[MSG] temp_dir exists")# Check if temp_dir exists

    for item in questions_data:
        task_id = item.get("task_id")
        if task_id in GAIA_FILE_MAP:
            file_path = GAIA_FILE_MAP[task_id]
            item["file_path"] = file_path  # Add file path to question item
            print(f"Task {task_id} has associated file: {file_path}")
        else:
            item["file_path"] = ""
    print(f"[DEBUG] Status of questions_data after manual mapping : {questions_data[:2]}")
    # NOTE: map is questions_data { task_id, question, file_path_from_GAIA_FILE_MAP }
    # GAIA_FILE_MAP { task_id: file_path } task_id is the connecting link

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    # 3. Run your Agent (modified to use downloaded files)
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_path = item.get("file_path")  # NEW: Get downloaded file path
        try:
            # CRITICAL NOTE: new list is created each time, so file_paths does not persist between calls
            # Pass file to agent if available
            if len(file_path) > 0:
                submitted_answer = run_agent_on_question(question_text, [file_path])
            else:
                submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)

        # Cleanup code
        # Clean up temporary files after submission
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up {temp_dir}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

        return final_status, results_df

    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
# IMPROVEMENT: File attachment handling for GAIA benchmark questions
# This allows the agent to process questions with attached files (audio, images, CSV, etc.)
def run_agent_on_question(question: str, file_attachments: Optional[List[str]] = None) -> str:
    """
    Run the agent on a single question with optional file attachments.

    This function processes file attachments and prepends relevant context to the question
    before passing it to the agent. This enables the agent to handle GAIA benchmark questions
    that include various types of attachments.

    Args:
        question: The question text from the user
        file_attachments: List of file paths for attachments (can be None)

    Returns:
        str: The agent's response to the question
    """
    # Step 1: Initialize context string for file attachments
    attachment_context = ""

    if file_attachments:
        for file_path in file_attachments:
            # Validate file exists before processing
            if not os.path.exists(file_path):
                print(f"[File Handler] ⚠️ File not found: {file_path}")
                continue

            file_ext = os.path.splitext(file_path)[1].lower()
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            print(f"[File Handler] Processing {file_ext} file: {file_name}")
            print(f"[File Handler] Full path: {file_path}")  # NEW: Log full path

            # Add file metadata to context with FULL PATH
            attachment_context += f"\n--- FILE AVAILABLE ---\n"
            attachment_context += f"File name: {file_name}\n"
            attachment_context += f"File path: {file_path}\n"  # FIX: Pass full path
            attachment_context += f"File type: {file_ext}\n"
            attachment_context += f"File size: {file_size} bytes\n"

            # File-specific processing instructions
            if file_ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                attachment_context += f" Use transcribe_audio(file_path='{file_path}')\n"
            elif file_ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                attachment_context += f" Use extract_text_from_image(image_path='{file_path}')\n"
            elif file_ext in ['.csv']:
                attachment_context += f" Use analyze_csv_file(file_path='{file_path}')\n"
            elif file_ext in ['.xlsx', '.xls']:
                attachment_context += f" Use analyze_excel_file(file_path='{file_path}')\n"
            elif file_ext == '.py':
                # Read Python code and embed it
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    attachment_context += f"Python code:\n```python\n{code_content}\n```\n"
                except Exception as e:
                    attachment_context += f"[ERROR] Error reading Python file: {e}\nfile_path='{file_path}'\n"

            attachment_context += "--- END FILE ---\n"

    # Prepend file context to question
    enhanced_question = attachment_context + "\n\n" + question if attachment_context else question

    print(f"[File Handler] Enhanced question length: {len(enhanced_question)} chars")

    # Invoke agent with file paths in state
    result = app.invoke({
        "messages": [HumanMessage(content=enhanced_question)],
        "file_paths": file_attachments  # Pass file paths to agent state
    })

    # Extract response
    last_message = result["messages"][-1]
    if hasattr(last_message, 'content'):
        if isinstance(last_message.content, list):
            return last_message.content[0].get('text', str(last_message.content))
        return str(last_message.content)
    return str(last_message)


# --- Gradio Interface Helper Function ---
def process_with_files(question: str, files) -> str:
    """
    Wrapper function for Gradio interface to handle file uploads.

    Args:
        question: The user's question
        files: Gradio File component output (can be single file or list)

    Returns:
        str: The agent's answer
    """
    # Convert Gradio file objects to file paths
    file_paths = None
    if files:
        if isinstance(files, list):
            file_paths = [f.name if hasattr(f, 'name') else f for f in files]
        else:
            file_paths = [files.name if hasattr(files, 'name') else files]

    return run_agent_on_question(question, file_paths)


with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    # IMPROVEMENT #3: Add file upload interface for testing individual questions
    with gr.Tab("Test Agent with Files"):
        gr.Markdown("### Test your agent with a single question and optional file attachments")
        test_question = gr.Textbox(label="Question", lines=3, placeholder="Enter your question here...")
        test_files = gr.File(label="Attachments (optional)", file_count="multiple")
        test_button = gr.Button("Run Agent")
        test_output = gr.Textbox(label="Answer", lines=10)

        test_button.click(
            fn=process_with_files,
            inputs=[test_question, test_files],
            outputs=test_output
        )

    # Original evaluation interface
    with gr.Tab("Run Full Evaluation"):
        gr.LoginButton()

        run_button = gr.Button("Run Evaluation & Submit All Answers")

        status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
        # Removed max_rows=10 from DataFrame constructor
        results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

        run_button.click(
            fn=run_and_submit_all,
            outputs=[status_output, results_table]
        )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=True)