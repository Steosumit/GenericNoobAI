"""
This file contains the code for the tools the agent will be using
1. retriever_tool
2. web_search_tool
"""

# Extra imports
import base64
import cmath
import io
import uuid
from typing import Optional, Dict, Any
import os
import tempfile
import numpy as np
import requests
from urllib.parse import urlparse
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import pytesseract
import pandas as pd
# Old imports
from dotenv import load_dotenv
from langchain_core.tools import tool, BaseTool
from langchain_core.vectorstores import VectorStoreRetriever  # imported for type validation in tool (security)
from langchain_community.tools import DuckDuckGoSearchRun
from oauthlib.uri_validate import query
from pydantic import Field
from code_interpreter import CodeInterpreter  # code interpreter import
# Tool imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
import whisper
from retriever import CustomRetriever  # custom import

# Initialization code
load_dotenv()
interpreter_instance = CodeInterpreter()

# Global variable to store Whisper model (lazy loading for performance)
_whisper_model = None



# 1. Retriever Tool: tool to retrieve relevant information from user's notes
# New approach
class CustomRetrieverTool(BaseTool):
    name: str = "retriever_tool"
    description : str = """
    Tool to retrieve the most relevant information about the user's notes
    Args:
        query (str): The user's query to search in the notes
        retriever (VectorStoreRetriever): The VectorStoreRetriever instance to use for retrieval
    Returns:
        str: The retrieved information from the notes most relevant to the query OR no relevant info found
    """
    retriever : VectorStoreRetriever = Field(exclude=True)
    def _run(self, query: str) -> str:
        results = []
        try:
            response = self.retriever.invoke(query)
            if not response:
                return "No relevant information found in the notes."
            for i, result in enumerate(response):
                results.append(f"Document {i + 1}:\n{result.page_content}\n")
            return "\n\n".join(results)
        except Exception as e:
            return f"[retrieval_tool Error]: {str(e)}"

# 2. Web Search Tool: tool to perform web searches
# @tool
# def web_search_tool(query: str) -> str:
#     """This tool is helpful to search the internet or web for more information when a question is asked"""
#     search_tool = DuckDuckGoSearchRun()
#     results = search_tool.invoke(query)
#     print("[web_search_tool] run")
#     return str(results)

# # 3. Add a vision tools for multimodal inputs (images) (later)
# class ImageDataExtractorTool(BaseTool):
#     name: str = "image_data_extractor_tool"
#     description: str = """
#         Tool to extract information from images using a vision model.
#
#         Args:
#             image_path (str): Path to the image file to analyze base64 string
#
#         Returns:
#             str: Description or analysis of the image content
#         """
#     vision_model: object = Field(exclude=True)
#
#     def _is_base64(self, s: str) -> bool:
#         """Check if a string is base64 encoded."""
#         import re
#         # Base64 pattern: alphanumeric characters, +, /, and optional = padding
#         base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
#         if not base64_pattern.match(s):
#             return False
#         try:
#             import base64
#             # Try to decode and re-encode to verify
#             base64.b64decode(s, validate=True)
#             return True
#         except Exception:
#             return False
#
#     def _run(self, image_path: str) -> str:
#         import base64
#         from langchain_core.messages import HumanMessage
#         try:
#             # Read and encode the image
#             if image_path.startswith('data:image'):
#                 # Extract base64 data from data URL
#                 image_data = image_path.split(',')[1] if ',' in image_path else image_path
#             elif self._is_base64(image_path):
#                 # Already base64 encoded
#                 image_data = image_path
#             else:
#                 with open(image_path, "rb") as image_file:
#                     image_data = base64.b64encode(image_file.read()).decode("utf-8")
#
#             # Create the message with image content
#             message = HumanMessage(
#                 content=[
#                     {
#                         "type": "text",
#                         "text": "Describe the following image in detail."
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": f"data:image/png;base64,{image_data}"
#                     }
#                 ]
#             )
#             # Invoke the vision model
#             response = self.vision_model.invoke([message])
#             return str(response.content)
#
#         except Exception as e:
#             return f"[image_data_extractor_tool Error]: {str(e)}"

# Additional tools
### =============== BROWSER TOOLS =============== ###


### ===Browser Search Tools=== ###
@tool
def wiki_search(query: str) -> dict[str, str]:
    """Search Wikipedia for a query and return maximum 2 results.
    Args:
        query: The search query."""
    print("[wiki_search_tool] run")
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    print(f"[wiki_search_tool] {formatted_search_docs[:500]}")
    return {"wiki_results": formatted_search_docs}


@tool
def web_search(query: str) -> dict[str, str]:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    print("[web_search_tool] run")
    search_docs = TavilySearchResults(max_results=3).invoke(query)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.get("url", "")}" title="{doc.get("title", "")}"/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    print(f"[web_search_tool] {formatted_search_docs[:500]}(truncated output ends)")
    return {"web_results": formatted_search_docs}



@tool
def arxiv_search(query: str) -> dict[str, str]:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    print("[arxiv_search_tool] run")
    search_docs = ArxivLoader(query=query).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:5000]}\n</Document>'
            for doc in search_docs
        ]
    )
    print(f"[arxiv_search_tool] {formatted_search_docs[:500]}")
    return {"arxiv_results": formatted_search_docs}


### ===Code Execution Tool=== ###
@tool
def execute_code_multilang(code: str, language: str = "python") -> str:
    """Execute code in multiple languages (Python, Bash, SQL, C, Java) and return results.
    Args:
        code (str): The source code to execute.
        language (str): The language of the code. Supported: "python".
    Returns:
        A string summarizing the execution results (stdout, stderr, errors, plots, dataframes if any).
    """
    print(f"[execute_code_multilang_tool] run - Language: {language}")

    # Input validation
    if not code or not code.strip():
        return "❌ Error: No code provided for execution"

    supported_languages = ["python"]
    language = language.lower()

    if language not in supported_languages:
        return f"❌ Unsupported language: {language}. Supported languages are: {', '.join(supported_languages)}"

    try:
        # Execute code with error handling
        result = interpreter_instance.execute_code(code, language=language)

        # Validate result structure
        if not isinstance(result, dict):
            return f"❌ Error: Invalid response from code interpreter (expected dict, got {type(result)})"

        response = []

        if result.get("status") == "success":
            response.append(f"✅ Code executed successfully in **{language.upper()}**")

            if result.get("stdout"):
                stdout_text = result["stdout"].strip()
                # Truncate very long outputs
                if len(stdout_text) > 10000:
                    stdout_text = stdout_text[:10000] + "\n... (output truncated)"
                response.append(
                    f"\n**Standard Output:**\n```\n{stdout_text}\n```"
                )

            if result.get("stderr"):
                stderr_text = result["stderr"].strip()
                if len(stderr_text) > 5000:
                    stderr_text = stderr_text[:5000] + "\n... (output truncated)"
                response.append(
                    f"\n**Standard Error (if any):**\n```\n{stderr_text}\n```"
                )

            if result.get("result") is not None:
                result_str = str(result["result"]).strip()
                if len(result_str) > 5000:
                    result_str = result_str[:5000] + "\n... (output truncated)"
                response.append(
                    f"\n**Execution Result:**\n```\n{result_str}\n```"
                )

            if result.get("dataframes"):
                for df_info in result["dataframes"]:
                    response.append(
                        f"\n**DataFrame `{df_info.get('name', 'unnamed')}` (Shape: {df_info.get('shape', 'unknown')})**"
                    )
                    if df_info.get("head"):
                        try:
                            df_preview = pd.DataFrame(df_info["head"])
                            response.append(f"First 5 rows:\n```\n{str(df_preview)}\n```")
                        except Exception as df_error:
                            response.append(f"Error displaying DataFrame: {str(df_error)}")

            if result.get("plots"):
                plot_count = len(result["plots"])
                response.append(
                    f"\n**Generated {plot_count} plot(s)** (Image data returned separately)"
                )

        else:
            response.append(f"❌ Code execution failed in **{language.upper()}**")

            if result.get("error"):
                error_msg = str(result["error"]).strip()
                if len(error_msg) > 5000:
                    error_msg = error_msg[:5000] + "\n... (error truncated)"
                response.append(f"\n**Error:**\n```\n{error_msg}\n```")

            if result.get("stderr"):
                stderr_text = result["stderr"].strip()
                if len(stderr_text) > 5000:
                    stderr_text = stderr_text[:5000] + "\n... (output truncated)"
                response.append(
                    f"\n**Error Log:**\n```\n{stderr_text}\n```"
                )

        return "\n".join(response)

    except Exception as e:
        error_message = f"❌ Unexpected error during code execution: {str(e)}"
        print(f"[execute_code_multilang_tool] Error: {error_message}")
        return error_message


### ===Methematical tools=== ###

@tool
def multiply(a: float, b: float) -> float:
    """
    Multiplies two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    print("[multiply_tool] run")
    return a * b


@tool
def add(a: float, b: float) -> float:
    """
    Adds two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    print("[add_tool] run")
    return a + b


@tool
def subtract(a: float, b: float) -> float:
    """
    Subtracts two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    print("[subtract_tool] run")
    return a - b
    return a - b


@tool
def divide(a: float, b: float) -> float:
    """
    Divides two numbers.
    Args:
        a (float): the first float number
        b (float): the second float number
    """
    print("[divide_tool] run")
    if b == 0:
        raise ValueError("Cannot divided by zero.")
    return a / b


@tool
def modulus(a: int, b: int) -> int:
    """
    Get the modulus of two numbers.
    Args:
        a (int): the first number
        b (int): the second number
    """
    print("[modulus_tool] run")
    return a % b


@tool
def power(a: float, b: float) -> float:
    """
    Get the power of two numbers.
    Args:
        a (float): the first number
        b (float): the second number
    """
    print("[power_tool] run")
    return a**b


@tool
def square_root(a: float) -> float | complex:
    """
    Get the square root of a number.
    Args:
        a (float): the number to get the square root of
    """
    print("[square_root_tool] run")
    if a >= 0:
        return a**0.5
    return cmath.sqrt(a)

### ===Audio Processing Tool=== ###
@tool
def transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio files (mp3, wav, m4a) to text using Whisper.

    Audio transcription capability
    - Enables the agent to process audio files from GAIA benchmark questions
    - Uses OpenAI Whisper model for speech-to-text conversion
    - Supports common audio formats: .mp3, .wav, .m4a

    Args:
        file_path (str): full path to the audio file to transcribe
    Returns:
        str: Transcribed text from the audio file
    """
    print("[transcribe_audio_tool] run")
    global _whisper_model
    try:
        # Step 1: Verify file exists
        if not os.path.exists(file_path):
            return f"Error: Audio file not found at path: {file_path}"

        # Step 2: Load Whisper model (lazy loading - only load once)
        # Using 'base' model for balance between speed and accuracy
        # Options: tiny, base, small, medium, large
        if _whisper_model is None:
            print("[transcribe_audio_tool] Loading Whisper model (first time only)...")
            _whisper_model = whisper.load_model("base")
            print("[transcribe_audio_tool] Whisper model loaded successfully")

        # Step 3: Transcribe the audio file
        print(f"[transcribe_audio_tool] Transcribing: {os.path.basename(file_path)}")
        result = _whisper_model.transcribe(file_path)

        # Step 4: Extract and return the transcribed text
        transcribed_text = result['text'].strip()
        print(f"[transcribe_audio_tool] Transcription complete. Length: {len(transcribed_text)} chars")

        return f"Transcribed audio from '{os.path.basename(file_path)}':\n\n{transcribed_text}"

    except Exception as e:
        error_msg = f"Error transcribing audio file '{os.path.basename(file_path)}': {str(e)}"
        print(f"[transcribe_audio_tool] {error_msg}")
        return error_msg

### ===File Processing Tools=== ###
@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    print("[save_and_read_file_tool] run")
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."

@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    print("[download_file_from_url_tool] run")
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    print("[extract_text_from_image_tool] run")
    try:
        # Open the image
        image = Image.open(image_path)

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"

@tool
def analyze_csv_file(file_path: str, query: str) -> str:
    """
    Analyze a CSV file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the CSV file.
        query (str): Question about the data
    """
    print("[analyze_csv_file_tool] run")
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Run various analyses based on the query
        result = f"CSV file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing CSV file: {str(e)}"

@tool
def analyze_excel_file(file_path: str, query: str) -> str:
    """
    Analyze an Excel file using pandas and answer a question about it.
    Args:
        file_path (str): the path to the Excel file.
        query (str): Question about the data
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path)

        # Run various analyses based on the query
        result = (
            f"Excel file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
        )
        result += f"Columns: {', '.join(df.columns)}\n\n"

        # Add summary statistics
        result += "Summary statistics:\n"
        result += str(df.describe())

        return result

    except Exception as e:
        return f"Error analyzing Excel file: {str(e)}"

### Image Processing Tools ###
# Helper Functions
def encode_image(image_path: str) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
def decode_image(base64_string: str) -> Image.Image:
    """Convert a base64 string to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))
def save_image(image: Image.Image, directory: str = "image_outputs") -> str:
    """Save a PIL Image to disk and return the path."""
    os.makedirs(directory, exist_ok=True)
    image_id = str(uuid.uuid4())
    image_path = os.path.join(directory, f"{image_id}.png")
    image.save(image_path)
    return image_path

# Actual tool function
@tool
def analyze_image(image_base64: str) -> Dict[str, Any]:
    """
    Analyze basic properties of an image (size, mode, color analysis, thumbnail preview).
    Args:
        image_base64 (str): Base64 encoded image string
    Returns:
        Dictionary with analysis result
    """
    print("[analyze_image_tool] run")
    try:
        img = decode_image(image_base64)
        width, height = img.size
        mode = img.mode

        if mode in ("RGB", "RGBA"):
            arr = np.array(img)
            avg_colors = arr.mean(axis=(0, 1))
            dominant = ["Red", "Green", "Blue"][np.argmax(avg_colors[:3])]
            brightness = avg_colors.mean()
            color_analysis = {
                "average_rgb": avg_colors.tolist(),
                "brightness": brightness,
                "dominant_color": dominant,
            }
        else:
            color_analysis = {"note": f"No color analysis for mode {mode}"}

        thumbnail = img.copy()
        thumbnail.thumbnail((100, 100))
        thumb_path = save_image(thumbnail, "thumbnails")
        thumbnail_base64 = encode_image(thumb_path)

        return {
            "dimensions": (width, height),
            "mode": mode,
            "color_analysis": color_analysis,
            "thumbnail": thumbnail_base64,
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def transform_image(
    image_base64: str, operation: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply transformations: resize, rotate, crop, flip, brightness, contrast, blur, sharpen, grayscale.
    Args:
        image_base64 (str): Base64 encoded input image
        operation (str): Transformation operation
        params (Dict[str, Any], optional): Parameters for the operation
    Returns:
        Dictionary with transformed image (base64)
    """
    try:
        img = decode_image(image_base64)
        params = params or {}

        if operation == "resize":
            img = img.resize(
                (
                    params.get("width", img.width // 2),
                    params.get("height", img.height // 2),
                )
            )
        elif operation == "rotate":
            img = img.rotate(params.get("angle", 90), expand=True)
        elif operation == "crop":
            img = img.crop(
                (
                    params.get("left", 0),
                    params.get("top", 0),
                    params.get("right", img.width),
                    params.get("bottom", img.height),
                )
            )
        elif operation == "flip":
            if params.get("direction", "horizontal") == "horizontal":
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif operation == "adjust_brightness":
            img = ImageEnhance.Brightness(img).enhance(params.get("factor", 1.5))
        elif operation == "adjust_contrast":
            img = ImageEnhance.Contrast(img).enhance(params.get("factor", 1.5))
        elif operation == "blur":
            img = img.filter(ImageFilter.GaussianBlur(params.get("radius", 2)))
        elif operation == "sharpen":
            img = img.filter(ImageFilter.SHARPEN)
        elif operation == "grayscale":
            img = img.convert("L")
        else:
            return {"error": f"Unknown operation: {operation}"}

        result_path = save_image(img)
        result_base64 = encode_image(result_path)
        return {"transformed_image": result_base64}

    except Exception as e:
        return {"error": str(e)}


