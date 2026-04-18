# Overview

Notes I am taking to keep the learnings organized

## How to implement Multi model tools to enable processing images, code attached?
Use custom tool class using BaseTool from langchain_core.tools
## How to handle the huggingface file endpoint failure?
Prepare a custom mapping of task_id : dir_file_path and
associate it with task_id : questions mapping by adding 
file_path parameter

## transcribe_audio tool error with finding file?
The error was in the ffmpeg dependency
do
```aiignore
conda install -c conda-forge ffmpeg
```