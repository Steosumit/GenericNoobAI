import os
from tools import transcribe_audio

print("Current working directory:", os.getcwd())
if os.path.exists(r".\local_audio.mp3"):
    print("Path exists!")
else:
    print("Path does not exist!")
r = transcribe_audio.invoke({"file_path": r".\local_audio.mp3"})
print(r)