"""
Test file for transcribe_audio_tool from tools.py
Tests audio transcription functionality with various scenarios
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from tools import transcribe_audio
import shutil

class TestTranscribeAudioTool:
    """Test cases for transcribe_audio tool"""

    # Copy file to working directory with simple name
    def test_with_local_file(self):
        """Test with file copied to local directory"""
        original = r"C:\Users\steos\.cache\huggingface\hub\datasets--gaia-benchmark--GAIA\snapshots\682dd723ee1e1697e00360edccf2366dc8418dd9\2023\validation\99c9cc74-fdc8-46c6-8f8d-3ce2d3bfeea3.mp3"
        local_copy = r".\local_audio.mp3"

        # Copy to working directory

        shutil.copy(original, local_copy)
        print("[transcribe_audio_tool] Copied to local directory")
        # except:
        #     print("Copying failed!")
        try:
            result = transcribe_audio.invoke({"file_path": local_copy})
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error during transcription(COPY AND USE METHOD): {e}")
        finally:
            os.unlink(local_copy)

if __name__ == "__main__":
    if os.path.exists(r".\local_audio.mp3"):
        print("Path exists!")
    else:
        print("Path does not exist!")
    r = transcribe_audio.invoke({"file_path": r".\local_audio.mp3"})
    #pytest.main([__file__, "-v"])
