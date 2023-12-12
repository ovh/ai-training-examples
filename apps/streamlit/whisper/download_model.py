import sys

# # Check if two command-line arguments are provided
if len(sys.argv) !=3:
    print("Usage: python main.py <whisper_model_id> <whisper_model_output_path>")
    print("Example: python main.py large-v3 /workspace/whisper-model/")
    sys.exit(1)

# Check if the model path ends with '/'
model_path = sys.argv[2]
if not model_path.endswith('/'):
    model_path += '/'
  
### Download the model in a local directory - Specify the version you want to use in the first parameter
import whisper
model_id = sys.argv[1]
model_path = f'{model_path}{model_id}'
# Available models = ['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large']

# The whisper module’s load_model() method loads a whisper model in your Python application. You must pass the model name as a parameter to the load_model() method.
try:
    model = whisper.load_model(model_id, download_root=model_path)
    print("Model has successfully been downloaded")
except Exception as e:
    print(f"Error downloading the model: {e}")
