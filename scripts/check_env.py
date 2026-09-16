import sys
print("Python:", sys.executable)
try:
    import peft
    print("peft:", peft.__version__)
except ImportError:
    print("peft: NOT INSTALLED")

try:
    import transformers
    print("transformers:", transformers.__version__)
except ImportError:
    print("transformers: NOT INSTALLED")

try:
    import torch
    print("torch:", torch.__version__, "cuda available:", torch.cuda.is_available())
except ImportError:
    print("torch: NOT INSTALLED")
