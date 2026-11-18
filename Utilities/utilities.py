import os

def logs(message):
    if os.getenv("DEBUG", True):
        print(f"   \033[90m{message}\033[0m")
    