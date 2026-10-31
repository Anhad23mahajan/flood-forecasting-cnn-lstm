# Load the Hugging Face token from Colab's secrets (key icon on the left), if there is one
try:
    from google.colab import userdata
    HF_TOKEN = userdata.get("HF_TOKEN")
    print("Token found")
except Exception:
    HF_TOKEN = None
    print("No token found, downloading without one")
