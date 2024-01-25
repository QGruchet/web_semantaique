# Import the libraries
from fastapi import FastAPI, Request
from transformers import pipeline
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create a FastAPI app
app = FastAPI()

# Create a class for the input data
class InputData(BaseModel):
 prompt: str

# Create a class for the output data
class OutputData(BaseModel):
 response: str

model = AutoModelForCausalLM.from_pretrained(
        "mlabonne/phixtral-2x2_8",
        load_in_4bit= True,
        device_map = 'auto',
        trust_remote_code=True
    )
tokenizer = AutoTokenizer.from_pretrained(
        "mlabonne/phixtral-2x2_8",
        trust_remote_code=True
    )
# Load a local LLM using Hugging Face Transformers
# You can change the model name and the task according to your needs
# For example, you can use “t5-base” for summarization or “bert-base-cased” for question answering
model = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create a route for the web application
@app.post("/generate", response_model=OutputData)
def generate(request: Request, input_data: InputData):
 # Get the prompt from the input data
 prompt = input_data.prompt

# Generate a response from the local LLM using the prompt
 response = model(prompt)[0]["generated_text"]

# Return the response as output data
 return OutputData(response=response)