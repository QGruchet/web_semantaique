from fastapi import FastAPI
import asyncio
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

from threading import Thread
import uvicorn
from queue import Queue

# Importing the TextStreamer class from transformers
from transformers import TextStreamer

# Defining a custom streamer which inherits the Text Streamer
class CustomStreamer(TextStreamer):

    def __init__(self, queue, tokenizer, skip_prompt, **decode_kwargs) -> None:
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        # Queue taken as input to the class
        self._queue = queue
        self.stop_signal=None
        self.timeout = 1
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Instead of printing the text, we add the text into the queue
        self._queue.put(text)
        if stream_end:
            # Similarly we add the stop signal also in the queue to 
            # break the stream
            self._queue.put(self.stop_signal)

app = FastAPI()

# Loading the model
base_model = "mistralai/Mistral-7B-v0.1"

model = AutoModelForCausalLM.from_pretrained(
    base_model,
        device_map = 'auto',
        trust_remote_code=True
    )
tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True
    )
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.model_max_length = 512
tokenizer.add_bos_token, tokenizer.add_eos_token

# Creating the queue
streamer_queue = Queue()

# Creating the streamer
streamer = CustomStreamer(streamer_queue, tokenizer, True)

# The generation process
def start_generation(query):

    prompt = """

            # You are assistant that behaves very professionally. 
            # You will extract relation between entity from text. The aim is to create RDF triples from text. 

            # ###Human: {instruction},
            # ###Assistant: """.format(instruction=query)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda:0")
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=64, temperature=0.1)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

# Generation initiator and response server
async def response_generator(query):

    start_generation(query)

    while True:
        value = streamer_queue.get()
        if value == None:
            break
        yield value
        streamer_queue.task_done()
        await asyncio.sleep(0.1)

@app.get('/query-stream/')
async def stream(query: str):
    print(f'Query receieved: {query}')
    return StreamingResponse(response_generator(query), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app)