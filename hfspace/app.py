import os
import time
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import gradio as gr
from threading import Thread

MODEL_LIST = ["LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"]
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MODEL = os.environ.get("MODEL_ID")

TITLE = """
<h1><center>EXAONE-3.0-7.8B-Instruct</center></h1>
<center>
<p>The model is licensed under EXAONE AI Model License Agreement 1.1 - NC</p>
</center>
"""

PLACEHOLDER = """
<center>
<p>EXAONE-3.0-7.8B-Instruct is a pre-trained and instruction-tuned bilingual (English and Korean) generative model with 7.8 billion parameters</p>
</center>
"""


CSS = """
.duplicate-button {
    margin: auto !important;
    color: white !important;
    background: black !important;
    border-radius: 100vh !important;
}
h3 {
    text-align: center;
}
"""

device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    ignore_mismatched_sizes=True)

@spaces.GPU()
def stream_chat(
    message: str, 
    history: list,
    system_prompt: str,
    temperature: float = 1.0, 
    max_new_tokens: int = 1024, 
    top_p: float = 1.0, 
    top_k: int = 50,
):
    print(f'message: {message}')
    print(f'history: {history}')

    conversation = [{"role": "system", "content": system_prompt}]
    for prompt, answer in history:
        conversation.extend([
            {"role": "user", "content": prompt}, 
            {"role": "assistant", "content": answer},
        ])

    conversation.append({"role": "user", "content": message})

    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)
    
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    
    generate_kwargs = dict(
        input_ids=inputs, 
        max_new_tokens = max_new_tokens,
        do_sample = False if temperature == 0 else True,
        top_p = top_p,
        top_k = top_k,
        temperature = temperature,
        streamer=streamer,
        pad_token_id = 0,
        eos_token_id = 361, # 361
    )

    with torch.no_grad():
        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()
        
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

            
chatbot = gr.Chatbot(height=600, placeholder=PLACEHOLDER)

with gr.Blocks(css=CSS, theme="soft") as demo:
    gr.HTML(TITLE)
    gr.DuplicateButton(value="Duplicate Space for private use", elem_classes="duplicate-button")
    gr.ChatInterface(
        fn=stream_chat,
        chatbot=chatbot,
        fill_height=True,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=False, render=False),
        additional_inputs=[
            gr.Textbox(
                value="You are EXAONE model from LG AI Research, a helpful assistant.",
                label="System Prompt",
                render=False,
            ),
            gr.Slider(
                minimum=0,
                maximum=1,
                step=0.1,
                value=1,
                label="Temperature",
                render=False,
            ),
            gr.Slider(
                minimum=128,
                maximum=4096,
                step=1,
                value=1024,
                label="Max new tokens",
                render=False,
            ),
            gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=1.0,
                label="top_p",
                render=False,
            ),
            gr.Slider(
                minimum=1,
                maximum=50,
                step=1,
                value=50,
                label="top_k",
                render=False,
            ),
        ],
        examples=[
            ["Help me study vocabulary: write a sentence for me to fill in the blank, and I'll try to pick the correct option."],
            ["What are 5 creative things I could do with my kids' art? I don't want to throw them away, but it's also so much clutter."],
            ["Explain who you are"],
            ["너의 소원을 말해봐"],
        ],
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
