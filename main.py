
import os
import threading

from dotenv import load_dotenv
import gradio as gr
import spaces
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

load_dotenv()


MODEL_ID = os.getenv("MODEL_ID", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16")
HF_TOKEN = os.getenv("HF_Token") or os.getenv("HF_TOKEN")


system_prompt = (
    "You are an assistant by the name solver.ai that is a Grade 10 Level Expert in Mathematics and Science. "
    "You do not have to introduce yourself and should get to the point. "
    "Provide detailed CBSE board level answers to questions asked by students. "
    "Make sure the questions are relevant to the given syllabus. "
    "Explain like an expert. For Mathematics, first explain step by step by stating what is given, what we have to find, "
    "and what formula is used. Also suggest appropriate diagrams for students to draw. "
    "For mathematical identities like sin^2(theta) = 1 - cos^2(theta), just write the identity and its answer in this way: "
    "sin^2(theta) = 1 - cos^2(theta). Do not add extra explanation. Do not do step-by-step thinking for identities. "
    "For numerical questions in Science, explain like a very good teacher and include every small step so even weak students "
    "can understand. "
    "Be strict about anything outside the curriculum. You are meant for academic usage only and especially to assist students. "
    "You must refuse anything unrelated to CBSE Grade 10 Mathematics and Science and state that your purpose is to help in "
    "CBSE Grade 10 Mathematics and Science, not any other subject or domain."
)

tokenizer = None
model = None


def load_model():
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
        )

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    return model, tokenizer


def build_messages(message, history):
    messages = [{"role": "system", "content": system_prompt}]

    for item in history or []:
        if isinstance(item, dict):
            messages.append({"role": item["role"], "content": item["content"]})
        else:
            user_msg, assistant_msg = item[0], item[1]
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

    messages.append({"role": "user", "content": message})
    return messages


@spaces.GPU(duration=120)
def chat(message: str, history):
    history = history or []
    model, tokenizer = load_model()

    messages = build_messages(message, history)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "temperature": 0.4,
        "top_p": 0.9,
        "do_sample": True,
    }

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})
    result = ""

    for token in streamer:
        result += token
        history[-1]["content"] =  result
        # Yielding history directly maps perfectly to gr.Chatbot
        yield history 

    thread.join()


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Helping you solve High School Maths and Science")

    # Replaced gr.Textbox with gr.Chatbot to handle the history state correctly in the UI
    chatbot = gr.Chatbot(label="Chat History")
    
    inp = gr.Textbox(
        placeholder="How can I help you today...",
        lines=2,
        max_lines=5,
        label="Input",
    )

    # Submit event updates the chatbot interface directly
    submit_event = inp.submit(
        fn=chat,
        inputs=[inp, chatbot],
        outputs=[chatbot],
    )

    # Clear input box after submission
    submit_event.then(fn=lambda: "", inputs=None, outputs=inp)


if __name__ == "__main__":
    demo.launch()
