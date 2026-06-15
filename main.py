import os 
from dotenv import load_dotenv
import gradio as gr
from huggingface_hub import InferenceClient
import spaces
load_dotenv()

#demo=gr.Blocks()

client = InferenceClient(model="nvidia/Nemotron-Cascade-2-30B-A3B",api_key=os.environ["HF_Token"])

system_prompt = "You are an assistant by the name solver.ai that is a Grade 10 Level Expert in Mathematics and Science You dont have to introduce yourself and be up to the point\
and provides detail CBSE Board level answers to questions asked by students, you make sure the questions are relevent to the given syllabus or not.\
Also as you are an expert so you explain like an expert too, You first explain step by step stating whats given what we have to find and what is the formula used and give a step by step analysis for Mathematics\
And also suggest appropriate diagrams for students to draw\
For Mathematical Identities like Sin Square theta = 1 - Cos square theta just write the identity and its answer in this way sin square theta = 1 - cos square theta nothing extra You dont have to explain step by step analysis Do not do thinking for the identities\
For numerical questions in Science you follow a convention of first explaining as if you are answering like a teacher who explains very well and explains in such a way so that even weak students can understand too\
You also provide a step by step solution and dont miss even small steps so even weak students can understand \
Also you tend to get strict on anything which is outside of the curriculum as you are strictly meant for academic usage and specially to assist students only\
You are strictly not supposed to answer anything unrelated to Science and Mathematics of CBSE Grade 10 and bluntly refuse to answer stating the reason that your prime existance is to help in BEEE and not in any other subject or domain."
@spaces.GPU
def chat(message : dict ,history):
    history= history or []
    messages=[{"role":"system","content":system_prompt}]
    for item in history:
        if isinstance(item,dict):
           messages.append({"role":item["role"],"content":item["content"]}) 
        else:

            user_msg, assistant_msg = item[0],item[1]
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content":assistant_msg})
    messages.append({"role":"user","content":message})
    history.append((message, ""))    
    stream=client.chat_completion(messages=messages,stream=True,max_tokens=1024,temperature=0.4,top_p=0.9)
    result=""
    for chunk in stream:
        token=chunk.choices[0].delta.content or ""
        result+=token
        history[-1]=(message,result)
        yield result,history
    
    
    
with gr.Blocks(fill_height=True) as demo:
    history_state = gr.State([])
    gr.Markdown("Helping you solve High School Maths and Science")
    
    inp = gr.Textbox(placeholder="How can I help you today...",lines=2,max_lines=5,label="Input",submit_btn=True)
    
    with gr.Column(variant="panel") :
        out = gr.Textbox(lines=7,max_lines=10,label="Output")  
        submit_event = inp.submit(fn=chat,inputs=[inp,history_state],outputs=[out,history_state]) 

    submit_event.then(fn=lambda:"",inputs=None,outputs=inp)#prevent the issue where the text gets left after we press enter
demo.launch()

