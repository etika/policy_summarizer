# app.py
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ----------------------------
# CPU-friendly model
# ----------------------------
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# ----------------------------
# Summarization function
# ----------------------------
def summarize_policy(pdf_text):
    if not pdf_text.strip():
        return "Please provide some text to summarize."
    
    # Prepend instruction for T5 model
    input_text = f"summarize: {pdf_text}"
    
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return summary

# ----------------------------
# Gradio interface
# ----------------------------
iface = gr.Interface(
    fn=summarize_policy,
    inputs=gr.Textbox(lines=15, placeholder="Paste policy text here..."),
    outputs="text",
    title="Policy Summarizer",
    description="Paste government policy text to get a short summary. CPU-friendly version."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)

