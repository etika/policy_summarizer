import gradio as gr
from transformers import pipeline
import pdfplumber

# Load a summarization pipeline
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # device=-1 for CPU

def summarize_pdf(file):
    text = ""
    with pdfplumber.open(file.name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

demo = gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(file_types=[".pdf"]),
    outputs="text",
    title="Policy Summarizer",
    description="Upload a PDF policy document and get a concise summary using CPU-only AI."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)

