import gradio as gr
from transformers import pipeline
import pdfplumber

# Use CPU-only pipeline
summarizer = pipeline("summarization", device=-1)  # device=-1 ensures CPU

def summarize_policy(pdf_file):
    text = ""
    # Extract text from PDF
    with pdfplumber.open(pdf_file.name) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    if len(text.strip()) == 0:
        return "No text found in PDF."

    # HuggingFace summarizer has max token limits; chunk if needed
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
    summary = ""
    for chunk in chunks:
        res = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += res[0]['summary_text'] + "\n"

    return summary.strip()

# Gradio interface
iface = gr.Interface(
    fn=summarize_policy,
    inputs=gr.File(label="Upload Policy PDF"),
    outputs=gr.Textbox(label="Policy Summary"),
    title="Policy Summarizer",
    description="Upload a government policy PDF, get a concise summary (CPU-friendly)."
)

if __name__ == "__main__":
    # Render expects 0.0.0.0 and PORT environment variable
    import os
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)

