

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load fine-tuned BERT model for fake news detection
bert_tokenizer = AutoTokenizer.from_pretrained("Pulk17/Fake-News-Detection")
bert_model = AutoModelForSequenceClassification.from_pretrained(
    "Pulk17/Fake-News-Detection"
).to(device)

# Function to detect if news is fake or real
def detect_news(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    label = "🟥 Fake News" if predicted_class == 0 else "🟩 Real News"
    return f"{label} (Confidence: {confidence:.2f})"

# Gradio Interface (Light Theme)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray", neutral_hue="gray", font=["Inter", "sans-serif"])) as demo:
    gr.Markdown("""
    <div style='text-align:center;'>
        <h2 style='color:#222;'>📰 Fake News Detector</h2>
    </div>
    """)
    with gr.Row():
        detect_input = gr.Textbox(
            label="Enter a News Article or Statement",
            placeholder="Paste a paragraph to detect if it's fake or real...",
            lines=6,
            elem_id="news-input"
        )
    detect_btn = gr.Button("Detect", elem_id="detect-btn")
    detect_output = gr.Textbox(label="Detection Result", elem_id="result-box")
    detect_btn.click(detect_news, inputs=detect_input, outputs=detect_output)

# Launch the app
demo.launch()


