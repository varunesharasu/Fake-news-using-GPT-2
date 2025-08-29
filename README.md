# Fake News Detection using GPT-2 (BERT)

## Overview
This project is a web-based application for detecting fake news using a fine-tuned BERT model. The app provides a simple interface for users to input news articles or statements and receive predictions on whether the content is real or fake, along with a confidence score.

## Features
- **State-of-the-art Model**: Utilizes a fine-tuned BERT model (`Pulk17/Fake-News-Detection`) for high-accuracy fake news classification.
- **User-friendly Interface**: Built with Gradio, offering a clean and modern web UI.
- **Confidence Score**: Displays the model's confidence in its prediction.
- **GPU Support**: Automatically uses GPU if available for faster inference.

## How It Works
1. **Input**: Users enter a news article or statement into the provided textbox.
2. **Processing**: The text is tokenized and passed to the BERT model for classification.
3. **Output**: The app displays whether the news is "Fake" or "Real" along with a confidence score.

## Model Details
- **Model Used**: [Pulk17/Fake-News-Detection](https://huggingface.co/Pulk17/Fake-News-Detection)
- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Task**: Sequence classification (binary: Fake vs Real)

## Installation
1. **Clone the repository**:
   ```cmd
   git clone https://github.com/varunesharasu/Fake-news-using-GPT-2.git
   cd Fake-news-using-GPT-2
   ```
2. **Install dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```

## Usage
1. **Run the application**:
   ```cmd
   python app.py
   ```
2. **Access the web interface**:
   - The app will launch in your default browser (usually at `http://localhost:7860`).
   - Enter a news article or statement and click "Detect" to see the result.

## File Structure
- `app.py`: Main application file containing the Gradio interface and model logic.
- `requirements.txt`: List of required Python packages.

## Example
```
Input: "The government has announced a new policy to reduce carbon emissions by 50% by 2030."
Output: 🟩 Real News (Confidence: 0.95)
```

## Notes
- The model is loaded from Hugging Face and may require internet access on first run.
- GPU is used automatically if available; otherwise, CPU is used.

## License
This project is for educational purposes. Please check the license of the model and dependencies for commercial use.

## Acknowledgements
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio](https://gradio.app/)
- [Pulk17/Fake-News-Detection Model](https://huggingface.co/Pulk17/Fake-News-Detection)
