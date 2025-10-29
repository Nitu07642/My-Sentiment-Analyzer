import gradio as gr
from transformers import pipeline

# 1. Load your pipeline (this will load just once)
print("Loading model...")
sentiment_classifier = pipeline("sentiment-analysis")
print("Model loaded!")

# 2. Create a function that will take input and give output
def analyze_sentiment(text):
    result = sentiment_classifier(text)
   # Extract the label and score from the result
    label = result[0]['label']
    score = result[0]['score']
   # Return the result in a good format
    return f"Bhavna (Sentiment): {label} \nConfidence: {score:.4f}"

# 3. Built to interface banao
app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Write your English sentence here..."),
    outputs="text",
    title="😊😐😡 Sentiment Analyzer",
    description="This tool analyzes your sentence to determine whether it's positive or negative (using the Hugging Face 'Distillbert' model)."
)

# 4. launch the App
app.launch()
