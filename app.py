import gradio as gr
from transformers import pipeline

# 1. Apna pipeline load karo (yeh bas ek baar load hoga)
print("Model load ho raha hai...")
sentiment_classifier = pipeline("sentiment-analysis")
print("Model load ho gaya!")

# 2. Ek function banao jo input lega aur output dega
def analyze_sentiment(text):
    result = sentiment_classifier(text)
    # Result se label aur score nikaalo
    label = result[0]['label']
    score = result[0]['score']
    # Result ko ek achhe format mein return karo
    return f"Bhavna (Sentiment): {label} \nConfidence: {score:.4f}"

# 3. Gradio ka interface banao
# gr.Interface ko batao:
# - fn: Kaun sa function call karna hai (hamara 'analyze_sentiment')
# - inputs: Input kaisa hoga ("text" box)
# - outputs: Output kaisa dikhega ("text" box)
# - title: App ka title
# - description: App ke baare mein chhoti jaankari
app = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Yahan apna English sentence likhein..."),
    outputs="text",
    title="😊😐😡 Sentiment Analyzer",
    description="Yeh tool aapke sentence ko analyze karke batata hai ki woh POSITIVE hai ya NEGATIVE. (Hugging Face 'distilbert' model ka istemaal karke)"
)

# 4. App ko launch karo
app.launch()