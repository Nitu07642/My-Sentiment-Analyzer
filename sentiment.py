# --- Sentiment Analyzer by Hugging Face ---

# Step 1: Zaroori cheez (pipeline) ko library se import karo
from transformers import pipeline

print("Sentiment model load ho raha hai... (Pehli baar time lagega)")

# Step 2: Apna 'Sentiment Analysis' pipeline (expert) taiyaar karo
# Hum ise bata rahe hain ki humein "sentiment-analysis" karna hai
sentiment_classifier = pipeline("sentiment-analysis")

print("Model load ho gaya! Ab aap test kar sakte hain.")
print("--------------------------------------------------")

# Step 3: User se ek English sentence maango
user_sentence = input("Aapka English sentence yahan likhein: ")

# Step 4: Sentence ko expert (pipeline) ke paas check karne ke liye bhejo
result = sentiment_classifier(user_sentence)

# Step 5: Result ko saaf tarike se dikhao
# 'result' ek list hoti hai, jisme dictionary hoti hai, jaise:
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Hum uss dictionary se 'label' (POSITIVE/NEGATIVE) nikaal rahe hain
label = result[0]['label']
# Hum 'score' (model kitna confident hai) bhi nikaal rahe hain
score = result[0]['score']

print(f"Result (Bhavna): {label}")
print(f"Confidence Score: {score:.4f} (Model itna sure hai)")