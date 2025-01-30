import matplotlib.pyplot as plt
import numpy as np
from transformers import pipeline

# List of pre-trained transformer models for sentiment analysis
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",  # DistilBERT
    "bert-base-uncased",                                # BERT
    "roberta-base",                                     # RoBERTa
    "nlptown/bert-base-multilingual-uncased-sentiment", # Multilingual BERT
    "cardiffnlp/twitter-roberta-base-sentiment"         # RoBERTa fine-tuned on Twitter data
]

# Sample social media posts in multiple languages
social_media_posts = [
    "I love this product! It's amazing!",  # English
    "HPV is one of main reason for cancer.",  # English
    "smoking Tobacco lead to cancer.",  # English
    "I'm so happy with the results!",  # English
    "Absolutely terrible, would not recommend.",  # English
    "Me encanta este producto! Es increíble!",  # Spanish
    "C'est le pire service que j'ai jamais eu.",  # French
    "この商品が大好きです！素晴らしい！",  # Japanese
    "이 제품을 정말 좋아해요! 놀라워요!",  # Korean
    "இந்த தயாரிப்பு அருமை! மிகுந்த மகிழ்ச்சி!",  # Tamil
    "यह उत्पाद बहुत अच्छा है! यह अद्भुत है!",  # Hindi
    "Das humane Papillomavirus (HPV) kann Krebs verursachen.",  # German
    "Le tabac est l'une des principales causes du cancer.",  # French
    "La dieta equilibrata aiuta a prevenire molte malattie.",  # Italian
    "Trop de soleil peut causer le cancer de la peau.",  # French
    "Rauchen erhöht das Krebsrisiko erheblich.",  # German
    "Una corretta alimentazione è essenziale per la salute.",  # Italian
    "L'esposizione al sole senza protection peut être dangereuse.",  # French
    "Il virus HPV è una delle principali cause di cancro cervicale.",  # Italian
    "HPV ist eine ernsthafte Bedrohung für die Gesundheit von Frauen.",  # German
    "Fumer nuit gravement à la santé et peut provoquer un cancer."  # French
]

# Dictionary to store model results
model_confidences = {model: [] for model in models}

# Loop through each model and analyze sentiment
for model_name in models:
    print(f"\nResults for model: {model_name}")
    print("=" * 50)
    
    # Load the sentiment analysis pipeline
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name)
    
    # Analyze sentiment for each post
    results = sentiment_analyzer(social_media_posts)
    
    # Store confidence scores and display results
    for post, result in zip(social_media_posts, results):
        print(f"Post: {post}")
        print(f"Sentiment: {result['label']}, Confidence: {result['score']:.4f}")
        print("-" * 50)
        model_confidences[model_name].append(result['score'])

# Plot confidence scores
plt.figure(figsize=(12, 6))
width = 0.15  # Bar width
x = np.arange(len(social_media_posts))

for i, model_name in enumerate(models):
    plt.bar(x + i * width, model_confidences[model_name], width, label=model_name)

plt.xlabel("Social Media Posts")
plt.ylabel("Confidence Score")
plt.title("Comparison of Transformer Model Confidence Scores")
plt.xticks(x + width, [f"Post {i+1}" for i in range(len(social_media_posts))], rotation=45)
plt.legend()
plt.show()
