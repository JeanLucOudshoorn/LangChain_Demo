import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# bart-large-mnli -- 'facebook/bart-large-mnli': 407M params
# 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli': 279M params


def load_model(model_name='facebook/bart-large-mnli'):
    """
    Function to load the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def classify_text(input_str, candidate_labels, tokenizer, model):
    """
    Function to classify text.
    """
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)
    start_time = time.time()
    outputs = classifier(input_str, candidate_labels)
    end_time = time.time()
    return outputs, end_time - start_time


# Load the model
tokenizer, model = load_model()

# Classify text
input_str = """

    Subject: Question about my parcel

    Beste sir/madam,

    I would like to know when I will receive my parcel. The order number is 123456.

    Looking forward to your answer!

    Kind regards,

    Mister X.
"""

candidate_labels = ['What is the delivery date?',
                    'What is the transportation method?',
                    'What is the name of the customer?',
                    'What is the delivery address?',
                    'I have a complaint.']

output_str, inference_time = classify_text(input_str, candidate_labels, tokenizer, model)
for l, s in list(zip(output_str['labels'], output_str['scores'])):
    print(l, round(s, 2))

print(f"Inference time: {round(inference_time, 1)} seconds")
