from typing import List, Optional
import fire
from llama import Dialog, Llama
import torch
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader

def preprocess_data():
    # Load the IMDB dataset
    imdb_data = list(IMDB(split='test'))
    reviews, labels = zip(*imdb_data)
    
    # Convert labels to binary format
    label_mapping = {'pos': 'Positive', 'neg': 'Negative'}
    labels = [label_mapping[label] for label in labels]
    
    return reviews, labels

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    batch_size: int = 16,  # Batch size for processing
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Preprocess IMDB data
    reviews, true_labels = preprocess_data()

    # Divide reviews into batches
    batches = [
        reviews[i : i + batch_size]
        for i in range(0, len(reviews), batch_size)
    ]

    correct_predictions = 0
    total_predictions = 0

    for batch in batches:
        dialogs = []
        for review in batch:
            dialogs.append([
                {
                    "role": "system",
                    "content": (
                        "Analyze the sentiment of the following statement. "
                        "Return the response in this template:\n"
                        "Sentiment: [Positive/Negative/Neutral]\n"
                        "Confidence: [High/Medium/Low]\n"
                    ),
                },
                {"role": "user", "content": review},
            ])
        
        # Get predictions from the model
        results = generator.chat_completion(
            dialogs,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Evaluate the results
        for result, true_label in zip(results, true_labels[total_predictions:total_predictions+batch_size]):
            response = result['generation']['content']
            predicted_sentiment = 'Neutral'  # Default sentiment
            
            # Extract predicted sentiment from the response
            if "Positive" in response:
                predicted_sentiment = "Positive"
            elif "Negative" in response:
                predicted_sentiment = "Negative"
            
            if predicted_sentiment == true_label:
                correct_predictions += 1

            total_predictions += 1

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on IMDB dataset: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    fire.Fire(main)
