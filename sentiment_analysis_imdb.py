import random
from typing import List, Optional
import fire
from llama import Dialog, Llama
import torch
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
import time

def preprocess_data():
    # Load the IMDB dataset
    imdb_data = IMDB(split='test')
    reviews, labels = [], []
    for label, review in imdb_data:
        reviews.append(review)
        labels.append(label)
        #print(label)
    return reviews, labels

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    max_num_pred: int = 64,
    batch_size: int = 16,  # Batch size for processing
    batch_start_idx: int = 0,  # Start index for processing
):
    start_time = time.time()
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Preprocess IMDB data
    reviews, true_labels = preprocess_data()
    # Shuffle the data
    combined = list(zip(reviews, true_labels))
    random.seed(42)
    random.shuffle(combined)
    reviews, true_labels = zip(*combined)
    # Divide reviews into batches
    batches = [
        reviews[i : i + batch_size]
        for i in range(0, len(reviews), batch_size)
    ]

    correct_predictions = 0
    total_predictions = 0

    for i, batch in enumerate(batches):
        if i < batch_start_idx:
            continue
        dialogs = []
        for review in batch:
            dialogs.append([
                {
                    "role": "system",
                    "content": (
                        "Analyze the sentiment of the following movie review from the IMDB dataset. "
                        "Consider the overall tone, emotional language, and context to determine if the sentiment is Positive or Negative. "
                        "Return only 'Sentiment: Positive' or 'Sentiment: Negative'."
                    )
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
            
            print("Review: ", batch[total_predictions % batch_size])
            print(f"True label: {true_label}")
            print(f"Response: {response}")
            # Extract predicted sentiment from the response
            if "Positive" in response:
                predicted_sentiment = 2
            elif "Negative" in response:
                predicted_sentiment = 1
            
            if predicted_sentiment == true_label:
                correct_predictions += 1

            total_predictions += 1
        print(f"finished {total_predictions} predictions")
        if total_predictions >= max_num_pred:
            break

    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    print(f"Accuracy on IMDB dataset: {accuracy * 100:.2f}%")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    fire.Fire(main)
