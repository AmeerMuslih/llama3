import os
import random
from typing import List, Optional
import fire
from llama import Dialog, Llama
import torch
from torchtext.datasets import IMDB
from torch.utils.data import DataLoader
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def preprocess_data():
    # Load the IMDB dataset
    imdb_data = IMDB(split='test')
    reviews, labels = [], []
    for label, review in imdb_data:
        reviews.append(review)
        labels.append(label)
    return reviews, labels

def process_batch_on_gpu(generator, dialogs, max_gen_len, temperature, top_p, device, true_labels):
    """
    Process a batch of data on a specific GPU.
    """
    # Get predictions from the model on the specified GPU
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        device=device
    )

    correct_predictions = 0
    for result, true_label in zip(results, true_labels):
        response = result['generation']['content']
        #print(f"True label: {true_label}, Predicted sentiment: {response}")
        predicted_sentiment = 'Neutral'  # Default sentiment
        
        # Extract predicted sentiment from the response
        if "Positive" in response:
            predicted_sentiment = 2
        elif "Negative" in response:
            predicted_sentiment = 1
        
        if predicted_sentiment == true_label:
            correct_predictions += 1

    return correct_predictions, len(true_labels)

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    max_num_pred: int = 64,
    batch_size: int = 1,  # Batch size for processing
    batch_start_idx: int = 0,  # Start index for saving output files
    group_id: int = 0,  # Group ID for saving output files
    bitFlips_number: int = 500000
):
    start_time = time.time()
    os.environ['group_id'] = str(group_id)
    os.environ['bitFlips_number'] = str(bitFlips_number)
    # Check if GPUs are available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please ensure you have a GPU setup.")
    
    # Get available GPU devices
    num_workers = num_gpus = torch.cuda.device_count()
    print(f"BitFlips_number = {bitFlips_number}")

    # Load the model on each GPU
    generators = [
        Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=torch.device(f"cuda:{i}")
        ) for i in range(num_gpus)
    ]

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

    total_correct_predictions = 0
    total_predictions = 0

    # Create a ThreadPoolExecutor to parallelize the GPU work
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        for i, batch in enumerate(batches):
            if i < batch_start_idx:
                continue
            gpu_id = i % min(num_gpus, num_workers)  # Round-robin GPU selection
            generator = generators[gpu_id]
            device = torch.device(f"cuda:{gpu_id}")
            true_label_batch = true_labels[total_predictions+batch_start_idx*batch_size:total_predictions+batch_size*(batch_start_idx+1)]

            dialogs = []
            for review in batch:
                dialogs.append([
                    {
                        "role": "system",
                        "content": (
                            "Analyze the sentiment of the following movie review from the IMDB dataset. "
                            "Consider the overall tone, emotional language, and context to determine if the sentiment is Positive or Negative. "
                            "Return only 'Sentiment: Positive' or 'Sentiment: Negative', Never respond with 'Sentiment: Neutral'."
                        )
                    },
                    {"role": "user", "content": review},
                ])

            # Submit a batch to be processed in parallel on a GPU
            #print(f"Processing batch {i}")
            futures.append(
                executor.submit(process_batch_on_gpu, generator, dialogs, max_gen_len, temperature, top_p, device, true_label_batch)
            )
            total_predictions += len(batch)

            if total_predictions >= max_num_pred:
                break

        # Gather results as they complete
        for future in as_completed(futures):
            correct_predictions, processed_batch_size = future.result()
            total_correct_predictions += correct_predictions
            #print(f"Processed {processed_batch_size} predictions")

    # Calculate accuracy
    accuracy = total_correct_predictions / total_predictions
    print(f"Total predictions: {total_predictions}")
    print(f"Total correct predictions: {total_correct_predictions}")
    print(f"Accuracy on IMDB dataset: {accuracy * 100:.2f}%")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

if __name__ == "__main__":
    fire.Fire(main)
