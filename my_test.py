from typing import List, Optional
import fire
from llama import Dialog, Llama

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # Define dialogs with a specified template for sentiment analysis
    dialogs: List[Dialog] = [
        [
            {
                "role": "system",
                "content": (
                    "Analyze the sentiment of the following statement. "
                    "Return the response in this template:\n"
                    "Sentiment: [Positive/Negative/Neutral]\n"
                    "Confidence: [High/Medium/Low]\n"
                ),
            },
            {"role": "user", "content": "I love this movie!"},
        ],
        [
            {
                "role": "system",
                "content": (
                    "Analyze the sentiment of the following statement. "
                    "Return the response in this template:\n"
                    "Sentiment: [Positive/Negative/Neutral]\n"
                    "Confidence: [High/Medium/Low]\n"
                ),
            },
            {"role": "user", "content": "This is the worst experience ever."},
        ],
    ]

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # Display the results following the template
    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        response = result['generation']['content']
        print(f"> {result['generation']['role'].capitalize()}: {response}")

        # Check if the response follows the template
        if "Sentiment:" in response and "Confidence:" in response:
            print("Response follows the template.")
        else:
            print("Response does not follow the template.")

        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
