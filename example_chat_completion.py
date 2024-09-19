# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

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
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs: List[Dialog] = [
        [
            {
                "role": "system",
                "content": "Respond with a funny joke to the following prompt.",
            },
            {"role": "user", "content": "How do you make a tissue dance?"},
        ],
        [
            {
                "role": "system",
                "content": "Respond like Shakespeare to the following prompt.",
            },
            {"role": "user", "content": "How is the weather today"},
        ],
        [
            {
                "role": "system",
                "content": "Respond with a funny joke to the following prompt.",
            },
            {"role": "user", "content": "What do you call a fake noodle?"},
        ],
        [
            {
                "role": "system",
                "content": "Respond with a funny joke to the following prompt.",
            },
            {"role": "user", "content": "Why couldn't the bicycle stand up by itself?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
