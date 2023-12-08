# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Optional

import fire
import os
import re

from llama import Llama, Dialog


def main(
    ckpt_dir: str = "/home/dataset/llama2/llama-2-models/llama-2-7b-chat/",
    tokenizer_path: str = "/home/dataset/llama2/llama-2-models/tokenizer.model",
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512 * 4,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    print_results: Optional[bool] = False,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 512.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 8.
        max_gen_len (int, optional): The maximum length of generated sequences. If None, it will be
            set to the model's max sequence length. Defaults to None.
    """
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompt = "You are training an autonomous vehicle in a series of ten environments to minimize the costs resulting from violations, such as going offtrack or overspeeding. In a series of ten environments, the weights learned from training in the one environment are used as the basis for starting training in the following environment. The environments are named Week8_01, Week8_02, Week8_03, Week8_04, Week9_01, Week9_02, Week9_03, Week9_03, Week9_04, Week10_01, and Week10_02.\n You are given the costs of ordering the environments by heuristics including number of obstacles (cost of -144.94), number of vehicles (cost of 45.61), and number of other autonomous vehicles (cost of -144.94).\nThe cost of running experiments on all ten environments in alphabetical and numerical order is 60.87. The cost on running an experiment on the same environment ten times is 110.66 for Week8_01, 43.49 for Week8_02, 43.49 for Week8_03, 43.49 for Week8_04, 106.74 for Week9_01, 106.74 for Week9_02, 109.90 for Week9_03, 88.76 for Week9_04, 43.49 for Week10_01, and 43.49 for Week10_02. The features of each environment are given below:\nWeek8_01 has num_obstacles = 0, num_vehicles = 2, num_auto_vehicles = 0\nWeek8_02 has num_obstacles = 0, num_vehicles = 1, num_auto_vehicles = 0\nWeek8_03 has num_obstacles = 0, num_vehicles = 1, num_auto_vehicles = 0\nWeek8_04 has num_obstacles = 0, num_vehicles = 3, num_auto_vehicles = 0\nWeek9_01 has num_obstacles = 1, num_vehicles = 0, num_auto_vehicles = 0\nWeek9_02 has num_obstacles = 1, num_vehicles = 0, num_auto_vehicles = 0\nWeek9_03 has num_obstacles = 1, num_vehicles = 0, num_auto_vehicles = 0\nWeek9_04 has num_obstacles = 2, num_vehicles = 0, num_auto_vehicles = 0\nWeek10_01 has num_obstacles = 0, num_vehicles = 0, num_auto_vehicles = 1\nWeek10_02 has num_obstacles = 0, num_vehicles = 1, num_auto_vehicles = 1\nGive an ordering of ten environments as a Python list to minimize the total cost. You can train on any combination of the ten environments, including using an environment multiple times or not at all."
    prompt += " Note that environment ordering does matter since weights are transfered sequentially."
    dialogs = [[
        {"role": "user", "content": prompt},
    ]]

    for i in range(10):
        results = generator.chat_completion(
            dialogs=dialogs,  # type: ignore
            temperature=temperature,
            max_gen_len=2048 * 2,
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
