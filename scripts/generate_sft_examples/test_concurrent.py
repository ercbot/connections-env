#!/usr/bin/env python3
"""Test script to check if rollouts run concurrently"""

import asyncio
import os
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers import setup_logging

from connections.environment import ConnectionsEnv
from connections.dataset import prep_dataset
from connections.rulesets import get_ruleset_config


def create_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


async def main():
    setup_logging("INFO")

    print("Loading dataset...")
    train_sft_dataset = load_dataset("ericbotti/connections-puzzles", split="train_sft")

    print("Preprocessing dataset...")
    ruleset_config = get_ruleset_config("nyt")
    train_sft_dataset = prep_dataset(train_sft_dataset, ruleset_config)

    print("Creating ConnectionsEnv...")
    vf_env = ConnectionsEnv(
        ruleset="nyt",
        eval_dataset=train_sft_dataset,
    )

    print("Creating client...")
    client = create_client()

    print("\nRunning evaluation with max_concurrent=32...")
    results = await vf_env.evaluate(
        client=client,
        model="deepseek-chat",
        sampling_args={},
        num_examples=1,
        rollouts_per_example=5,
        max_concurrent=32,
        max_concurrent_generation=32,  # explicitly set
        max_concurrent_scoring=32,      # explicitly set
        interleave_scoring=True,
        state_columns=["guess_history", "complete_reason"],
    )

    print(f"\nDone! Processed {len(results.completion)} rollouts")


if __name__ == "__main__":
    asyncio.run(main())
