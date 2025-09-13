from datasets import Dataset, DatasetDict

connections_starting_prompt = """\
Words: {words}

You have found 0 categories.
You have 4 mistakes remaining.
"""


def prep_dataset(dataset_dict: DatasetDict) -> tuple[Dataset, Dataset | None]:
    """
    Internal function to prepare the default HuggingFace dataset for the Connections environment.
    
    Converts the raw dataset format:
    - all_words: list of words
    - group_words: list of word lists
    - group_themes: list of theme descriptions

    To the format expected by the environment:
    - question: formatted starting prompt with words
    - info: {categories: list of {group: theme, members: word_list}}

    Args:
        dataset_dict: DatasetDict from HuggingFace with train/test splits

    Returns:
        tuple[Dataset, Dataset | None]: (train_dataset, eval_dataset)
    """

    def format_dataset(example):
        # Convert group_words and group_themes to categories format
        categories = []
        for words_list, theme in zip(example["group_words"], example["group_themes"]):
            categories.append({"group": theme, "members": words_list})

        return {
            "question": connections_starting_prompt.format(
                words=", ".join(example["all_words"])
            ),
            "info": {"categories": categories},
        }

    # Extract train and test splits from DatasetDict
    train_dataset = None
    eval_dataset = None

    if "train" in dataset_dict:
        train_dataset = dataset_dict["train"].map(format_dataset, num_proc=1)

    if "test" in dataset_dict:
        eval_dataset = dataset_dict["test"].map(format_dataset, num_proc=1)

    # If no train split, use the first available split as train
    if train_dataset is None:
        first_split = list(dataset_dict.keys())[0]
        train_dataset = dataset_dict[first_split].map(format_dataset, num_proc=1)

    return train_dataset, eval_dataset
