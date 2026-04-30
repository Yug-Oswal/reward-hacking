"""
Dataset loading and train/eval splitting utilities.
"""

from datasets import load_dataset as hf_load_dataset
from sklearn.model_selection import train_test_split


def load_dataset(dataset_name="longtermrisk/school-of-reward-hacks"):
    """
    Load a HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset identifier.

    Returns:
        The 'train' split of the dataset.
    """
    ds = hf_load_dataset(dataset_name)["train"]
    return ds


def make_train_eval_splits(ds, test_size=0.15, seed=42):
    """
    Split on question indices (not sample indices).

    Args:
        ds: The dataset.
        test_size: Fraction to hold out for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_q, eval_q) — lists of question indices.
    """
    all_q = list(range(len(ds)))
    train_q, eval_q = train_test_split(
        all_q, test_size=test_size, random_state=seed, shuffle=True
    )
    return train_q, eval_q


def map_q_to_hidden_indices(hidden_data, train_q, eval_q):
    """
    Map question indices to hidden_data sample indices.

    Each question produces two samples in hidden_data (hack + control),
    so this maps from question-level splits to sample-level indices.

    Args:
        hidden_data: Dict with 'groups' key (array of question indices).
        train_q: List of training question indices.
        eval_q: List of evaluation question indices.

    Returns:
        Tuple of (train_indices, eval_indices).
    """
    groups = hidden_data["groups"]
    train_set = set(train_q)
    eval_set = set(eval_q)

    train_indices = [i for i, g in enumerate(groups) if g in train_set]
    eval_indices = [i for i, g in enumerate(groups) if g in eval_set]

    return train_indices, eval_indices


def get_phase1_splits(ds, hidden_data, test_size=0.15, seed=42):
    """
    Convenience function: split questions then map to hidden_data indices.

    Returns:
        Tuple of (train_q, eval_q, train_indices, eval_indices).
    """
    train_q, eval_q = make_train_eval_splits(ds, test_size=test_size, seed=seed)
    train_indices, eval_indices = map_q_to_hidden_indices(
        hidden_data, train_q, eval_q
    )

    print(f"Train questions: {len(train_q)}, Eval questions: {len(eval_q)}")
    print(f"Train samples (hack+control): {len(train_indices)}, "
          f"Eval samples: {len(eval_indices)}")

    return train_q, eval_q, train_indices, eval_indices
