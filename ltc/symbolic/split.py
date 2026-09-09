"""The half/half agent split shared by the distillation paths.

The split is written once, as <prefix>.split.json, and both the forest and the
symbolic fit read it back. That keeps the two models trained on exactly the same
agents while leaving them free to be refit independently of each other.
"""

import argparse
import json

import pandas as pd


def split_agents(agents: list[int]) -> tuple[list[int], list[int]]:
    """Split the sorted agent ids into a train half and a held-out test half."""
    agents = sorted(agents)
    half = len(agents) // 2
    return agents[:half], agents[half:]


def write_split(df: pd.DataFrame, path: str) -> tuple[list[int], list[int]]:
    train_agents, test_agents = split_agents(df["agent"].unique().tolist())
    with open(path, "w") as f:
        json.dump(
            {"train_agents": train_agents, "test_agents": test_agents}, f, indent=2
        )
    return train_agents, test_agents


def load_split(path: str) -> tuple[list[int], list[int]]:
    with open(path) as f:
        split = json.load(f)
    return split["train_agents"], split["test_agents"]


def train_frame(df: pd.DataFrame, split_path: str) -> pd.DataFrame:
    """The rows of the train half, as both distillation paths see them."""
    train_agents, test_agents = load_split(split_path)
    print(f"Train agents: {train_agents}")
    print(f"Held-out test agents: {test_agents}")
    return df[df["agent"].isin(train_agents)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the train/test agent split of a history CSV."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to .csv file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .json path. Defaults to <csv without extension>.split.json",
    )
    args = parser.parse_args()

    output = args.output or f"{args.file.removesuffix('.csv')}.split.json"
    train_agents, test_agents = write_split(pd.read_csv(args.file), output)
    print(f"Train agents: {train_agents}")
    print(f"Held-out test agents: {test_agents}")
    print(f"Saved: {output}")
