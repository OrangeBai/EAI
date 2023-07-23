import pandas as pd
import wandb


def retrieve_train_acc(run):
    df = pd.DataFrame(list(run.scan_history()))
    df["epoch"] += 1
    valid_df = df[~df["train/top1"].isna()]
    df_plot = pd.DataFrame({
        name: df.sort_values(by="_step").iloc[-1] for name, df in valid_df.groupby("epoch")
    }).T
    return df_plot


def retrieve_val_acc(run):
    df = run.history(keys=["val/top1"], x_axis="epoch")
    df["epoch"] += 1
    return df


if __name__ == "__main__":
    api = wandb.Api(timeout=120)
    runs = api.runs(
        "orangebai/EAI_comp_act",
        filters={"config.bn": True, "config.init": True, "config.net": "vgg16"}
    )
    xx = {run.config["name"]: retrieve_train_acc(run) for run in runs}
