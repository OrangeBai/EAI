import argparse
import sys


class TrainParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = sys.argv[1:]

        self.parser.add_argument("--dataset", type=str, required=True)
        self.parser.add_argument("--net", type=str, required=True)
        self.parser.add_argument("--project", type=str, required=True)
        self.parser.add_argument("--name", type=str, required=True)

        self.parser.add_argument("--batch_size", type=int, default=256)
        self.parser.add_argument("--bn", type=bool, default=True)
        self.parser.add_argument("--no_bn", dest="bn", action="store_false")
        self.parser.add_argument("--init", type=bool, default=True)
        self.parser.add_argument("--no_init", dest="init", action="store_false")
        self.parser.add_argument("--act", type=str, default="relu")

        # step-wise or epoch-wise
        self.parser.add_argument("--num_workers", default=4, type=int)
        self.parser.add_argument("--num_epoch", default=160, type=int)
        # scheduler and optimizer
        self.parser.add_argument(
            "--lr_scheduler",
            default="milestones",
            choices=["milestones", "exp", "cyclic", "static"],
        )
        self.parser.add_argument("--optimizer", default="SGD", choices=["SGD", "Adam"])
        self.parser.add_argument("--momentum", default=0.9, type=float)
        self.parser.add_argument("--lr", default=0.1, type=float)
        # training settings
        self.parser.add_argument("--npbar", default=True, action="store_false")

        # prune settings

    def get_args(self):
        args = self.parser.parse_known_args(self.args)[0]
        return args
