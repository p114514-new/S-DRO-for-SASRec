import argparse
from ast import arg

from recbole.quick_start import run_recbole, run_recboles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="DRO-S", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    config_dict = {'seed': 114514, 'stopping_step': 15, 'loss_type': 'CE', 'n_layers': 2,
                   'n_heads': 2, 'hidden_size': 64, 'inner_size': 256, 'hidden_dropout_prob': 0.5,
                   'attn_dropout_prob': 0.5, 'hidden_act': 'gelu', 'layer_norm_eps': 1e-12, 'initializer_range': 0.02,
                   'num_groups': 3,
                   'thresholds': [0.6, 0.7],  # must be ascending order, not given, needs tuning
                   'alpha': 0.1,  # not given, needs tuning
                   'eta': 1e-5,  # not given, needs tuning
                   }

    if args.nproc == 1 and args.world_size <= 0:
        run_recbole(
            model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=config_dict
        )
    else:
        if args.world_size == -1:
            args.world_size = args.nproc
        import torch.multiprocessing as mp

        mp.spawn(
            run_recboles,
            args=(
                args.model,
                args.dataset,
                config_file_list,
                args.ip,
                args.port,
                args.world_size,
                args.nproc,
                args.group_offset,
            ),
            nprocs=args.nproc,
        )
