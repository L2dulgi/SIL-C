import argparse
import warnings
from AppOSI.config.kitchen_scenario import kitchen_scenario
from AppOSI.config.baseline_config import * # PTGMConfig
from AppOSI.trainer.skill_trainer import SkillTrainer

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='skill incremental learning algorithm trainer.'
    )
    # Debug mode flag
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Set experiment to debug mode'
    )
    parser.add_argument(
        '-id', '--exp_id',
        type=str,
        nargs='?',
        const='',
        help='Experiment ID e.g.',
        default='',
    )
    # Scenario configuration arguments
    parser.add_argument(
        '-e', '--env',
        type=str,
        help='Environment e.g. kitchen',
        default='kitchen'
    )
    parser.add_argument(
        '-sc', '--scenario_type',
        type=str,
        help="Scenario type ['objective', 'quality', 'skill']",
        default='objective'
    )
    parser.add_argument(
        '-st', '--sync_type',
        type=str,
        help="Synchronization type ['sync', 'async']",
        default='sync'
    )
    # Algorithm configuration arguments
    parser.add_argument(
        '-al', '--algorithm',
        type=str,
        help='Skill incremental learning algorithm name',
        default='ptgm'
    )
    parser.add_argument(
        '-ll', '--lifelong',
        type=str,
        help='Decoder lifelong learning name',
        default='append'
    )
    # Evaluation parameters
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        '--do_eval',
        dest='do_eval',
        action='store_true',
        help='Perform remote evaluation after each phase'
    )
    eval_group.add_argument(
        '--no_eval',
        dest='do_eval',
        action='store_false',
        help='Skip remote evaluation after each phase'
    )
    parser.set_defaults(do_eval=True)

    # Experiment configuration
    parser.add_argument(
        '-epoch', '--epoch',
        type=int,
        help='Number of experiment epcochs',
        default=None, 
    )
    parser.add_argument(
        '-rank', '--rank',
        type=int,
        help='Rank of the appender',
        default=16, # 4is too small
    )

    parser.add_argument(
        '-dt', '--dist_type',
        type=str,
        choices=['maha', 'euclidean', 'cossim'],
        default='maha',
        help='Distance metric type for LazySI'
    )

    # Evaulation seed   
    parser.add_argument(
        '-seed','--seed',
        type=int,
        help='Evaluation seed for reproducibility',
        default=0
    )
    return parser.parse_args()


from AppOSI.config.mmworld_scenario import mmworld_scenario
from AppOSI.config.libero_scenario import libero_scenario
def get_scenario_config(args):
    """Return the scenario configuration based on the provided environment."""
    if args.env.lower() == "kitchen":
        return kitchen_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
        )
    elif args.env.lower() == "mmworld":
        return mmworld_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
        )
    elif args.env.lower() == "libero":
        return libero_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

def main():
    # Suppress warnings if needed
    warnings.filterwarnings("ignore")

    args = parse_arguments()
    scenario_config = get_scenario_config(args)

    # Map algorithm names to their configuration classes.
    algo_config_map = {
        # Type 1
        'ptgm': PTGMConfig,
        'buds': BUDSConfig,  
        # Type 2
        'iscil': IsCiLConfig,  
        'imanip': ImanipConfig,  
        'assil': AsSILConfig,  
        'lazysi': LazySIConfig,  
    }

    algo_config_cls = algo_config_map.get(args.algorithm.lower())
    if algo_config_cls is None:
        raise ValueError(f"Invalid algorithm type: {args.algorithm}")
    
    print(f"Using algorithm: {args.algorithm}")

    # Initialize the experiment configuration
    init_kwargs = {
        'scenario_config': scenario_config,
        'exp_id': args.exp_id,
        'seed': args.seed,
        'lifelong_algo': args.lifelong,
    }
    if algo_config_cls is LazySIConfig:
        init_kwargs['distance_type'] = args.dist_type

    exp_config = algo_config_cls(**init_kwargs)

    # Post updatae the experiment configuration TODO
    
    if args.epoch is not None :
        update_dict = {
            'phase_epochs': args.epoch,
        }
        exp_config.update_config(**update_dict)

    # debug overwrite
    if args.debug:
        exp_config.phase_epochs = 1

    # call the methods for the exp_config(skill_stream_config : algorithm method)
    # e.g. exp_config.set_skill_lifelong_algo(args.lifelong)

    # Create and run the SkillTrainer
    trainer = SkillTrainer(
        experiment_config=exp_config,
        do_eval=args.do_eval,
    )
    try:
        trainer.continual_train()
        print("[Test] Continual training finished. Check any saved models or logs.")
    finally:
        trainer.close_logger()

if __name__ == "__main__":
    main()