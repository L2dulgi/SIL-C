import argparse
import warnings
from SILGym.config.kitchen_scenario import kitchen_scenario
from SILGym.config.variant_registry import resolve_kitchen_vis_variant
from SILGym.config.baseline_config import * # PTGMConfig, SILCConfig
from SILGym.trainer.skill_trainer import SkillTrainer
from SILGym.utils.logger import get_logger, set_experiment_path

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
        help='Environment (kitchen, kitchen_vis[-s|-l], mmworld, libero-b, libero-l, libero-s)',
        default='kitchen'
    )
    parser.add_argument(
        '-sc', '--scenario_type',
        type=str,
        help="Scenario type ['kitchenem', 'kitehcnex', 'skill']",
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
    parser.add_argument(
        '-dec', '--decoder',
        type=str,
        choices=['ddpm', 'diffusion', 'fql', 'flow'],
        help='Skill decoder architecture type',
        default='ddpm'
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
    
    # Evaluation noise parameters
    parser.add_argument(
        '--eval_noise',
        action='store_true',
        help='Enable noise during evaluation'
    )
    parser.add_argument(
        '--eval_noise_scale',
        type=float,
        default=0.01,
        help='Scale/magnitude of evaluation noise (Gaussian)'
    )
    parser.add_argument(
        '--eval_noise_clip',
        type=float,
        default=None,
        help='Optional clipping range for noisy observations'
    )
    parser.add_argument(
        '--eval_noise_seed',
        type=int,
        default=None,
        help='Random seed for evaluation noise reproducibility'
    )

    # Experiment configuration
    parser.add_argument(
        '-epoch', '--epoch',
        type=int,
        help='Number of experiment epcochs',
        default=None, 
    )
    parser.add_argument(
        '-dt', '--dist_type',
        type=str,
        choices=['maha', 'euclidean', 'cossim'],
        default='maha',
        help='Distance metric type for LazySI'
    )

    # Evaluation seed
    parser.add_argument(
        '-seed','--seed',
        type=int,
        help='Evaluation seed for reproducibility',
        default=0
    )

    # Action chunking parameters
    parser.add_argument(
        '--action_chunk',
        type=int,
        default=1,
        help='Number of actions to predict in a single forward pass (1=disabled, default: 1)'
    )
    parser.add_argument(
        '--action_chunk_padding',
        type=str,
        choices=['repeat_last', 'zero'],
        default='repeat_last',
        help='Padding mode for action chunks at trajectory end (default: repeat_last)'
    )

    return parser.parse_args()


from SILGym.config.mmworld_scenario import mmworld_scenario
from SILGym.config.libero_scenario import libero_scenario
from SILGym.config.variant_registry import LIBERO_ENV_MODEL_MAP
from SILGym.config.data_paths import DEFAULT_LIBERO_MODEL
def get_scenario_config(args):
    """Return the scenario configuration based on the provided environment."""
    env_key = args.env.lower()
    normalized_env = env_key.replace('_', '-')

    # Handle kitchen visual embedding variants (including kitchenstudio_vis)
    if (normalized_env.startswith('kitchen-vis') or normalized_env.startswith('kitchenvis') or
        normalized_env.startswith('kitchenstudio-vis') or normalized_env.startswith('kitchenstudio') or
        normalized_env.startswith('kitchen-studio')):
        variant = resolve_kitchen_vis_variant(env_key)
        return kitchen_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
            use_embeddings=True,
            embed_variant=variant,
            env_alias=env_key,
        )

    model_name = None

    if env_key in LIBERO_ENV_MODEL_MAP:
        model_name = LIBERO_ENV_MODEL_MAP[env_key]
        base_env = "libero"
    else:
        base_env = env_key

    if base_env == "kitchen":
        return kitchen_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
        )
    elif base_env == "mmworld":
        return mmworld_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
        )
    elif base_env == "libero":
        return libero_scenario(
            scenario_type=args.scenario_type,
            sync_type=args.sync_type,
            model_name=model_name or DEFAULT_LIBERO_MODEL,
            requested_env=env_key,
        )
    else:
        raise ValueError(f"Unknown environment: {args.env}")

def log_ml_backend_status(logger):
    """Log the ML backend configuration (cuML GPU acceleration status)."""
    try:
        from SILGym.utils.cuml_wrapper import get_backend, is_cuml_available
        cuml_enabled = is_cuml_available()
        backend_name = get_backend()

        logger.info("="*80)
        logger.info("Clustering Algorithms Backend Configuration")
        logger.info("="*80)
        if cuml_enabled:
            logger.info(f"✓ GPU Acceleration: ENABLED")
            logger.info(f"✓ Backend: {backend_name}")
            logger.info(f"✓ Algorithms using GPU: KMeans, t-SNE, UMAP")
            logger.info(f"✓ Expected speedup: 5-50x for clustering, 10-100x for manifold learning")
        else:
            logger.warning(f"⚠ GPU Acceleration: DISABLED")
            logger.warning(f"⚠ Backend: {backend_name}")
            logger.warning(f"⚠ Using CPU-based sklearn/umap-learn")
            logger.warning(f"⚠ To enable GPU: bash setup/python12/cuml.sh")
        logger.info("="*80)
    except Exception as e:
        logger.warning(f"Could not check cuML backend status: {e}")

def main():
    # Initialize logger
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("SILGym Training Started")
    logger.info("="*80)

    # Check and log ML backend configuration
    log_ml_backend_status(logger)

    # Suppress warnings if needed
    warnings.filterwarnings("ignore")

    args = parse_arguments()
    logger.info(f"Command line arguments: {vars(args)}")
    
    scenario_config = get_scenario_config(args)
    logger.info(f"Loaded scenario config: {scenario_config.scenario_type} ({scenario_config.sync_type})")

    # Map algorithm names to their configuration classes.
    algo_config_map = {
        # Type 1
        'ptgm': PTGMConfig,
        'buds': BUDSConfig,  
        # Type 2
        'iscil': IsCiLConfig,  
        'imanip': ImanipConfig,  
        'lazysi': LazySIConfig,
        'silc': SILCConfig,  
    }

    algo_config_cls = algo_config_map.get(args.algorithm.lower())
    if algo_config_cls is None:
        logger.error(f"Invalid algorithm type: {args.algorithm}")
        raise ValueError(f"Invalid algorithm type: {args.algorithm}")
    
    logger.info(f"Using algorithm: {args.algorithm}")
    logger.info(f"Using decoder type: {args.decoder}")

    # Initialize the experiment configuration
    init_kwargs = {
        'scenario_config': scenario_config,
        'exp_id': args.exp_id,
        'seed': args.seed,
        'lifelong_algo': args.lifelong,
        'decoder_type': args.decoder,
        'action_chunk': args.action_chunk,
        'action_chunk_padding': args.action_chunk_padding,
    }
    if algo_config_cls is LazySIConfig:
        init_kwargs['distance_type'] = args.dist_type
    elif algo_config_cls is SILCConfig:
        init_kwargs['distance_type'] = args.dist_type

    exp_config = algo_config_cls(**init_kwargs)
    
    # Set experiment path for logging
    set_experiment_path(exp_config.exp_save_path)
    logger.info(f"Experiment path: {exp_config.exp_save_path}")

    # Post update the experiment configuration
    # Update noise configuration if specified
    noise_update = {}
    if args.eval_noise:
        noise_update['eval_noise_enabled'] = True
        noise_update['eval_noise_scale'] = args.eval_noise_scale
        if args.eval_noise_clip is not None:
            noise_update['eval_noise_clip'] = args.eval_noise_clip
        if args.eval_noise_seed is not None:
            noise_update['eval_noise_seed'] = args.eval_noise_seed

    if noise_update:
        exp_config.update_config(**noise_update)
        logger.info(f"Evaluation noise enabled (Gaussian): {noise_update}")
    
    if args.epoch is not None :
        update_dict = {
            'phase_epochs': args.epoch,
        }
        exp_config.update_config(**update_dict)
        logger.info(f"Updated phase_epochs to {args.epoch}")

    # debug overwrite
    if args.debug:
        exp_config.phase_epochs = 1
        logger.warning("Debug mode: phase_epochs set to 1")

    # call the methods for the exp_config(skill_stream_config : algorithm method)
    # e.g. exp_config.set_skill_lifelong_algo(args.lifelong)

    # Create and run the SkillTrainer
    logger.info("Creating SkillTrainer...")
    trainer = SkillTrainer(
        experiment_config=exp_config,
        do_eval=args.do_eval,
    )
    try:
        logger.info("Starting continual training...")
        trainer.continual_train()
        logger.info("Continual training finished successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        trainer.close_logger()
        logger.info("Training session ended.")

if __name__ == "__main__":
    main()
