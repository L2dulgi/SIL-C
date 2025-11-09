import argparse
import warnings
import os
from SILGym.trainer.evaluator import SkillEvaluator
from SILGym.utils.logger import get_logger, set_experiment_path

def parse_arguments():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='SILGym evaluation script for evaluating trained models.'
    )
    
    # Required argument
    parser.add_argument(
        'exp_path',
        type=str,
        default="/home/meohee/SILGym/logs/kitchenstudio_vis-b/kitchenem/sync/ptgm_umaps40/append8/1030seed0",
        help='Path to experiment directory containing trained models'
    )
    
    # Evaluation mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['datastream', 'direct'],
        default='datastream',
        help='Evaluation mode: datastream (default) or direct config'
    )
    
    # Direct config mode arguments
    parser.add_argument(
        '--decoder_phase',
        type=str,
        help='Decoder phase ID for direct evaluation (e.g., "pre_0")'
    )
    parser.add_argument(
        '--interface_phase',
        type=str,
        help='Interface phase ID for direct evaluation (e.g., "pre_0")'
    )
    parser.add_argument(
        '--policy_phase',
        type=str,
        help='Policy phase ID for direct evaluation (e.g., "policy_2/pre_1")'
    )
    parser.add_argument(
        '--eval_tasks',
        type=str,
        nargs='+',
        help='Evaluation task names for direct mode (e.g., mbls kettle)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=3,
        help='Number of evaluation episodes per task'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=280,
        help='Maximum steps per evaluation episode'
    )
    
    # Noise parameters
    parser.add_argument(
        '--eval_noise',
        action='store_true',
        help='Enable noise during evaluation'
    )
    parser.add_argument(
        '--eval_noise_type',
        type=str,
        choices=['gaussian', 'uniform'],
        default='gaussian',
        help='Type of noise to add during evaluation'
    )
    parser.add_argument(
        '--eval_noise_scale',
        type=float,
        default=0.01,
        help='Scale/magnitude of evaluation noise'
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
    
    # Output options
    parser.add_argument(
        '--output_suffix',
        type=str,
        default='',
        help='Suffix to add to output filenames (e.g., "noise_0.05")'
    )
    parser.add_argument(
        '--output_postfix',
        type=str,
        default='post',
        help='Postfix for the output filename (default: "post", results in "eval_results_post.json")'
    )
    
    # Agent override options
    parser.add_argument(
        '--force_static',
        action='store_true',
        help='Force use of static agent instead of default agent class'
    )
    
    # Remote evaluation settings
    parser.add_argument(
        '--remote_host',
        type=str,
        default='127.0.0.1',
        help='Remote evaluation server host'
    )
    parser.add_argument(
        '--remote_port',
        type=int,
        default=9999,
        help='Remote evaluation server port'
    )
    
    return parser.parse_args()

def validate_direct_config_args(args):
    """Validate arguments for direct config mode."""
    if args.mode == 'direct':
        if not all([args.decoder_phase, args.interface_phase, args.policy_phase]):
            raise ValueError("Direct mode requires --decoder_phase, --interface_phase, and --policy_phase")
        if not args.eval_tasks:
            raise ValueError("Direct mode requires --eval_tasks")

def main():
    # Initialize logger
    logger = get_logger(__name__)
    logger.info("="*80)
    logger.info("SILGym Evaluation Started")
    logger.info("="*80)
    
    # Suppress warnings if needed
    warnings.filterwarnings("ignore")
    
    args = parse_arguments()
    logger.info(f"Command line arguments: {vars(args)}")
    
    # Validate arguments
    validate_direct_config_args(args)
    
    # Set experiment path for logging
    set_experiment_path(args.exp_path)
    
    # Build output suffix if noise is enabled and no custom suffix provided
    if args.eval_noise and not args.output_suffix:
        args.output_suffix = f"noise_{args.eval_noise_type}_{args.eval_noise_scale}"
        logger.info(f"Auto-generated output suffix: {args.output_suffix}")
    
    # Create evaluator
    logger.info("Creating SkillEvaluator...")
    evaluator = SkillEvaluator(
        exp_save_path=args.exp_path,
        remote_eval_host=args.remote_host,
        remote_eval_port=args.remote_port,
        eval_num_episodes=args.num_episodes,
        eval_max_steps=args.max_steps,
        eval_noise_enabled=args.eval_noise,
        eval_noise_type=args.eval_noise_type,
        eval_noise_scale=args.eval_noise_scale,
        eval_noise_clip=args.eval_noise_clip,
        eval_noise_seed=args.eval_noise_seed,
        output_suffix=args.output_suffix,
        output_postfix=args.output_postfix,
        force_static=args.force_static
    )
    
    try:
        if args.mode == 'datastream':
            logger.info("Starting datastream evaluation...")
            results = evaluator.evaluate_from_datastream()
            logger.info("Datastream evaluation completed successfully!")
        else:  # direct mode
            logger.info("Starting direct config evaluation...")
            # Build agent config
            agent_config = {
                'decoder': (0, args.decoder_phase),
                'interface': (0, args.interface_phase),
                'policy': (0, args.policy_phase)
            }
            # Build eval tasks
            eval_tasks = [{'data_name': task} for task in args.eval_tasks]
            
            results = evaluator.evaluate_direct_config(agent_config, eval_tasks)
            logger.info("Direct evaluation completed successfully!")
            
        # Print summary
        if results:
            logger.info("\nEvaluation Summary:")
            if args.mode == 'datastream':
                for phase, phase_results in results.items():
                    if phase_results.get('overall_avg_reward') is not None:
                        logger.info(f"Phase {phase}: avg_reward = {phase_results['overall_avg_reward']:.3f}")
            else:
                logger.info(f"Overall average reward: {results.get('overall_avg_reward', 0):.3f}")
                
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}", exc_info=True)
        raise
    finally:
        logger.info("Evaluation session ended.")

if __name__ == "__main__":
    main()