import cProfile
import pstats
import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import parse_arguments, setup_logger, main_gym  # Import necessary functions

def run_main_gym():
    args = parse_arguments()
    logger = setup_logger(log_dir=args.log_dir, log_name=args.exp)
    main_gym(args, logger)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_main_gym()  # Run the main_gym function

    profiler.disable()
    # Save the stats to a file
    stats_file = 'profile_stats_z3.txt'
    with open(stats_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f).sort_stats('cumtime')
        stats.print_stats()

    print(f"Profiling stats saved to {stats_file}")
