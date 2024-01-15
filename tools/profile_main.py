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
    main_gym(args, logger, train=args.train)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    run_main_gym()  # Run the main_gym function

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')  # Sorting by cumulative time
    stats.print_stats()
