"""
Start interactive mode within Python. Code adapted from:
https://github.com/facebookresearch/ParlAI/blob/master/parlai/scripts/interactive.py
"""
import random

from parlai.core.worlds import create_task
from parlai.core.opt import Opt
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent

DEFAULT_INTERACTION_OPTS = {
    "display_examples": False,
    "display_add_fields": "",
    "interactive_task": True,
    "outfile": "",
    "save_format": "conversations",
    "log_keep_fields": "all",
    "interactive_mode": True,
    "task": "interactive",
}

def start_interaction(agent, interaction_opts, seed=None):
    # Starts interactive mode. Code adapted from parlai/scripts/interactive.py
    if seed != None:
        random.seed(seed)

    opt = Opt(interaction_opts)
    human_agent = LocalHumanAgent(opt)
    world_logger = WorldLogger(opt)
    world = create_task(opt, [human_agent, agent])

    # Show some example dialogs:
    while not world.epoch_done():
        world.parley()
        if world.epoch_done() or world.get_total_parleys() <= 0:
            # chat was reset with [DONE], [EXIT] or EOF
            if world_logger is not None:
                world_logger.reset()
            continue

        if world_logger is not None:
            world_logger.log(world)
        if opt.get('display_examples'):
            print("---")
            print(world.display())

    if world_logger is not None:
        # dump world acts to file
        world_logger.write(opt['outfile'], world, file_format=opt['save_format'])


