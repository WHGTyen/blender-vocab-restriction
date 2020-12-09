import types
import argparse

from parlai.core.agents import create_agent_from_model_file

import interactive_utils as interact
import restricted_search
from wordpool import WordPool


parser = argparse.ArgumentParser()
parser.add_argument("restricted_vocab_path", help="Path to text file containing allowed vocabulary")
parser.add_argument("extended_vocab_path", help="Path to text file containing extended vocabulary", default=None)
parser.add_argument("model", help="ParlAI model, e.g. 'zoo:blender/blender_3B/model", default="zoo:blender/blender_3B/model")

args = parser.parse_args()
RESTRICTED_VOCAB_PATH = args.restricted_vocab_path
EXTENDED_VOCAB_PATH = args.extended_vocab_path
MODEL = args.model

restricted_vocab = []
with open(RESTRICTED_VOCAB_PATH, 'r', encoding="utf-8") as vocab_file:
    for row in vocab_file:
        token = row.strip()
        restricted_vocab.append(token)

extended_vocab = []
if EXTENDED_VOCAB_PATH:
    with open(EXTENDED_VOCAB_PATH, 'r', encoding="utf-8") as vocab_file:
        for row in vocab_file:
            token = row.strip()
            extended_vocab.append(token)

agent = create_agent_from_model_file(MODEL,
                                     opt_overrides={
                                         "inference": "restricted"
                                     })
agent.set_interactive_mode(True)

# Replace original _treesearch_factory and observe methods to include restricted beam search
agent._treesearch_factory = types.MethodType(restricted_search._restricted_treesearch_factory, agent)
agent.observe = types.MethodType(restricted_search.adaptive_observe, agent)
agent.wordpool = WordPool(restricted_vocab, extended_vocab, agent.dict)
agent.dynamic_wordpool = bool(EXTENDED_VOCAB_PATH)

interact.start_interaction(agent, interact.DEFAULT_INTERACTION_OPTS, seed=0)
