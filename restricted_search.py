import torch

from parlai.core.dict import DictionaryAgent, TokenizationMode
from parlai.core.message import Message
from parlai.core.torch_generator_agent import TreeSearch, GreedySearch, BeamSearch, DelayedBeamSearch, TopKSampling, NucleusSampling
from parlai.utils.torch import neginf

class RestrictedSearch(TreeSearch):

    def __init__(self, wordpool, dict_agent, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wordpool = wordpool
        self.dict = dict_agent

    def select_paths(self, logprobs, prior_scores, current_length):
        """
        Select the next vocabulary item in these beams. Uses regular beam search.
        Adapted from https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1590
        """
        # if numel is 1, then this is the first time step, only one hyp is expanded
        if prior_scores.numel() == 1:
            logprobs = logprobs[0:1]

        # beam search actually looks over all hypotheses together so we flatten
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)

        # before flattening, assign neginf for words outside of wordpool
        bad_ids = torch.tensor(self.wordpool.bad_ids, dtype=torch.long, device=beam_scores.device)
        beam_scores.index_fill_(-1, bad_ids, neginf(logprobs.dtype))

        # look for subword combinations
        if beam_scores.size(0) == self.beam_size:
            for hypid in range(self.beam_size):
                last_word = self.get_last_word(self.partial_hyps[hypid])
                for non_boundary_id in self.wordpool.non_boundary_ids:
                    if last_word + self.dict[non_boundary_id] not in self.wordpool.good_str:
                        beam_scores[hypid][non_boundary_id] = neginf(beam_scores.dtype)

        # flatten and identify top hypotheses
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)

        voc_size = logprobs.size(-1)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs // voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size


        return (hyp_ids, tok_ids, best_scores)

    def get_last_word(self, token_ids):
        """
        Return the last word as a string from an iterable of token ids.
        Last word refers to the last string before the next whitepace (Ġ).
        Does not account for punctuation or special tokens (for now).
        """
        last_word = ""

        for position in range(-1, -len(token_ids) - 1, -1):
            last_word = self.dict[token_ids[position]] + last_word

            if last_word[0] == "Ġ":
                break

        return last_word[1:]


def _restricted_treesearch_factory(self, device):
    """
    Adapted from https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L939
    """
    method = self.opt.get('inference', 'greedy')
    beam_size = self.opt.get('beam_size', 1)
    if method == 'greedy':
        return GreedySearch(
            beam_size,
            min_length=0,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    elif method == 'beam':
        return BeamSearch(
            beam_size,
            min_length=self.beam_min_length,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    elif method == 'delayedbeam':
        return DelayedBeamSearch(
            self.opt['topk'],
            self.opt['beam_delay'],
            beam_size,
            min_length=self.beam_min_length,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    elif method == 'topk':
        return TopKSampling(
            self.opt['topk'],
            beam_size,
            min_length=self.beam_min_length,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    elif method == 'nucleus':
        return NucleusSampling(
            self.opt['topp'],
            beam_size,
            min_length=self.beam_min_length,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    elif method == 'restricted':
        return RestrictedSearch(
            self.wordpool,
            self.dict,
            beam_size,
            min_length=self.beam_min_length,
            block_ngram=self.beam_block_ngram,
            context_block_ngram=self.beam_context_block_ngram,
            length_penalty=self.opt.get('beam_length_penalty', 0.65),
            padding_token=self.NULL_IDX,
            bos_token=self.START_IDX,
            eos_token=self.END_IDX,
            device=device,
        )
    else:
        raise ValueError(f"Can't use inference method {method}")


def adaptive_observe(self, observation):
    """
    Process incoming message in preparation for producing a response.
    This includes remembering the past history of the conversation.
    Adapted from https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_agent.py#L1643
    """
    # TODO: Migration plan: TorchAgent currently supports being passed
    # observations as vanilla dicts for legacy interop; eventually we
    # want to remove this behavior and demand that teachers return Messages
    observation = Message(observation)

    # Sanity check everything is in order
    self._validate_observe_invariants()

    if observation.get('episode_done'):
        self.__expecting_clear_history = True
    elif 'labels' in observation or 'eval_labels' in observation:
        # make sure we note that we're expecting a reply in the future
        self.__expecting_to_reply = True

    # keep around the observation for updating history based on label
    self.observation = observation

    # update wordpool
    if self.dynamic_wordpool:
        tokens = DictionaryAgent.re_tokenize(observation["text"])
        for token in tokens:
            if token in self.wordpool.good_str:
                continue
            if token in self.wordpool.ok_str:
                self.wordpool.add_word(token)

    # possibly change tokenization methodology based on if this is a
    # training example
    is_training_mode = 'labels' in observation
    if hasattr(self.dict, 'set_tokenization_mode'):
        if is_training_mode:
            self.dict.set_tokenization_mode(TokenizationMode.TRAIN_TIME_TEXT)
        else:
            self.dict.set_tokenization_mode(TokenizationMode.TEST_TIME_TEXT)

    # Update the history using the observation.
    # We may also consider adding a temporary string to the history
    # using the `get_temp_history()` function: this string will
    # persist until it is updated.
    self.history.update_history(
        observation, temp_history=self.get_temp_history(observation)
    )

    if hasattr(self.dict, 'set_tokenization_mode'):
        if is_training_mode:
            self.dict.set_tokenization_mode(TokenizationMode.TRAIN_TIME_LABEL)
        else:
            self.dict.set_tokenization_mode(TokenizationMode.TEST_TIME_LABEL)

    return self.vectorize(
        observation,
        self.history,
        text_truncate=self.text_truncate,
        label_truncate=self.label_truncate,
    )
