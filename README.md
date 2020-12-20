# Vocabulary restriction for BlenderBot

*WIP*

This is a modification to [Facebook's BlenderBot](https://ai.facebook.com/blog/state-of-the-art-open-source-chatbot/) on [ParlAI](https://parl.ai/). It restricts the chatbot to only form sentences using words in a given text file (the opposite of `beam-block-list-filename`). Code is based on TorchGeneratorAgent's [BeamSearch](https://github.com/facebookresearch/ParlAI/blob/master/parlai/core/torch_generator_agent.py#L1585). Tested on ParlAI v0.9.4, Blender 3B.

The chatbot maintains a pool of words (the **restricted** vocabulary) that are allowed in its responses. If an **extended** vocabulary is also provided, words that are in the extended vocabulary will be added to the pool of words if a user response contains it. Note: vocab lists are not case-sensitive.

To start interactive mode: `python run_interaction.py path/to/restricted/vocab.txt path/to/extended/vocab.txt zoo:blender/blender_3B/model`

Known issue: Words in the restricted list also automatically includes subwords that are words in the model's dictionary. E.g. including the word *someday* in the restricted list also automatically adds *some* to the restricted list.
