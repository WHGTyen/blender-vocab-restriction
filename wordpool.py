class WordPool:
    """
    Keeps track of different lists of words, including:
    - good_ids: ids of vocabulary from the restricted pool
    - bad_ids: ids of vocabulary not in the restricted pool
    - boundary_ids: ids of tokens that contain a word boundary (e.g. punctuation, or words that starts with whitespace)
    - non_boundary_ids: ids of tokens that do not contain word boundaries (e.g. word endings, word pieces)
    """
    def __init__(self, restricted_pool, extended_pool, dict_agent):
        self.all_ids = set(dict_agent.ind2tok.keys())
        self.good_ids = list()
        self.bad_ids = list(dict_agent.ind2tok.keys())

        self.boundary_ids = list()
        self.non_boundary_ids = list()

        self.good_str = set()   # Keeps track of words in the restricted pool that are not in the model's dictionary
        self.ok_str = set()     # Keeps track of words in the extended pool

        self.dict_agent = dict_agent
        if dict_agent.null_token: self._add_id(dict_agent[dict_agent.null_token])
        if dict_agent.start_token: self._add_id(dict_agent[dict_agent.start_token])
        if dict_agent.end_token: self._add_id(dict_agent[dict_agent.end_token])
        if dict_agent.unk_token: self._add_id(dict_agent[dict_agent.unk_token])

        for idx, word in dict_agent.ind2tok.items():
            if word[0] != "Ġ":      # Allow wordpieces that are not full words e.g. morphemes, punctuation
                self._add_id(idx)

            if word[0] == "Ġ" or not word.isalpha():
                self.boundary_ids.append(idx)
            else:
                self.non_boundary_ids.append(idx)

        for word in restricted_pool:
            self.add_word(word)

        for word in extended_pool:
            self.ok_str.add(word)
            self.ok_str.add(word.lower())
            self.ok_str.add(word.capitalize())

        assert len(self.boundary_ids) + len(self.non_boundary_ids) == len(self.all_ids)
        assert len(self.good_ids) + len(self.bad_ids) == len(self.all_ids)


    def add_word(self, word):
        if word in self.good_str:
            return

        self.good_str.add(word)
        self.good_str.add(word.lower())
        self.good_str.add(word.capitalize())

        self._add_id(self.dict_agent.txt2vec(word))
        self._add_id(self.dict_agent.txt2vec(word.lower()))
        self._add_id(self.dict_agent.txt2vec(word.capitalize()))

    def _add_id(self, ids):
        if type(ids) == int:
            if ids not in self.good_ids:
                self.good_ids.append(ids)
            if ids in self.bad_ids:
                self.bad_ids.remove(ids)
        else:
            for idx in ids:
                if idx not in self.good_ids:
                    self.good_ids.append(idx)
                if idx in self.bad_ids:
                    self.bad_ids.remove(idx)


