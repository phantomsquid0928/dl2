import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from konlpy.tag import Mecab  # Import Mecab tokenizer
import numpy as np


class preprocess:
    def __init__(self, cache_path="tokenized_cache.pkl"):
        self.mecab = Mecab()  # Initialize Mecab
        self.char_to_id = {'<>' : 0}
        self.id_to_char = {0 : '<>'} ##<> for failed to change
        self.cache_path = cache_path 
        self.tokenized_cache = self.load_cache()

    def load_cache(self):
        """Load the tokenized cache from file if it exists."""
        if os.path.exists(self.cache_path):
            print(f"Loading tokenized cache from {self.cache_path}...")
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        """Save the tokenized cache to file."""
        print(f"Saving tokenized cache to {self.cache_path}...")
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.tokenized_cache, f)

    def _update_vocab(self, text):
        """Update the vocabulary with characters from the given text."""
        for char in text:
            if char not in self.char_to_id:
                new_id = len(self.char_to_id)
                self.char_to_id[char] = new_id
                self.id_to_char[new_id] = char

    def tokenize_utterance(self, utterance):
        """Tokenize a single utterance using Mecab and update cache."""
        if utterance in self.tokenized_cache:
            return self.tokenized_cache[utterance]
        tokens = self.mecab.morphs(utterance)
        self.tokenized_cache[utterance] = tokens
        return tokens

    def tokenize_utterances(self, utterances):
        """Tokenize a list of utterances using Mecab in parallel."""
        print(f"Tokenizing {len(utterances)} utterances...")
        tokenized_results = []
        batch_size = 10000  # Process in smaller batches to optimize performance

        for i in range(0, len(utterances), batch_size):
            batch = utterances[i:i + batch_size]
            tokenized_results.extend([self.tokenize_utterance(utt) for utt in batch])

        return tokenized_results

    def preprocess_dataset(self, dataset, include_context=True, mod='words'):
        generation_train, classification1_train, classification2_train = [], [], []
        generation_val, classification1_val, classification2_val = [], [], []
        all_utterances = []
        t = 0
        # Collect all utterances for tokenization
        for k, v in dataset.items():
            for title, datas in v.items():
                for data in datas:
                    t += 1
                    conversation = data['dataset']['conversations'][0]
                    utterances = conversation['utterances']
                    for utterance in utterances:
                        all_utterances.append(utterance['utterance_text'])

        # Tokenize all utterances
        tokenized_utterances = self.tokenize_utterances(all_utterances)

        # Update vocabulary with all tokens
        for tokens in tokenized_utterances:
            if mod == 'words' :
                self._update_vocab(tokens)
            else : self._update_vocab(" ".join(tokens))

        print(f'conversations : {t}')
        print(f'utterances len : {len(all_utterances)}')

        
        # Map tokenized data back to dataset structure
        tokenized_index = 0  # Ensure the index tracks all utterances globally
        for k, v in dataset.items():
            is_training = k == "TL"  # Identify TL (training) or VL (validation)
            for title, datas in v.items():
                for data in datas:
                    conversation = data['dataset']['conversations'][0]
                    likeability = conversation.get('conversation_evaluation', {}).get('likeability', [])
                    utterances = conversation['utterances']

                    # Process context for classification2
                    tokenized_contexts = tokenized_utterances[tokenized_index:tokenized_index + len(utterances)]
                    full_context = " ".join([" ".join(tokens) for tokens in tokenized_contexts])
                    
                    if mod == 'words' : 
                        flattened_tokens = [token for tokens in tokenized_contexts for token in tokens]
                        # Convert tokens to IDs
                        full_context_ids = np.array([self.char_to_id[char] for char in flattened_tokens], dtype=np.int32)
                    else : 
                        full_context_ids = np.array([self.char_to_id[char] for char in full_context], dtype=np.int32)

                    likeability_label = [1] if sum(1 for lbl in likeability if lbl == "yes") >= 2 else [0]
                    classification2_item = {
                        "id": conversation.get("conversation_id", ""),
                        "input": full_context_ids,
                        "output": likeability_label,
                    }

                    if is_training:
                        classification2_train.append(classification2_item)
                    else:
                        classification2_val.append(classification2_item)

                    # Process human-bot pairs for generation and classification1
                    context = []
                    for i in range(0, len(utterances), 2):
                        # Ensure indices are valid
                        if i + 1 >= len(utterances) or tokenized_index + i + 1 >= len(tokenized_utterances):
                            break

                        human_utterance = utterances[i]
                        bot_utterance = utterances[i + 1]
                    

                        human_tokens = tokenized_utterances[tokenized_index + i]
                        bot_tokens = tokenized_utterances[tokenized_index + i + 1]
                        human_text = " ".join(human_tokens)
                        bot_text = " ".join(bot_tokens)
                        if mod == 'words' : 
                            human_ids = np.array([self.char_to_id.get(char, 0) for char in human_tokens], dtype = np.int32)
                            bot_ids = np.array([self.char_to_id.get(char, 0) for char in bot_tokens], dtype=np.int32)
                        else :
                            human_ids = np.array([self.char_to_id.get(char, 0) for char in human_text], dtype=np.int32)
                            bot_ids = np.array([self.char_to_id.get(char, 0) for char in bot_text], dtype=np.int32)

                        # if include_context:
                        #     if mod == 'words' : 
                        #         human_context = np.array(
                        #             [self.char_to_id.get(char, 0) for chars in context for char in chars],
                        #             dtype=np.int16,
                        #         )
                        #     else :
                        #         human_context = np.array(
                        #             [self.char_to_id.get(char, 0) for char in " ".join(context)],
                        #             dtype=np.int16,
                        #         )
                        #     input_data = np.concatenate((human_context, human_ids))
                        # else:
                        #    input_data = human_ids
                        input_data = human_ids

                        generation_item = {
                            "id": bot_utterance.get("exchange_id", "") + "_" + bot_utterance.get("utterance_id", ""),
                            "input": input_data,
                            "output": bot_ids,
                        }
                        if is_training:
                            generation_train.append(generation_item)
                        else:
                            generation_val.append(generation_item)

                        # Classification1 task
                        evaluations = bot_utterance.get("utterance_evaluation", [])
                        if len(evaluations) == 3:
                            output = np.zeros(9, dtype=np.int32)
                            for key in evaluations[0].keys():
                                votes = sum(1 for e in evaluations if e[key] == "yes")
                                if votes >= 2:
                                    output[list(evaluations[0].keys()).index(key)] = 1

                            classification1_item = {
                                "id": bot_utterance.get("exchange_id", "") + "_" + bot_utterance.get("utterance_id", ""),
                                "input": bot_ids,
                                "output": output,
                            }
                            if is_training:
                                classification1_train.append(classification1_item)
                            else:
                                classification1_val.append(classification1_item)

                        # Update context and indices
                        if mod == 'words' : 
                            context.extend([human_tokens, bot_tokens])
                        else : context.extend([human_text, bot_text])

                    # Increment tokenized_index by the total number of utterances processed
                    tokenized_index += len(utterances)

        # Save tokenized cache
        self.save_cache()

        return (
            (generation_train, classification1_train, classification2_train),
            (generation_val, classification1_val, classification2_val),
        )





# Example usage
if __name__ == '__main__':
    from responseQAloader import ResponseQAloader
    loader = ResponseQAloader(base_path="projectdataset/responsedata")
    dataset = loader.load_data()

    preprocess_instance = preprocess()
    trains, valids = preprocess_instance.preprocess_dataset(dataset, include_context=True)

    print(f"Generation Data Size: {len(trains[0]) + len(valids[0])}")
    print(f"Classification Data Size: {len(trains[1]) + len(valids[1])}")
    print(f"Classification2 Data Size: {len(trains[2]) + len(valids[2])}")

    char_to_id, id_to_char = preprocess_instance.char_to_id, preprocess_instance.id_to_char
    print(f"Vocabulary Size: {len(char_to_id)}")
    for i in range(0, 5) : print(f'id_to_char[{i}] : {id_to_char[i]}')
    
    x = [x['input'] for x in trains[1]]
    t = [x['output'] for x in trains[1]]

    x1 = [x['input'] for x in trains[2]]
    t1 = [x['output'] for x in trains[2]]

    print(f'exists : {np.sum(np.array(t).flatten() == 0)}, {np.sum(np.array(t).flatten() == 1)}')
    print(f'exists: {np.sum(np.array(t1).flatten() == 0)}, {np.sum(np.array(t1).flatten() == 1)}')
    