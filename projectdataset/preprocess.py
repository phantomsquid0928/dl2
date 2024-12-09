import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import numpy as np

class preprocess:
    id_to_char = {0: ' ',}
    char_to_id = {' ': 0,}


    def _update_vocab(self, txt):
        chars = list(txt)

        for i, char in enumerate(chars):
            if char not in self.char_to_id:
                tmp_id = len(self.char_to_id)
                self.char_to_id[char] = tmp_id
                self.id_to_char[tmp_id] = char

    def fetch_article_content(self, url):
        """
        Fetches the content of an article from a given URL.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text from the article
            paragraphs = soup.find_all('p')
            content = " ".join([p.get_text() for p in paragraphs])
            return content

        except Exception as e:
            print(f"Failed to fetch content from {url}: {e}")
            return ""

    def extract_relevant_sentences(self, article, conversation_context, top_n=3):
        """
        Extracts the most relevant sentences from the article based on similarity to the conversation context.
        """
        if not article.strip():
            return ""

        # Tokenize the article into sentences
        sentences = sent_tokenize(article)

        # Use TF-IDF to rank relevance of sentences
        vectorizer = TfidfVectorizer().fit_transform([conversation_context] + sentences)
        cosine_similarities = (vectorizer[0] * vectorizer.T).toarray()[0][1:]

        # Get top N sentences
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        relevant_sentences = " ".join([sentences[i] for i in top_indices])
        return relevant_sentences

    def compute_final_score(self, evaluations):
        """
        Computes the final score (0-9) for bot utterances based on majority voting.
        """
        final_score = 0
        for key in evaluations[0].keys():  # Assuming all dictionaries have the same keys
            votes = sum(1 for e in evaluations if e[key] == "yes")
            if votes >= 2:  # Majority voting
                final_score += 1
        return final_score

    def preprocess_dataset(self, dataset, mod=0, include_context=True):
        """
        Preprocesses the dataset for multitask training with unique IDs for each utterance,
        while separating TL (training) and VL (validation) sets.

        - Generation task:
            Input: human utterance (+ optional context) + relevant evidence.
            Target: bot utterance.
        - Classification task:
            Input: bot utterance + relevant evidence.
            Label: final_score (0–9).

        :param dataset: Dataset to process (contains TL and VL keys).
        :param mod: If 0, skips fetching data from URLs.
        :param include_context: Whether to include full conversation context as part of input.
        :return: 
            (generation_train, classification_train), 
            (generation_val, classification_val)
        """
        generation_train, classification_train = [], []
        generation_val, classification_val = [], []

        # Build the vocabulary by iterating over all utterances
        for k, v in dataset.items():
            for title, datas in v.items():
                for data in datas:
                    conversation = data['dataset']['conversations'][0]
                    utterances = conversation['utterances']

                    for utterance in utterances:
                        utterance_text = utterance['utterance_text']
                        self._update_vocab(utterance_text)  # Update vocab with the utterance

        # Process the dataset
        for k, v in dataset.items():
            is_training = k == "TL"  # Identify TL (training) or VL (validation)
            for title, datas in v.items():
                for data in datas:
                    conversation = data['dataset']['conversations'][0]
                    evidence_sources = conversation['metadata'].get('answer_evidence', [])

                    # Fetch evidence content if mod != 0
                    article_content = ""
                    if mod != 0:
                        for source in evidence_sources:
                            article_content += self.fetch_article_content(source.get('source', ''))

                    # Process conversation
                    utterances = conversation['utterances']
                    context = []  # Context for human-bot conversation flow

                    for i, utterance in enumerate(utterances):
                        speaker_id = utterance['speaker_id']
                        utterance_text = utterance['utterance_text']
                        exchange_id = utterance.get('exchange_id', '')
                        utterance_id = utterance.get('utterance_id', '')
                        unique_id = f"{exchange_id}_{utterance_id}"

                        # Convert utterance_text to int16 array, 64 is too huge
                        utterance_ids = np.array([self.char_to_id[char] for char in utterance_text], dtype=np.int16)

                        # Add to classification dataset
                        if speaker_id == 0:  # Bot utterance
                            evaluations = utterance.get('utterance_evaluation', [])
                            final_score = self.compute_final_score(evaluations) if len(evaluations) == 3 else 0

                            relevant_evidence = self.extract_relevant_sentences(article_content, " ".join(context)) if mod != 0 else ""
                            evidence_ids = np.array([self.char_to_id[char] for char in relevant_evidence], dtype=np.int16)

                            classification_item = {
                                "id": unique_id,
                                "input": np.concatenate((utterance_ids, evidence_ids)),
                                "label": final_score
                            }
                            if final_score > 0:
                                if is_training:
                                    classification_train.append(classification_item)
                                else:
                                    classification_val.append(classification_item)

                        # Add to generation dataset
                        if i > 0 and utterances[i - 1]["speaker_id"] != 0:  # Prior utterance is human
                            if include_context:
                                # Include full context
                                human_context = np.array([self.char_to_id[char] for char in " ".join(context)], dtype=np.int16)
                                input_data = np.concatenate((human_context, evidence_ids))
                            else:
                                # Exclude context, only current utterance
                                input_data = np.concatenate((utterance_ids, evidence_ids))
                            
                            generation_item = {
                                "id": unique_id,
                                "input": input_data,
                                "output": utterance_ids
                            }
                            if is_training:
                                generation_train.append(generation_item)
                            else:
                                generation_val.append(generation_item)

                        # Update context if including
                        if include_context:
                            context.append(utterance_text)

        return (generation_train, classification_train), (generation_val, classification_val)



    def remove_mismatch(self, gen_data, class_data) :
        gen_ids = set(item['id'] for item in gen_data)
        class_ids = set(item['id'] for item in class_data)

        # Find common IDs
        print(len(gen_ids))
        print(len(class_ids))
        noncommon_ids = gen_ids & class_ids

        print(f'dd{len(noncommon_ids)}')

        # Filter both datasets
        filtered_generation_data = [item for item in gen_data if item['id'] in noncommon_ids]
        filtered_classification_data = [item for item in class_data if item['id'] in noncommon_ids]

        print(f'ss{len(filtered_generation_data)}')

        return filtered_generation_data, filtered_classification_data

    def get_vocab(self):
        return self.char_to_id, self.id_to_char


# Example usage
# Assuming `dataset` is already loaded
if __name__ == '__main__' :
    from responseQAloader import ResponseQAloader
    loader = ResponseQAloader(base_path="projectdataset/responsedata")
    dataset = loader.load_data()

    mod = 0  # Skip fetching URLs
    a = preprocess()
    trains, valids = a.preprocess_dataset(dataset, mod)

    print(f"Generation Data Size: {len(trains[0]) + len(valids[0])}")
    print(f"Classification Data Size: {len(trains[1]) + len(valids[1])}")

    print(f'gen tra size : {len(trains[0])}')
    print(f'gen val size : {len(valids[0])}')
    print(f'class tra size : {len(trains[1])}')
    print(f'class val size : {len(valids[1])}')

    gen_train, class_train = a.remove_mismatch(trains[0], trains[1])
    gen_val, class_val = a.remove_mismatch(valids[0], valids[1])

    print(f'managed gen tra size : {len(gen_train)}')
    print(f'managed gen val size : {len(gen_val)}')
    print(f'managed class tra size : {len(class_train)}')
    print(f'managed class val size : {len(class_val)}')

    
