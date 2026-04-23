To enable a flexible addition of datasets to this project, we convert all datasets from HuggingFace to the unified format.

The schema of the dataset:
 - context (json): conversation history in OpenAI format, where each message has a role and content.
 - answer_candidate_a (str): first answer candidate.
 - answer_candidate_b (str): second answer candidate.
 - ground_truth (int):
   - 0 if the first answer candidate is better;
   - 1 if the second answer candidate is better;
   - 2 if they are equal.
 - metadata (json): optional, you may have here additional information, e.g. context length.