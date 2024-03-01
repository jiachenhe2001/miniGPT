from transformers import GPT2Tokenizer, GPTNeoForCausalLM
import torch
from torch.nn.functional import cosine_similarity
'''
# Initialize tokenizer and GPT-Neo model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')

# Tokenize and encode the phrase "Valentine's Day"
input_ids = tokenizer("Valentine's Day", return_tensors='pt').input_ids

# Get embeddings for the input phrase
with torch.no_grad():
    outputs = model(input_ids)
    hidden_states = outputs.last_hidden_state  # Shape: (batch_size, num_tokens, hidden_size)

# Aggregate embeddings for the phrase (mean pooling here)
phrase_embedding = hidden_states.mean(dim=1)

# You would need to define or have the following:
# 1. `get_embeddings` function to obtain embeddings for your candidate words/phrases
# 2. `candidate_words` list containing the words/phrases to compare against

# Assuming the above are defined, you would proceed as follows:

# candidate_embeddings = get_embeddings(candidate_words)  # Shape: (num_candidates, hidden_size)

# Calculate cosine similarity between the phrase and each candidate word/phrase
# similarities = torch.nn.functional.cosine_similarity(phrase_embedding.unsqueeze(0), candidate_embeddings)

# Rank candidates based on similarity scores and select top 5
# top_indices = torch.topk(similarities, 5).indices
# top_words = [candidate_words[i] for i in top_indices]
'''

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')

# Function to get embeddings
def get_phrase_embedding(phrase, model, tokenizer):
    input_ids = tokenizer(phrase, return_tensors='pt').input_ids
    with torch.no_grad():
        outputs = model(input_ids,output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        phrase_embedding = hidden_states.mean(dim=1)
    return phrase_embedding

# Get embeddings for each phrase
word_list1 = ["dinner","movie","gift", "chocolate", "love", "music", "flower"]
word_list2 = ["sweet","read","book", "cat", "cake", "computer", "family"]
for word1 in word_list1:
    for word2 in word_list2:
        embedding_phrase_1 = get_phrase_embedding("Enjoy valentine's day with your partner", model, tokenizer)
        embedding_phrase_2 = get_phrase_embedding(word1, model, tokenizer)
        embedding_phrase_3 = get_phrase_embedding(word2, model, tokenizer)

        # Compute cosine similarities
        similarity_1 = cosine_similarity(embedding_phrase_1, embedding_phrase_2)
        similarity_2 = cosine_similarity(embedding_phrase_1, embedding_phrase_3)

        print(word1,{similarity_1.item()})
        print(word2,{similarity_2.item()})
        print(similarity_1.item() - similarity_2.item())
        print('\n')
        