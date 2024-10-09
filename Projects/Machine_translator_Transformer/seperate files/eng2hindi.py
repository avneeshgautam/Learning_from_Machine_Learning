import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import Transformer
hindi_file= "/data3/home/avneeshg/Self_study/MLDL/Learning_from_Machine_Learning/Projects/Machine_translator_Transformer/dataset/iitb-english-hindi/target_train.txt"
english_file= "/data3/home/avneeshg/Self_study/MLDL/Learning_from_Machine_Learning/Projects/Machine_translator_Transformer/dataset/iitb-english-hindi/source_train.txt"
 
START_TOKEN = "<START>"
PADDING_TOKEN = "<PAD"
END_TOKEN = "<END>"

english_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', 
                      '*', '+', ',', '-', '.', '/', 
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     ':', '<', '=', '>', '?', '@', 
                     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                     'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e',
                     'f', 'g', 'h', 'i', 'j', 'k', 'l','m', 'n', 'o', 'p', 'q', 
                     'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', 
                     '~', PADDING_TOKEN, END_TOKEN]

hindi_vocabulary = [START_TOKEN, ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', 
                    '*', '+', ',', '-', '.', '/',
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    ':', '<', '=', '>', '?', '@',
                    'ँ', 'ं', 'ः', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ए', 'ऐ', 'ऑ', 'ओ', 'औ', 'क', 'ख', 'ग',
                    'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ',
                    'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', '़', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॅ',
                    'े', 'ै', 'ॉ', 'ो', 'ौ', '्', 'ॐ', '।', '॥', PADDING_TOKEN, END_TOKEN]

index_to_hindi = {k:v for k,v in enumerate(hindi_vocabulary)}
hindi_to_index = {v:k for k,v in enumerate(hindi_vocabulary)}
index_to_english = {k:v for k,v in enumerate(english_vocabulary)}
english_to_index = {v:k for k,v in enumerate(english_vocabulary)}

with open(english_file) as f:
    english_sentence = [line.strip() for line in f.readlines()]  # Removes "\n" from each line
    
with open(hindi_file) as f:
    hindi_sentence = [line.strip() for line in f.readlines()]  # Removes "\n" from each line

TOTAL_SENTENCES = len(english_sentence)
print("Total sentences: ", TOTAL_SENTENCES)
max_sequence_length = 300

def is_valid_tokens(sentence, vocabulary):
    for token in list(sentence):
        if token not in vocabulary:
            return False
    return True

def is_valid_length(sentence, max_sentence_length):
    return len(list(sentence)) < (max_sentence_length - 1) # need to reserve one space for <END> token



valid_sentence_indicies = []
for index in range(len(hindi_sentence)):
    hindi_sentence, english_sentence = hindi_sentence[index], english_sentence[index]
    if is_valid_length(hindi_sentence, max_sequence_length) \
      and is_valid_length(english_sentence, max_sequence_length) \
      and is_valid_tokens(hindi_sentence, hindi_vocabulary):
        valid_sentence_indicies.append(index)
    
hindi_sentence = [hindi_sentence[i] for i in valid_sentence_indicies]
english_sentence = [english_sentence[i] for i in valid_sentence_indicies]
print(len(hindi_sentence), len(english_sentence))


class TextDataset(Dataset):
    def __init__(self, english_sentence, hindi_sentence):
        self.english_sentence = english_sentence
        self.hindi_sentence = hindi_sentence
    
    def __len__(self):
        return len(self.english_sentence)
    
    def __getitem__(self, index):
        return self.english_sentence[index], self.hindi_sentence[index]

dataset = TextDataset(english_sentence, hindi_sentence)
len(dataset)

batch_size = 3
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
iterator = iter(train_loader)

for batch_num, batch in enumerate(iterator):
    print(batch)
    if batch_num > 3:
        break

def tokenize(sentence, language_to_index, start_token=True, end_token=True):
    sentence_word_indicies = [language_to_index[token] for token in list(sentence)]
    if start_token:
        sentence_word_indicies.insert(0, language_to_index[START_TOKEN])
    if end_token:
        sentence_word_indicies.append(language_to_index[END_TOKEN])
    
    for _ in range(len(sentence_word_indicies), max_sequence_length):
        sentence_word_indicies.append(language_to_index[PADDING_TOKEN])
    return torch.tensor(sentence_word_indicies)


eng_tokenized, hindi_tokenized = [], []
for sentence_num in range(batch_size):
    eng_sentence, hn_sentence = batch[0][sentence_num], batch[1][sentence_num]
    eng_tokenized.append(tokenize(eng_sentence, english_to_index, start_token=False, end_token=False) )
    hindi_tokenized.append( tokenize(hn_sentence, hindi_to_index, start_token=True, end_token=True) )


eng_tokenized = torch.stack(eng_tokenized)
hindi_tokenized = torch.stack(hindi_tokenized)
print(eng_tokenized.shape, hindi_tokenized.shape)   

NEG_INFTY = -1e9
def create_masks(eng_batch, kn_batch):
    num_sentences = len(eng_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    print(f"encoder_self_attention_mask {encoder_self_attention_mask.size()}: {encoder_self_attention_mask[0, :10, :10]}")
    print(f"decoder_self_attention_mask {decoder_self_attention_mask.size()}: {decoder_self_attention_mask[0, :10, :10]}")
    print(f"decoder_cross_attention_mask {decoder_cross_attention_mask.size()}: {decoder_cross_attention_mask[0, :10, :10]}")
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

d_model = 512
batch_size = 30
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 300
hindi_vocab_size = len(hindi_vocabulary)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          hindi_vocab_size,
                          english_to_index,
                          hindi_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PADDING_TOKEN)


criterian = nn.CrossEntropyLoss(ignore_index=hindi_to_index[PADDING_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 1

for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        eng_batch, hindi_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(eng_batch, hindi_batch)
        optim.zero_grad()
        
        hindi_predictions = transformer(eng_batch,
                                     hindi_batch,
                                     encoder_self_attention_mask.to(device), 
                                     decoder_self_attention_mask.to(device), 
                                     decoder_cross_attention_mask.to(device),
                                     enc_start_token=False,
                                     enc_end_token=False,
                                     dec_start_token=True,
                                     dec_end_token=True)
        
        labels = transformer.decoder.sentence_embedding.batch_tokenize(hindi_batch, start_token=False, end_token=True)
        loss = criterian(
            hindi_predictions.view(-1, hindi_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == hindi_to_index[PADDING_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"English: {eng_batch[0]}")
            print(f"Hindi Translation: {hindi_batch[0]}")
            kn_sentence_predicted = torch.argmax(hindi_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in kn_sentence_predicted:
              if idx == hindi_to_index[END_TOKEN]:
                break
              predicted_sentence += index_to_hindi[idx.item()]
            print(f"Hindi Prediction: {predicted_sentence}")


            transformer.eval()
            kn_sentence = ("",)
            eng_sentence = ("should we go to the mall?",)
            for word_counter in range(max_sequence_length):
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(eng_sentence, kn_sentence)
                predictions = transformer(eng_sentence,
                                          kn_sentence,
                                          encoder_self_attention_mask.to(device), 
                                          decoder_self_attention_mask.to(device), 
                                          decoder_cross_attention_mask.to(device),
                                          enc_start_token=False,
                                          enc_end_token=False,
                                          dec_start_token=True,
                                          dec_end_token=False)
                next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                next_token_index = torch.argmax(next_token_prob_distribution).item()
                next_token = index_to_hindi[next_token_index]
                kn_sentence = (kn_sentence[0] + next_token, )
                if next_token == END_TOKEN:
                  break
            
            print(f"Evaluation translation (should we go to the mall?) : {kn_sentence}")
            print("-------------------------------------------")