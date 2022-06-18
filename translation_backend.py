import json
from flask import Flask, request, jsonify
import re
import torch
import torch.nn as nn


def tokenize_eng(sentence):
    sentence = re.sub(r'\n', '', sentence)
    # sentence = re.sub(r'[^\w\s\']', '', sentence.lower())
    sentence = re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', '', sentence.lower())
    return [words for words in sentence.split()]


def tokenize_tam(sentence):
    sentence = re.sub(r'\n', '', sentence)
    sentence = re.sub(r'\([^)]*\)', '', sentence)
    sentence = re.sub(r'[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\\\]\^\_\`\{\|\}\~]', '', sentence)
    return [words for words in sentence.split()]


english = torch.load('D:/language_models/english_vocab.pth')
tamil = torch.load('D:/language_models/tamil_vocab.pth')


class Transformer_model(nn.Module):
    def __init__(self, embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers,
                 num_decoder_layers, feed_forward, dropout_p, max_len, device):
        super(Transformer_model, self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.src_position_embedding = nn.Embedding(max_len, embedding_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.trg_position_embedding = nn.Embedding(max_len, embedding_size)

        self.device = device
        self.transformer = nn.Transformer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers,
                                          feed_forward, dropout_p)
        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout_p)
        self.src_pad_idx = src_pad_idx

    def make_src_mask(self, src):
        # src_shape=(src_len,N)
        src_mask = src.transpose(0, 1) == self.src_pad_idx
        # src_shape=(N,src_len)
        return src_mask

    def forward(self, src, trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.shape

        src_positions = (torch.arange(0, src_seq_length).unsqueeze(1).expand(src_seq_length, N).to(self.device))
        trg_positions = (torch.arange(0, trg_seq_length).unsqueeze(1).expand(trg_seq_length, N).to(self.device))

        embed_src = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions))
        embed_trg = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions))

        src_padding_mask = self.make_src_mask(src).to(self.device)
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)

        out = self.transformer(embed_src, embed_trg, src_key_padding_mask=src_padding_mask, tgt_mask=trg_mask)
        out = self.fc_out(out)
        return out


app = Flask(__name__)

"""##Hyperparameters for Model"""

device = torch.device('cuda')

num_epoch = 100
learning_rate = 3e-4
batch_size = 128

src_vocab_size = len(tamil.vocab)
trg_vocab_size = len(english.vocab)
# print(src_vocab_size, trg_vocab_size)
embedding_size = 512
num_heads = 4
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.10
max_len = 100
feed_forward = 2048
src_pad_idx = tamil.vocab.stoi['<pad>']

model = Transformer_model(embedding_size, src_vocab_size, trg_vocab_size, src_pad_idx, num_heads, num_encoder_layers,
                          num_decoder_layers, feed_forward, dropout, max_len, device).to(device)

pad_idx = english.vocab.stoi['<pad>']
model_save_name = 'seq2seq_transformer_220.pt'
path = f"D:/language_models/{model_save_name}"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()


def test_model(tam_sen_in):
    tam_sen_in='<sos> '+tam_sen_in+' <eos>'
    tam_encoded = []
    for x in tokenize_tam(tam_sen_in):
        tam_encoded.append(tamil.vocab.stoi[x])
    tam_sen = torch.Tensor(tam_encoded).long().to(device)
    tam_sen = tam_sen.reshape(-1, 1)

    outputs = [english.vocab.stoi["<sos>"]]
    for i in range(100):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(tam_sen, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == english.vocab.stoi["<eos>"]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]
    trans_eng = ''

    for a in translated_sentence:
        if a=='<unk>':
            continue
        trans_eng = trans_eng + ' ' + a
    return trans_eng[6:-5]


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        global tam_sen
        received_data = request.data
        received_data = json.loads(received_data.decode('utf-8'))
        tam_sen = received_data['sen']
        print(tam_sen)
        return tam_sen
    if request.method == 'GET':
        eng_sen = test_model(tam_sen)
        response = jsonify({'greetings': eng_sen})
        # response.headers.add("Access-Control-Allow-Origin", "*")
        return response


if __name__ == '__main__':
    app.run(debug=True)
