from data import *
from models.model.transformer import Transformer
import torch
from torch import nn
from util.bleu import idx_to_word, get_bleu

model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

model.load_state_dict(torch.load("saved/model-5.199447572231293.pt"))
print('Model Loaded.')

criterion = nn.CrossEntropyLoss(ignore_index=src_pad_idx)

model.eval()
model.to(device)

epoch_loss = 0
batch_bleu = []
with torch.no_grad():
    f = open('result/preds.txt', 'w')
    for i, batch in enumerate(test_iter):
        src = batch.src
        trg = batch.trg
        # print("src:", src.shape, src)
        # print("trg:", trg.shape, trg)
        # print("trg_sos_idx:", trg_sos_idx)
        # print("predict trg:", torch.Tensor([trg_sos_idx]).to(device).repeat(batch_size).unsqueeze(dim=1))

        # Autoregressive inference
        infer_trg = torch.IntTensor([trg_sos_idx]).to(device).repeat(src.shape[0]) #.unsqueeze(dim=1)
        # print("infer_trg:", infer_trg.shape, infer_trg)
        input_trg = torch.ones(src.shape[0], max_len).type(torch.IntTensor).to(device)  # one=<pad>
        input_trg[:, 0] = infer_trg
        # print("input_trg:", input_trg.shape, input_trg)

        for j in range(max_len):
            output = model(src, input_trg)
            # output_reshape = output.contiguous().view(-1, output.shape[-1])
            # print("output_reshape:", output_reshape.shape, output_reshape)
            # print("output_reshape[0]:", output_reshape[0].shape, output_reshape[0][:6])

            infer_trg = output.max(dim=-1)[1]
            # print("infer_trg:", infer_trg.shape, infer_trg)
            input_trg[:, j+1] = infer_trg[:, j]
            # print("input_trg:", input_trg[:, j+1].shape, input_trg[:, j+1])

            if input_trg[:, j+1].max() <= 3:  # stop inference when predict <eos> for the whole batch, to save time
                break

        output_infer = input_trg[:, 1:] #.contiguous().view(-1)
        trg = trg[:, 1:] #.contiguous().view(-1)
        # print("output_infer:", output_infer.shape, output_infer)
        # print("trg:", trg.shape, trg)

        # can we calculate loss and bleu when target and pred have different word numbers??
        # loss = criterion(output_infer, trg)
        # epoch_loss += loss.item()
        # print("loss:", epoch_loss / len(test_iter))

        # Write to file
        for source, target, pred in zip(src, trg, output_infer):  # sentence-wise
            # print("target:", i, target.shape, target)
            # print("pred:", i, pred.shape, pred)
            # print(idx_to_word(source, loader.source.vocab))
            source = idx_to_word(source, loader.source.vocab) #.split("<eos>")[0].strip()
            target = idx_to_word(target, loader.target.vocab) #.split("<eos>")[0].strip()
            got = idx_to_word(pred, loader.target.vocab) #.split("<eos>")[0].strip()

            f.write("- source: " + source + "\n")
            f.write("- expected: " + target + "\n")
            f.write("- got: " + got + "\n")
            f.write("\n")
    f.close()


        # total_bleu = []
        # for j in range(batch_size):
        #     try:
        #         trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
        #         output_words = output[j].max(dim=1)[1]
        #         output_words = idx_to_word(output_words, loader.target.vocab)
        #         bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
        #         total_bleu.append(bleu)
        #     except:
        #         pass
        #
        # total_bleu = sum(total_bleu) / len(total_bleu)
        # batch_bleu.append(total_bleu)
        #
        # batch_bleu = sum(batch_bleu) / len(batch_bleu)

