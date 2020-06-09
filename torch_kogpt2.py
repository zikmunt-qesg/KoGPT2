import torch
from kogpt2.pytorch_kogpt2 import get_pytorch_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer, SentencepieceDetokenizer
from kogpt2.utils import get_tokenizer

tok_path = get_tokenizer()
model, vocab = get_pytorch_kogpt2_model()
tok = SentencepieceTokenizer(tok_path)
dtok = SentencepieceDetokenizer(tok_path)

sent = '그 여자는 아까부터 이쪽을 바라보았다. 나는 정신없이'

toked = tok(sent)
i=0
while i<40:
    input_ids = torch.tensor([vocab[vocab.bos_token],] + vocab[toked]).unsqueeze(0)
    pred = model(input_ids)[0]
    #gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
    #if gen == '</s>':
    #    break
    #sent += gen.replace('▁', ' ')
    #toked = tok(sent)
    #i += 1
    output_sequences = model.generate(input_ids=input_ids,
    max_length = 400,
    temperature = 0.7,
    repetition_penalty=1.5,
    do_sample=True)
    if len(output_sequences.shape) > 2:
        output_sequences.squeeze_()

    i=0
    for generated_sequence_idx, generated_sequence in enumerate(output_sequences):
        print("=== GENERATED SEQUENCE {} ===".format(generated_sequence_idx + 1))
        generated_sequence = generated_sequence.tolist()

        generated_tokens=[]
        for item in generated_sequence:
            new_token = vocab.to_tokens(item)
            generated_tokens.append(new_token)
        print( sent + dtok(generated_tokens))
        i+=1
        if i > 2:
            break
