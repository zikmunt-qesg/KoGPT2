import mxnet as mx
from kogpt2.mxnet_kogpt2 import get_mxnet_kogpt2_model
from gluonnlp.data import SentencepieceTokenizer
from kogpt2.utils import get_tokenizer
from random import randint

if mx.context.num_gpus() >0:
    ctx = mx.gpu()
else:
    ctx = mx.cpu()

tok_path = get_tokenizer()
model, vocab = get_mxnet_kogpt2_model(ctx=ctx)
tok = SentencepieceTokenizer(tok_path)

sent = '즐거운 아침'

toked = tok(sent)
i = 0
while i < 100:
    input_ids = mx.nd.array([vocab[vocab.bos_token]] + vocab[toked]).expand_dims(axis=0)
    pred = model(input_ids.as_in_context(ctx))
    if len(pred) > 1:
        NUM = len(pred) % randint(1, len(pred))
    pred = pred[NUM]
    print(pred)
    gen = vocab.to_tokens(mx.nd.argmax(pred, axis=-1).squeeze().astype('int').asnumpy().tolist())[-1]
    #if gen == '</s>':
    #    break
    sent += gen.replace('▁',' ')
    toked = tok(sent)
    i = i+1
print(sent)
