from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

def train_tokenizer(texts, vocab_size=30000):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.post_processor = processors.BertProcessing(("</s>", tokenizer.token_to_id("</s>")), ("<s>", tokenizer.token_to_id("<s>")))
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer

def encode(text, tokenizer):
    return tokenizer.encode(text).ids

def decode(ids, tokenizer):
    return tokenizer.decode(ids)
