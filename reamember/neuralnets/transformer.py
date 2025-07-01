from transformers import AutoModel, AutoTokenizer

class Transformer:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts, device=None):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        if device is not None:
            inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.last_hidden_state
    
    def decode(self, embeddings):
        inputs = self.tokenizer.batch_decode(embeddings, skip_special_tokens=True)
        return inputs

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path):
        model = cls()
        model.model = AutoModel.from_pretrained(path)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        return model