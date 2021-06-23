
import bentoml
from bentoml.frameworks.transformers import TransformersModelArtifact
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModelForTokenClassification
import torch
from bentoml.adapters import JsonInput, JsonOutput

@bentoml.env(pip_packages=["transformers==4.7.0", "torch==1.9.0", "protobuf", "tokenizers==0.10.3"])
@bentoml.artifacts([TransformersModelArtifact("bertNerModel")])
class TransformerService(bentoml.BentoService):
    @bentoml.api(input=JsonInput(), output=JsonOutput(), batch=False)
    def predict(self, parsed_json):
        label_list = [
            "O",       # Outside of a named entity
            "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
            "I-MISC",  # Miscellaneous entity
            "B-PER",   # Beginning of a person's name right after another person's name
            "I-PER",   # Person's name
            "B-ORG",   # Beginning of an organisation right after another organisation
            "I-ORG",   # Organisation
            "B-LOC",   # Beginning of a location right after another location
            "I-LOC"    # Location
        ]
        sequence = parsed_json.get("text")
        model = self.artifacts.bertNerModel.get("model")
        tokenizer = self.artifacts.bertNerModel.get("tokenizer")
        # Bit of a hack to get the tokens with the special tokens
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
        inputs = tokenizer.encode(sequence, return_tensors="pt")

        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)
        results = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())]
        return results

if __name__ == "__main__":
    ts = TransformerService()

    model_name = "bert-ner"
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    tokenizer = AutoTokenizer.from_pretrained("/Users/manuellara/Work/FourthBrain/GLG_Specialist/bert-base-cased")
    # Option 1: Pack using dictionary (recommended)
    artifact = {"model": model, "tokenizer": tokenizer}
    ts.pack("bertNerModel", artifact)
    # Option 2: pack using the name of the model
    # ts.pack("bertNerModel","gpt2")
    # Note that while packing using the name of the model,
    # ensure that the model can be loaded using
    # transformers.AutoModelWithLMHead (eg GPT, Bert, Roberta etc.)
    # If this is not the case (eg AutoModelForQuestionAnswering, BartModel etc)
    # then pack the model by passing a dictionary
    # with the model and tokenizer declared explicitly
    saved_path = ts.save()
