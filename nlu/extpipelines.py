
from rasa_nlu.registry import registered_components
import thulac_tokenizer
import thulac_nim_tokenizer
import char_tokenizer
from bert_ner.ner import BertExtractor


def reg(comp):
    registered_components[comp.name] = comp


reg(thulac_tokenizer.ThulacTokenizer)
reg(thulac_nim_tokenizer.NimThulacTokenizer)
reg(char_tokenizer.CharTokenizer)
reg(BertExtractor)

