import jieba

from typing import List

from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.decoders import Decoder


class JiebaNormalizer:
    def normalize(self, normalized: NormalizedString):
        normalized.nfkc()
        normalized.filter(lambda char: not char.isnumeric())
        normalized.replace(Regex("\s+"), " ")
        normalized.lowercase()


class JiebaPreTokenizer:
    def jieba_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        splits = []
        for token, start, stop in jieba.tokenize(str(normalized_string)):
            splits.append(normalized_string[start:stop])
        return splits
    
    
    def pre_tokenize(self, pretok: PreTokenizedString):
         pretok.split(self.jieba_split)


class JiebaDecoder:
    def decode(self, tokens: List[str]) -> str:
        return "".join(tokens)
    
    
