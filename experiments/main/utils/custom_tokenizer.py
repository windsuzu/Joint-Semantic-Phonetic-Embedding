import jieba

from typing import List

from tokenizers import Tokenizer, Regex, NormalizedString, PreTokenizedString
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.decoders import Decoder


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
    
    
def load_jieba_tokenizer(tokenizer_path):
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.pre_tokenizer = PreTokenizer.custom(JiebaPreTokenizer())
    tokenizer.decoder = Decoder.custom(JiebaDecoder())
    
    return tokenizer




if __name__ == "__main__":
    
    jieba_tokenizer = load_jieba_tokenizer("../../tokenizer/tokenizer_jieba.json")
    
    # Print first ten vocab
    print([(key, val) for key, val in jieba_tokenizer.get_vocab().items()][:10])
    print(jieba_tokenizer.get_vocab_size())
    print()

    # Encode and Decode Testing
    encoded = jieba_tokenizer.encode("主茎及1次分蘖精米蛋白质含量的标准偏差小,为0.28~0.35%,2次分蘖的标准偏差大,为0.44~0.60%。😀")

    print(encoded.ids)
    print(encoded.tokens)
    print(jieba_tokenizer.decode(encoded.ids))