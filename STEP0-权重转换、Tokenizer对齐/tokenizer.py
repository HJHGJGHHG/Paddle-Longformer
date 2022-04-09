from paddle.utils import try_import
from paddlenlp.transformers.tokenizer_utils import AddedToken
from paddlenlp.transformers.roberta.tokenizer import RobertaBPETokenizer

from transformers.models.longformer.tokenization_longformer import LongformerTokenizer as PTLongformerTokenizer
__all__ = [
    "LongformerTokenizer",
]


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    _chr = chr
    bs = list(range(ord("!"), ord("~") + 1)) + list(
        range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [_chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class LongformerTokenizer(RobertaBPETokenizer):
    r"""
    Construct a Longformer tokenizer, derived from the GPT tokenizer, using
    byte-level Byte-Pair-Encoding.
    This tokenizer inherits from :class:`~paddlenlp.transformers.GPTTokenizer`,
    which contains most of the main methods.
    Please should refer to the superclass for more information regarding methods.
    Args:
        vocab_file (str): file path of the vocabulary
        merges_file (str): file path of the merges file.
        errors (str): The method to handle errors in decoding
        max_len (int): The specified maximum sequence length. Default: "None".
        special_tokens (dict): The additional special tokens. Default: "None".
        bos_token (str): The special token for beginning of sequence token. Default: "<s>".
        eos_token (str): The special token for end of sequence token. Default: "</s>".
        cls_token (str): The special token for cls. Default: "<s>".
        sep_token (str): The special token for separator token . Default: "</s>".
        pad_token (str): The special token for padding. Default: "<pad>".
        eol_token (str): The special token for newline. Default: "\u010a".
        add_prefix (bool): Whether or not to add an initial space to the input.
            This allows to treat the leading word just as any other word.
            (Blenderbot adds an initial space when tokenizes input text, which
             is differnt from BlenderbotSmall)
    Examples:
        .. code-block:: python
            from paddlenlp.transformers import BlenderbotTokenizer
            tokenizer = BlenderbotTokenizer.from_pretrained("blenderbot-400M-distill")
            text = "My friends are cool but they eat too many carbs."
            inputs = tokenizer(text)
            # above line outputs:
            # {'input_ids': [863, 1329, 366, 1449, 373, 382, 1861, 618, 847, 911, 1372, 21, 2],
            # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
    """
    resource_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt"
    }
    pretrained_resource_files_map = {
        "vocab_file": {
            "longformer-base-4096":
                "D:/python/pyprojects/论文复现与实验/models/longformer-base-4096/vocab.json",
            "longformer-large-4096":
                "D:/python/pyprojects/论文复现与实验/models/longformer-base-4096/vocab.json",
        },
        "merges_file": {
            "longformer-base-4096":
                "D:/python/pyprojects/论文复现与实验/models/longformer-base-4096/merges.txt",
            "longformer-large-4096":
                "D:/python/pyprojects/论文复现与实验/models/longformer-base-4096/merges.txt"
        }
    }
    pretrained_init_configuration = {
        "longformer-base-4096": {
            "add_prefix": True
        },
        "longformer-large-4096": {
            "add_prefix": True
        }
    }


if __name__ == '__main__':
    tokenizer_pytorch = PTLongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    tokenizer_paddle = LongformerTokenizer.from_pretrained("longformer-base-4096")
    text = "It is a nice day today , I want to go to the park !"
    print(tokenizer_pytorch.encode_plus(text))
    print(tokenizer_paddle.encode(text))
    
    print(tokenizer_pytorch(text))
    print(tokenizer_paddle(text))
