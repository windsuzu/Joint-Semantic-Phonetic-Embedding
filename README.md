# Enhance Chinese-to-Japanese Neural Machine Translation

## Methods

| **Topic**                                                       | **Content**                                                                                                                                                                                                                                                                                            |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Embedding](methods/1.%20embedding/README.md)                   | Embedding 介紹，包含：<br/><ul><li>Word-level</li><li>Phrase-level</li><li>Sentence-level</li></ul>                                                                                                                                                                                                    |
| [Encoder-Decoder](methods/2.%20encoder-decoder/README.md)       | 現今 NMT 主流架構，但被固定長度的 Embedding 所限制。                                                                                                                                                                                                                                                   |
| [Attention](methods/3.%20attention/README.md)                   | Attention, Self-attention, Multi-head attention 是 NMT 成功的主要原因，裡面還比較了 RNN 和 CNN-based NMT，必看影片：<br/><ul><li>[Conditional Generation by RNN & Attention](https://www.youtube.com/watch?v=f1KUUz7v8g4)</li><li>[Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)</li></ul> |
| [Decoding](methods/4.%20decoding/README.md)                     | Decoding 是最終翻譯過程的核心，最常見的方法有：<br/><ul><li>Beam Search</li><li>Generating Diverse Translations</li><li>[Top-k sampling](approaches/octanove.md)</li></ul>另外還有即時翻譯的 decoding 方法。                                                                                           |
| [Open-Vocabulary](methods/5.%20open-vocabulary/README.md)       | 如何解決單字量過多膨脹的問題？使用 UNK token，或是改變模型的輸入為：<br><ul><li>Word-based NMT</li><li>Character-based NMT</li><li>Subword-based NMT</li></ul>其中 subword 擷取的最常見方法為 BPE (Byte pair encoding)。                                                                               |
| [Monolingual dataset](methods/6.%20monolingual/README.md)       |
| [Multilingual NMT](methods/7.%20multilingual/README.md)         |
| [Model Errors](methods/8.%20model-errors/README.md)             |
| [Training Methods](methods/9.%20training-methods/README.md)     |
| [Interpretability](methods/10.%20interpretability/README.md)    |
| [Alternative Models](methods/11.%20alternatives/README.md)      |
| [Data Sparsity](methods/12.%20data-sparsity/README.md)          |
| [Model Size](methods/13.%20model-size/README.md)                |
| [Extended Context](methods/14.%20extended-context/README.md)    |
| [NMT-SMT Hybird System](methods/15.%20nmt-smt-hybrid/README.md) |

## Approaches



## Motivation


## Datasets


## Experiments



# DeepL

https://techcrunch.com/2017/08/29/deepl-schools-other-online-translators-with-clever-machine-learning/

DeepL 已經很好，但在中日翻譯還是有問題

# Motivation (我可以做什麼別人沒做過的，別人沒用的方法)

我想提升中文和日文的翻譯效能
JPCN
https://www.aclweb.org/anthology/2020.iwslt-1.12/
https://www.aclweb.org/anthology/2020.iwslt-1.20/

Unsupervised learning
https://paperswithcode.com/paper/phrase-based-neural-unsupervised-machine
https://paperswithcode.com/task/unsupervised-machine-translation

Low resource learning
https://paperswithcode.com/task/low-resource-neural-machine-translation
https://paperswithcode.com/paper/two-new-evaluation-datasets-for-low-resource

Reinforcement
https://www.aclweb.org/anthology/2020.acl-main.319/

Adversarial
https://www.aclweb.org/anthology/2020.acl-main.529/
https://www.aclweb.org/anthology/2020.acl-main.370/

或許能用一個文法 discriminator 來強化 generator

Multimodal
https://paperswithcode.com/task/multimodal-machine-translation


# Method (哪些東西可以幫助我，現有的方法，基於這些方法再加上我的方法)

所有 NMT 技巧
https://arxiv.org/pdf/1912.02047.pdf

Transformer
https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html

CNN
https://paperswithcode.com/sota/machine-translation-on-wmt-2017-english-1
https://paperswithcode.com/paper/pay-less-attention-with-lightweight-and
https://zhuanlan.zhihu.com/p/60482693

分析
https://paperswithcode.com/paper/how-much-attention-do-you-need-a-granular

Scoring
BLEU
https://paperswithcode.com/search?q_meta=&q=translation+evaluation


# Other j-c implementation

1. Chinese–Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information
Unsupervised neural machine translation

logographic language pairs
Sub-character Level Information (ideograph or stroke)

sub-character-level > character-level
stroke > ideograph

---

2. Improving Character-level Japanese-Chinese Neural Machine Translation with Radicals as an Additional Input Feature

additional linguistic features in character-level NMT
radicals of Chinese characters (or kanji)

---

3. LIT Team’s System Description for Japanese-Chinese Machine

data processing
Large-scale back-translation on monolingual corpus
exclusive word embedding
different granularity of tokens like sub-word level

---

4. Octanove Labs’ Japanese-Chinese Open Domain Translation System

parallel corpus filtering
back translation

# Measurement, Contribution

BLEU => transformer-only implementation
=> try to break it

Blind testing through questionnaire

## Progress

- [x] Learning Training skills
- [x] Implement a new tf transformer translation system
- [ ] **Collect datasets**
- [ ] Evaluate the performance of the system
- [ ] Update the system with some skills and features
- [ ] Think of a new approach
- [ ] Evaluation (BLEU, perplexity, other semantic metrics...)
- [ ] Identify the problem, find a solution, or leave a vision

1. 先搞懂所有 translation 訓練方法
2. 試做基本翻譯系統 (transformer, CNN)
3. 測試他們的分數 (SOTA baseline, Japanese-Chinese baseline)
4. 開始加入一些特徵
5. 使用自己的方法
6. 測試分數、若不夠好再加上盲測
7. 實際找出問題，想辦法解決、或留下未來展望