# Enhance Chinese-to-Japanese Neural Machine Translation

## Methods

| **Topic**                                                       | **Content**                                                                                                                                                                                                                                                                                              |
| --------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Embedding](methods/1.%20embedding/README.md)                   | Embedding 介紹，包含：<br/><ul><li>Word-level</li><li>Phrase-level</li><li>Sentence-level</li></ul>                                                                                                                                                                                                      |
| [Encoder-Decoder](methods/2.%20encoder-decoder/README.md)       | 現今 NMT 主流架構，但被固定長度的 Embedding 所限制。                                                                                                                                                                                                                                                     |
| [Attention](methods/3.%20attention/README.md)                   | Attention, Self-attention, Multi-head attention 是 NMT 成功的主要原因，裡面還比較了 RNN 和 CNN-based NMT，必看影片：<br/><ul><li>[Conditional Generation by RNN & Attention](https://www.youtube.com/watch?v=f1KUUz7v8g4)</li><li>[Transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA)</li></ul>   |
| [Decoding](methods/4.%20decoding/README.md)                     | Decoding 是最終翻譯過程的核心，最常見的方法有：<br/><ul><li>Beam Search</li><li>Generating Diverse Translations</li><li>[Top-k sampling](approaches/octanove.md)</li></ul>另外還有即時翻譯的 decoding 方法。                                                                                             |
| [Open-Vocabulary](methods/5.%20open-vocabulary/README.md)       | 如何解決單字量過多膨脹的問題？使用 UNK token，或是改變模型的輸入為：<br><ul><li>Word-based NMT</li><li>Character-based NMT</li><li>Subword-based NMT</li></ul>其中 subword 擷取的最常見方法為 BPE (Byte pair encoding)。                                                                                 |
| [Monolingual dataset](methods/6.%20monolingual/README.md)       | 應用非常巨量的單邊語言詞庫 (monolingual corpus)，來擴充平行詞庫 (parallel corpus)，加強翻譯的資料集數量。目前最有名的作法是 **Back-translation**。其他還有利用 `Unsupervised NMT` 來取消對平行詞庫的需求。([Data-Sparsity](methods/12.%20data-sparsity/README.md) 會提到)                                |
| [Multilingual NMT](methods/7.%20multilingual/README.md)         | 從 1-1 翻譯進階為**多種語言之間的交叉翻譯**，常見的名詞為 **Zero-shot translation, Pivot-based zero-shot translation**。                                                                                                                                                                                 |
| [Model Errors](methods/8.%20model-errors/README.md)             | 比較 **model errors** 和 **search errors** 的關係，以及進階解決 **length deficiency** 的問題。                                                                                                                                                                                                           |
| [Training Methods](methods/9.%20training-methods/README.md)     | 介紹 training 所會用到的 `loss`，以及多種問題如 `overfitting`、`vanishing gradient`、`degradation`、`exposure bias`，再分別介紹解決方法：<br/><ul><li>Regularization</li><li>Residual connections</li><li>Reinforcement Learning</li><li>Adversarial Learning</li><li>Dual Supervised Learning</li></ul> |
| [Interpretability](methods/10.%20interpretability/README.md)    | 解釋 NMT 模型和翻譯結果的方法：<br><ul><li>Post-hoc Interpretability</li><li>Model-intrinsic Interpretability</li><li>Quality estimation</li><li>Word Alignment (Alignment-based NMT)</li></ul>                                                                                                          |
| [Alternative Models](methods/11.%20alternatives/README.md)      | 提出更多基於 `Transformer`、`attention`、`Encoder-Decoder` 的架構之上，加入 memory 或修改原架構的方法。例如：<br/><ul><li>Relative Position Representation</li><li>Memory-augmented Neural Networks</li><li>Non-autoregressive</li></ul>                                                                 |
| [Data Sparsity](methods/12.%20data-sparsity/README.md)          | 處理資料集雜訊 (noise) 過多時的處理方法：<br/><ul><li>Corpus Filtering</li><li>Domain Adaptation</li></ul>處理平行資料集 (parallel corpus) 太少時的處理方法：<br/><ul><li>Low-resource NMT</li><li>[Unsupervised NMT](https://zhuanlan.zhihu.com/p/30649985)</li></ul>                                   |
| [Model Size](methods/13.%20model-size/README.md)                | Model size 是在移動平台上的一大限制，有的方法刪除不必要的 weights、或重複的 neurons，其中兩個最核心的方法為：<br/><ul><li>Neural Architecture Search (NAS)</li><li>Knowledge Distillation</li></ul>                                                                                                      |
| [Extended Context](methods/14.%20extended-context/README.md)    | 基於一般 NMT 架構，加入其他系統來和 NMT 合併，例如：<br/><ul><li>Multimodal NMT</li><li>Tree-based NMT</li><li>Graph-Structured Input NMT</li><li>Document-level Translation</li></ul>                                                                                                                   |
| [NMT-SMT Hybird System](methods/15.%20nmt-smt-hybrid/README.md) | 利用 SMT 依然優於 NMT 的部分，來和 NMT 結合互補。結合方法有：<br/><ul><li>SMT-supported NMT</li><li>System Combination</li><li>Others</li></ul>                                                                                                                                                          |

## Approaches

| **Approaches**                                                                                                                                                                    | **Description** |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| [Chinese–Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information Unsupervised neural machine translation](approaches/unsupervised_subcharacter.md) | TBD             |
| [Improving Character-level Japanese-Chinese Neural Machine Translation with Radicals as an Additional Input Feature](approaches/radical_feature.md)                               | TBD             |
| [LIT Team’s System Description for Japanese-Chinese Machine](approaches/data_preprocessing.md)                                                                                    | TBD             |
| [Octanove Labs’ Japanese-Chinese Open Domain Translation System](approaches/octanove.md)                                                                                          | TBD             |

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