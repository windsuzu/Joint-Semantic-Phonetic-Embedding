# TODO

- [ ] Learning Training skills
  - [ ] Convolutional Neural Machine Translation (ConvS2S)
  - [ ] Neural Machine Translation Decoding
    - [ ] Greedy & Beam Search
    - [ ] Generating Diverse Translations
    - [ ] Simultaneous Translation
  - [ ] Dataset
    - [ ] Large Output Vocabularies
    - [ ] Parallel Corpus
    - [ ] Character-based NMT
    - [ ] Subword-unit-based NMT (radical, ideograph, stroke)
    - [ ] Monolingual Corpus
    - [ ] Back-translation
  - [ ] Training
    - [ ] Large Batch Learning
    - [ ] Reinforcement Learning
    - [ ] Dual Supervised Learning
    - [ ] Adversarial Learning
  - [ ] Interpretability
    - [ ] Post-hoc
    - [ ] Model-intrinsic
    - [ ] Confidence Estimation
    - [ ] Word Alignment
  - [ ] Data Sparsity
    - [ ] Corpus Filtering
    - [ ] Domain Adaptation
    - [ ] Low-resource
    - [ ] Unsupervised
  - [ ] Multilingual NMT
  - [ ] Extended Method
    - [ ] Multimodal
    - [ ] Tree-based
    - [ ] Graph structured input
    - [ ] document-level
- [ ] Implement a new tf transformer translation system
- [ ] Evaluate the performance of the system
  - [ ] BLEU
- [ ] Update the system with some skills and features
- [ ] Think of a new approach
- [ ] Evaluation
- [ ] Identify the problem, find a solution, or leave a vision

# BPE

# Seq2Seq

https://easyai.tech/ai-definition/encoder-decoder-seq2seq

* 只要是輸入序列、輸出序列就是 seq2seq
* seq2seq 算是 Encoder-Decoder 的一種
* 簡單的 attention 介紹


https://blog.csdn.net/u014595019/article/details/52826423

* Encoder-Decoder 的不足
* Attention 的補足
* 但解碼看不太懂

# Attention, Self-attention, Multi-head attention, Transformer

https://zhuanlan.zhihu.com/p/47063917

* RNN 和 Attention 的結合


https://zhuanlan.zhihu.com/p/47282410

* Self-attention 捨棄 RNN 只留 attention
  * 每個字和其他字的關係產生 attention score
  * 在 encoder 會層層疊加 self-attention
  * 在 decoder 不僅看前一個輸出字，也會看 encoder 的
  * self-attention 由 QKV 組成算出 Z (https://www.zhihu.com/question/325839123)
    * Query = 自己
    * Key = 所有字
    * Value = 該 key 的價值 (通常 k=v)
    * WQ, WK, WV 就是要訓練的參數
* Multi-head attention 只是很多個 self-attention 結合而已
  * 也就是可以有多組 QKV
  * 每個 self-attention 關注的點可能不同
  * 最後還是會組成一個 Z
* Mask multi-head attention
  * 放在第一層，輸入來自前一層的 decoder
  * Mask 讓我們只能 attend 翻譯過的 encoder
  * 在預測第 t 個詞的時候要把 t+1 到末尾的詞遮住，只對前面 t 個詞做 self-attention
* Encoder-Decoder attention layer
  * Q 來自前一個 decoder 輸出
  * KV 來自 encoder 輸出
  * 讓每個位置的 decoder 都能對到 input 的每個位置
* Position encoding
  * 因為沒有 CNN 或 RNN 無法知道字的位置
  * 所以加一個 encoding 在 embedding 上，知道字的位置
* Add & Norm
  * Residual connection
  * Layer normalization


https://zhuanlan.zhihu.com/p/47613793

* Universal Transformers
  * 加入 transition function 來循環 attention
* BERT
  * 雙向 transformer
* Generating Wikipedia by Summarizing Long Sequences
  * 讀很多文章，自動生出 wiki 風格的內容
* Show, Attend and Tell
  * 影像處理 (caption) 才是 attention 最早使用的地方
* Image Transformer
  * 用 attention 圖像合成，還原解析度




https://techcrunch.com/2017/08/29/deepl-schools-other-online-translators-with-clever-machine-learning/

指出 deepL 已經很好，但在中日翻譯還是有問題

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

7.
parallel corpus filtering
back translation

6.
data processing
Large-scale back-translation on monolingual corpus
exclusive word embedding
different granularity of tokens like sub-word level

5.
additional linguistic features in character-level NMT
radicals of Chinese characters (or kanji)

4.
Unsupervised neural machine translation
logographic language pairs
Sub-character Level Information (ideograph or stroke)

sub-character-level > character-level
stroke > ideograph


# Measurement, Contribution

BLEU => transformer-only implementation
=> try to break it

Blind testing through questionnaire


# Progress

1. 先搞懂所有 translation 訓練方法
2. 試做基本翻譯系統 (transformer, CNN)
3. 測試他們的分數 (SOTA baseline, Japanese-Chinese baseline)
4. 開始加入一些特徵
5. 使用自己的方法
6. 測試分數、若不夠好再加上盲測
7. 實際找出問題，想辦法解決、或留下未來展望