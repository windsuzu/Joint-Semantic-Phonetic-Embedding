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

| **Approaches**                                                                                                                                                                    | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Chinese–Japanese Unsupervised Neural Machine Translation Using Sub-character Level Information Unsupervised neural machine translation](approaches/unsupervised_subcharacter.md) | 作者將 UNMT 運用於中日文這類 `logographic languages`，特別是將中日文切成更小的 `sub-character-level` 來實作。裡面運用到的最新方法為：<br/><ul><li>Shared BPE Embeddings</li><li>Encoder–Decoder Language Models</li><li>Back-Translation</li></ul>結果展示了 `sub-character` 和 `high token sharing rate` 的重要性，也點出了 quality metrics 的不足。                                                                                       |
| [Improving Character-level Japanese-Chinese Neural Machine Translation with Radicals as an Additional Input Feature](approaches/radical_feature.md)                               | 作者嘗試在 character-level NMT 加入額外特徵－部首 (radical)。因為中文屬於 `logograms`，無法拆成 `subword-level`，所以作者基於 `character-level` 找到了部首當作特徵。<br/>結果展示了部首當作特徵能提升效能，甚至翻譯出 reference 沒有翻譯成功的單詞。                                                                                                                                                                                        |
| [LIT Team’s System Description for Japanese-Chinese Machine](approaches/data_preprocessing.md)                                                                                    | IWSLT 2020 open domain translation task 的回饋，該 task 強調 **open domain** 的翻譯，並且給予大量含雜訊資料集，而作者使用了以下方法處理資料集：<br/><ul><li>Parallel Data Filter</li><li>Web Crawled Sentence Alignment</li><li>Back-translation</li></ul>並且對 baseline 模型進行了以下加強：<br><ul><li>Bigger Transformer</li><li>Relative Position Representation</li></ul>實驗結果每個方法都起到了幫助。                               |
| [Octanove Labs’ Japanese-Chinese Open Domain Translation System](approaches/octanove.md)                                                                                          | 同上為 IWSLT 2020 open domain translation task 的回饋，作者利用以下方法處理資料集：<br/><ul><li>Parallel Corpus Filtering</li><li>Back-Translation</li></ul>而模型做了以下處理：<br/><ul><li>Random parameter search</li><li>ensembling</li></ul>作者先對資料進行分析，並自訂 `rules` 和 `classifiers` 來去除不必要資料，且隨著 `back-translation` 使用率提高，獲得更好成積。<br/>另外也提出了 `top-k sampling` 與 `external data` 的幫助。 |
| CASIA’s System for IWSLT 2020 Open Domain Translation                                                                                                                             | <ul><li>[Video](https://slideslive.com/38929589/casias-system-for-iwslt-2020-open-domain-translation)</li><li>[PDF](https://www.aclweb.org/anthology/2020.iwslt-1.15/)                                                                                                                                                                                                                                                                      |


## Motivation

NMT 透過 attention, transformer 效能提升，再加上 back-translation, corpus filtering 技術進一步提升，還有什麼方法可以進一步提升。

受到兩篇論文的啟發:

1. Diversity by Phonetics and its Application in Neural Machine Translation
2. Robust Neural Machine Translation with Joint Textual and Phonetic Embedding

使用讀音資訊作為新的特徵。

## Datasets

| Provenance                                                                                                                           | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Link                                                                                                                                                                                                               | Date |
| ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---- |
| IWSLT 2020 - Open Domain Translation                                                                                                 | <ul><li>`existing_parallel` 包含: <ul><li>Global Voices</li><li>News Commentary</li><li>Ubuntu corpora (OPUS)</li><li>OpenSubtitles (Lison and Tiedemann, 2016)</li><li>TED Talks (Dabre and Kurohashi, 2017)</li><li>Wikipedia (Chu et al., 2014, 2015)</li><li>WikiMatrix (Schwenk et al., 2019)</li><li>Tatoeba.org</li></ul></li><li>`webcrawled parallel filtered` 包含 19M "可能"平行的資料</li><li>`webcrawled parallel unfiltered` 包含 161.5M 較差的平行資料</li><li>`webcrawled unaligned` 包含 15.6M 各種網站兩種語言的"可能"平行資料</li></ul> | <ul><li>[PDF](https://www.aclweb.org/anthology/2020.iwslt-1.1.pdf#page=12)</li><li>[Dataset](https://github.com/didi/iwslt2020_open_domain_translation)</li></ul>                                                  | 2020 |
| [WAT 2020 The 7th Workshop on Asian Translation](http://lotus.kuee.kyoto-u.ac.jp/WAT/WAT2020/index.html) (Unfree)                    | 包含三大資料集，但都需要寄信審核索取。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | <ul><li>[ASPEC](http://lotus.kuee.kyoto-u.ac.jp/ASPEC/)</li><li>[JPO Patent Corpus](http://lotus.kuee.kyoto-u.ac.jp/WAT/patent/)</li><li>[JIJI Corpus](http://lotus.kuee.kyoto-u.ac.jp/WAT/jiji-corpus/)</li></ul> | 2020 |
| Japanese to English/Chinese/Korean Datasets for Translation Quality Estimation and Automatic Post-Editing                            | 簡單資料集，用來測試 QE 和 APE 的。 (travel (8,783 segments), hospital (1,676 segments))                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | <ul><li>[PDF](https://www.aclweb.org/anthology/W17-5705.pdf)</li><li>[Dataset](http://paraphrasing.org/~fujita/resources/NICT-QEAPE.html)</li></ul>                                                                | 2017 |
| Inflating a Small Parallel Corpus into a Large Quasi-parallel Corpus Using Monolingual Data for Chinese-Japanese Machine Translation | 有講解中日資料集缺乏的問題，還有他解決的方法。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | [PDF](https://www.jstage.jst.go.jp/article/ipsjjip/25/0/25_88/_article/-char/en)                                                                                                                                   | 2017 |
| Constructing a Chinese—Japanese Parallel Corpus from Wikipedia                                                                       | 從維基百科自動擷取的中日平行資料，包含:<ul><li>126,811 parallel sentences</li><li>131,509 parallel fragments</li><li>198 dev</li><li>198 test</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                    | <ul><li>[PDF](https://www.aclweb.org/anthology/L14-1209/)</li><li>[Dataset](http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?Wikipedia%20Chinese-Japanese%20Parallel%20Corpus)</li></ul>                                | 2015 |
| JEC Basic Sentence Data                                                                                                              | Excel file containing all the sentences in Japanese, English and Chinese, it contains: 5304 sentences                                                                                                                                                                                                                                                                                                                                                                                                                                                      | [Dataset](http://nlp.ist.i.kyoto-u.ac.jp/EN/index.php?JEC%20Basic%20Sentence%20Data)                                                                                                                               | 2011 |

## Method

1. Feature Extraction
   1. Bopomofo
   2. Hiragana
2. Data Preprocessing
   1. Escape character transformation
   2. Numbers and punctuation normalization
   3. Segmentation
      1. Jieba (Chinese) 
      2. Mecab (Japanese)
   4. BPE for subword tokenization
3. Embedding
   1. Semantic
   2. Phonetic
   3. Mixed
   4. Unsupervised Relation Binding
4. Transformer, ConvS2S
5. Metrics
   1. BLEU
   2. Quality Estimation
6. Data Augmentation
   1. Corpus Filtering
   2. Back-Translation

## Experiments


## Results


## Other Resources


| Field            | Description                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DeepL            | [DeepL 已經很好，但在中日翻譯還是有問題](https://techcrunch.com/2017/08/29/deepl-schools-other-online-translators-with-clever-machine-learning/)                                                                                                                                                                                                                                                                             |
| Transformer      | [淺談神經機器翻譯 & 用 Transformer 與 TensorFlow 2 英翻中](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)                                                                                                                                                                                                                                                                              |
| NMT Review       | [Neural Machine Translation: A Review](https://arxiv.org/pdf/1912.02047.pdf)                                                                                                                                                                                                                                                                                                                                                 |
| Unsupervised NMT | <ul><li>Task: [Unsupervised Machine Translation](https://paperswithcode.com/task/unsupervised-machine-translation)</li><li>Paper: [Phrase-Based & Neural Unsupervised Machine Translation](https://paperswithcode.com/paper/phrase-based-neural-unsupervised-machine)</li></ul>                                                                                                                                              |
| Low resource NMT | <ul><li>Task: [Low-Resource Neural Machine Translation](https://paperswithcode.com/task/low-resource-neural-machine-translation)</li><li>Paper: [The FLoRes Evaluation Datasets for Low-Resource Machine Translation](https://paperswithcode.com/paper/two-new-evaluation-datasets-for-low-resource)</li></ul>                                                                                                               |
| Reinforcement    | <ul><li>Paper: [A Reinforced Generation of Adversarial Examples for Neural Machine Translation](https://www.aclweb.org/anthology/2020.acl-main.319/)</li></ul>                                                                                                                                                                                                                                                               |
| Adversarial      | <ul><li>Paper: [AdvAug: Robust Adversarial Augmentation for Neural Machine Translation](https://www.aclweb.org/anthology/2020.acl-main.529/)</li><li>Paper: [Adversarial and Domain-Aware BERT for Cross-Domain Sentiment Analysis](https://www.aclweb.org/anthology/2020.acl-main.370/)</li></ul>                                                                                                                           |
| Multimodal       | <ul><li>Task: [Multimodal Machine Translation](https://paperswithcode.com/task/multimodal-machine-translation)</li></ul>                                                                                                                                                                                                                                                                                                     |
| CNN              | <ul><li>SOTA: [Machine Translation on WMT 2017 English-Chinese](https://paperswithcode.com/sota/machine-translation-on-wmt-2017-english-1)</li><li>Paper: [Pay Less Attention with Lightweight and Dynamic Convolutions](https://paperswithcode.com/paper/pay-less-attention-with-lightweight-and)</li><li>Notes: [Pay less attention with light-weight &dynamic CNN](https://zhuanlan.zhihu.com/p/60482693)</li></ul>       |
| Score Metrics    | <ul><li>[Bleu: a Method for Automatic Evaluation of Machine Translation](https://www.aclweb.org/anthology/P02-1040/)</li><li>[A Call for Clarity in Reporting BLEU Scores](https://paperswithcode.com/paper/a-call-for-clarity-in-reporting-bleu-scores)</li><li>[Beyond BLEU: Training Neural Machine Translation with Semantic Similarity](https://paperswithcode.com/paper/beyond-bleu-training-neural-machine)</li></ul> |
| WMT 20           | http://www.statmt.org/wmt20/index.html                                                                                                                                                                                                                                                                                                                                                                                       |



## Idea

| Field     | Description                                     |
| --------- | ----------------------------------------------- |
| Adversial | 或許能用一個文法 discriminator 來強化 generator |
| Phonetic  | 利用語音作為新的特徵，同時和文字訓練            |

## Progress

- [x] Learning Training skills
- [x] Implement a new tf transformer translation system
- [ ] **Collect datasets**
- [ ] Training with a transformer, ConvS2S library
- [ ] Evaluation (BLEU, perplexity, other semantic metrics, blind testing...)
- [ ] Update the system with some skills and features, new ideas
- [ ] **Contribute a different evaluation method**
- [ ] Identify the problem, find a solution, or leave a vision
