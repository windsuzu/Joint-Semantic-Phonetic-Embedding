# Using Monolingual Training Data

在翻譯實作中 `parallel training data` 是非常稀少的，但 `untranslated monolingual data` 卻非常豐富

所以在傳統的 SMT 中非常吃重 `language models (LMs)`，但簡單的 NMT 卻無法將 `LM` 及 `monolingual data` 引入

關於 NMT 引入 `LM` 和 `monolingual data` 的研究已有許多: 

1. 將訓練好的 `RNN-LM` 套入到 NMT decoder 中
   1. `On using monolingual corpora in neural machine translation`
2. 利用一個 `log-linear model` (shallow fusion) 來串接 `RNN-LM` 和 NMT score，更進一步利用 `controller network` (deep fusion) 來動態調整 RNN-LM 和 NMT 之間的權重
   1. `On integrating a language model into neural machine translation`
3. `Simple fusion` 先訓練一個翻譯模型來預測 training data 的 `residual probability` 再加到 pre-trained fixed LM 的預測 
   1. `Simple fusion: Return of the language model`

第二個研究方向是使用 `monolingual data` 作為 data augmentation，將 target language 的 `monolingual data` 加到要訓練的 `parallel training corpus`

有很多方法可以填充 source 這邊對應的句子:

1. 利用 single dummy token
   1. `Improving neural machine translation models with monolingual data`
2. 複製 target sentence 到 source 這邊
   1. `Copied monolingual data improves low-resource neural machine translation`
3. `Back-translation`
   1. `Investigations on large-scale lightly-supervised training for statistical machine translation.`

## Back-translation

Back-translation 利用另一個翻譯系統來反向從 target monolingual data 翻譯回 source sentence

* Back-translation system 通常比原本的系統網路要來得小和簡單
* 但若 back-transaltion system 的品質提升，可以對最終結果帶來好處
  * `Using monolingual data in neural machine translation`

Back-translation 現在已經被廣泛用於各種競賽當中:

* `Edinburgh neural machine translation systems for WMT 16`
* `Findings of the 2017 conference on machine translation (WMT17)`
* `Findings of the 2018 conference on machine translation (WMT18)`

Back-translation 的最大限制是合成回來的句子，必須和 real parallel data 的數量相當 (保持平衡)，所以通常只能用到一小塊 `monolingual data`，有一些人正在解決這些問題:

1. `Over-sampling` - 複製 real data 來符合合成的資料數量 (synthetic data size)
   * 大量的 over-sampling 反而會讓品質變差
2. 在 back-translation 句子中加入一些 noise，使這些句子更像 real data
   * `Understanding back-translation at scale`
   * `An efficient data augmentation algorithm for neural machine translation`
   * 效果非常好
3. 利用從 `back-translation model` 採樣一些句子來增加合成句子的多樣性
   * `Enhancement of encoder and attention using target monolingual corpora in neural machine translation`

## Others

第三種方向是修改 loss function 來讓模型能和 `monolingual data` 一起運作，例如加入 `autoencoder` 來計算翻譯品質 (`reconstruction error`)

* `Semi-supervised learning for neural machine translation`
* `Neural machine translation with reconstruction`
* `Autoencoder-based (self-attentive) universal language representation for machine translation`

另外還有:

1. 針對 source / target side 使用 multi-task learning
   * `Exploiting source-side monolingual data in neural machine translation`
   * `Using target-side monolingual data for neural machine translation through multi-task learning`
2. Warm start Seq2seq training
   * `Unsupervised pretraining for sequence to sequence learning`
   * `Semi-supervised neural machine translation with language models`
3. Unsupervised NMT 直接移除了 parallel training data 的需求
   * [12. Data-Sparsity](../12.%20data-sparsity/README.md)

