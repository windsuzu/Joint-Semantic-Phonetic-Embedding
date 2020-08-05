# Using Large Output Vocabularies

單字的數量 => 影響 Embedding 大小 => 讓 model 膨脹，影響整個 model 的效能

大的 model 需要用`小的 batch size`，因為需要分配 GPU memory，但這會導致:

* Noisier gradients
* Slower training
* Large softmax output layer is very expensive
* Worse model performance

另外單字的 long-tail distribution 會讓實作變得困難，例如在 English-German WMT18 training set (140M words) 當中:

* 875K 個英文單字中的 843K (96.5%) 在資料集中只出現 100 次以下
* 代表 96.5% 的單字在整個資料中只出現了 `0.00007%` 

常見的解決方法是使用 `word-based NMT model`

* 限制字典大小為 `n`
* 剩餘的單字都改用 `<UNK>` 等 token 表示

缺點是:

1. 產生的翻譯可能出現 `<UNK>`
2. Out-Of-Vocabulary (OOV) 機率約為 3%
   * 35 個字就會出現一個 `<UNK>` 
3. `<UNK>` 有時太多會造成平衡問題，讓模型偏好輸出 `<UNK>`

要緩解 `<UNK>` 造成的影響有兩種方向: `translation-specific` 和 `model-specific`

## Translation-specific Approaches

這類方法保留長度為 n 的字典，然後針對 UNK token 做修正

1. 追蹤 `<UNK>` 在原始句子的位置，並利用訓練好的 `bilingual word-level dictionary` 來直接翻譯 `<UNK>`
   * `Addressing the rare word problem in neural machine translation`
   * `Neural machine translation systems with rare word processing`
   * 治標不治本，看字典方法不是好的翻譯方式
2. 將所有 OOV 單字使用相似的單字取代 (cosine similarity)
   * `Towards zero unknown word in neural machine translation`
   * 沒辦法 cover 到所有 OOV 單字
   * 取代的可能反而是差別很大的單字

> All share the inevitable limitation of all translation-specific approaches, namely that the translation model itself is indiscriminative between a large number of OOVs.

## Model-specific Approaches

直接改變 model 讓他可以和大型字典一起訓練

1. 新增 `lexical translation model` 到 NMT 當中，直接連接 source 和 target words
   * `Improving lexical choice in neural machine translation`
2. 取消使用 full softmax 來預測單字
   * `When and why are log-linear models self-normalizing?`
3. Noise-contrastive estimation (NCE) 訓練一個 logistic regression model 分辨真實訓練資料或是噪音
   * `Learning word embeddings efficiently with noise-contrastive estimation`
   * `A fast and simple algorithm for training neural probabilistic language models`
4. Vocabulary selection 利用 vocabulary 的 subset 來近似出 full softmax 的結果，subset 的選法有很多種 (e.g., important sampling)
   * `On using very large target vocabulary for neural machine translation`

> * Both softmax sampling and UNK replace have been used in one of the winning systems at the WMT’15 evaluation on English-German.
> * `Montreal neural machine translation systems for WMT’15`

# Character-based NMT





# Subword-unit-based NMT



# Words, Subwords, or Characters?



























<img src="https://latex.codecogs.com/png.latex?"/>
