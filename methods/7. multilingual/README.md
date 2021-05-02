# Multilingual NMT

NMT 通常只處理兩種語言 **1 對 1 直接的翻譯**，而 multi-lingual NMT 旨在使用單個 model 就能在**多種語言之間交叉翻譯**

1. 能挖掘語言之間的相似性 (similarities)
2. 減少要在多語言翻譯所需的系統數量 (number of systems)

`Multilingual NMT` 系統根據共享的元件 (shared components) 不同而有所區分:

1. 共享整個網路架構 (encoder & decoder)，且使用 `annotating sentences, words` 和 `language ID tags` 或 `embedding` 來表示 source/target 語言
   * `Google’s multilingual neural machine translation system: Enabling zero-shot translation`
2. 每個語言都有自己專屬的 encoder 和 decoder
   * `Multi-task sequence to sequence learning`
3. 基於上面的系統 (每個語言都有 encoder & decoder)，加入 attention 機制讓每種語言共享 attention
   * `Multi-way, multilingual neural machine translation with a shared attention mechanism`
4. `One-to-many` translation: 只有單個 encoder 但有多個 decoder 給各種語言
   * `Multi-task learning for multiple language translation`

`Multilingual NMT` 的一個優勢是 `zero-shot translation` (翻譯兩種語言，該兩種語言完全沒有訓練資料)

例如由 multilingual 產生的 zero-shot translation，在學過 `Portuguese ⬄ English` 和 `Spanish ⬄ English` 過後，能夠直接翻譯 `Portuguese ⬄ Spanish` 但是沒有比 `pivot-based zero-shot translation` 效果好

Pivot-based zero-shot translation 指的是先翻譯到某個中間語言，再翻譯成目標語言，例如上面的例子就是 `Portuguese > English > Spanish` 而 English 就是中間語言

Pivot-based zero-shot translation 可以再透過一些技巧強化:

1. Fine-tuning on `Pseudo parallel corpus`
   1. `Zero-resource translation with multi-lingual neural machine translation`
2. 合併訓練一個 `source-pivot` 及 `pivot-target` 之間的元件 (e.g., word embedding matrices)
   1. `Neural machine translation of rare words with subword units`
3. `"Neural interlingual"` component between encoder & decoder
   1. `A neural interlingua for multilingual machine translation`
4. 更多關於 multilingual 用於 zero-translation 的技巧
   1. `A comparison of Transformer and recurrent neural networks on multilingual neural machine translation`
   2. `Massively multilingual neural machine translation`
   3. `Overview of the IWSLT 2017 evaluation campaign`

`Multilingual NMT` 的另一個形態是 `Multi-source NMT`，系統使用多個 source language 來翻譯一個 target language

`Multi-source` 的架構可以用於很多地方: 

1. Multi-modal NMT
2. Morphological inflection
3. Zero-shot translation
4. Low-resource MT
5. Syntax-based NMT
6. Document-level MT
7. Bidirectional decoding

但 `Multi-source NMT` 有一問題是當 source data 不足時 (data sparsity)，需要合成 source data 來補齊
