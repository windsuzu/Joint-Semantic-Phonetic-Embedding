# Improving Character-level Japanese-Chinese Neural Machine Translation with Radicals as an Additional Input Feature

## Abstract

在 `word-level NMT` 有許多人加入額外的特徵取得更好效果，但還沒有人在 `character-level NMT` 加入過額外特徵。在這篇論文中，作者在 `character-level NMT` 中加入了中文字的部首 (**radical**) 作為額外特徵，有效提升了翻譯效果。

在 WAT2016 Japanese-Chinese scientific paper excerpt corpus (ASPEC-JP) 這個資料集上訓練，得到了兩個方面的進步:

1. Perplexity
2. BLEU

## Introduction

### Logogram problem

`Word-level NMT` 的最大缺陷是架構造成的 `vocabulary size` 限制，另外在一些語言中 (例如中文和日文) 要獲得 `uniformed correct word segmentation` 是較為困難的，因為這些語言的文字 (**words**) 在書寫上是無法分割的。

一個語言的 `character` 則比 `word` 數量要少，且被廣泛應用在不同領域中，展示出 `character-level` 的優勢。接著又出現了介於 `character` 和 `word` 之間的 `subword`，將一個 `word` 拆成多個 `characters` 組成。

但中文字、日文漢字等屬於 `logograms`，指的是文字即代表了某樣事物的意義，所以非常難將這類文字 (word) 拆成 subword。

### Feature selection



9: features for word level nmt
we discovered radical is useful for character level nmt

Use WAT2016 system as baseline, 2 as nmt word level system, in our case we use character level



## Dataset
ASPEC-JC
WAT2016 Japanese Chinese scientific paper excerpt corpus

Abstract+some body
Not all fields

Training 672315
Development test 2148
Test 2107


## Method

### neural machine translation
Global attentional encoder decoder Network
Recurrent neural networks
Character level



### input features for Japanese characters
214 kangxi radicals in stroke count order
Hiragana and katakana also derived from radicals, get the radical from original kanji
Also deal with exceptions

Final source+radical (from cjklib)

### network


### evaluation



## Result


