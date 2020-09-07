# LIT Team’s System Description for Japanese-Chinese Machine Translation Task in IWSLT 2020

## Abstract

這篇論文講解作者 (LIT Team) 參加 IWSLT2020 open domain translation task (Japanese-Chinese) 的經驗分享，他們以 baseline system 為基準，加強了:

1. `data-preprocessing`
2. `large-scale back-translation on monolingual corpus`
3. `shared and exclusive word embeddings`
4. `different granularity of tokens (sub-word level)`

## Introduction

論文使用最流行的 `encoder-decoder architecture` (特別是 `transformer`)，這種架構在 `rich resource corpus` 的幫助下表現特別好。論文中使用了最大的 transformer 架構，因為 transformer 依賴 model capacity (the number of dimensions of the feed-forward network)。

論文主要的貢獻為 `data pre-processing`，特別是 `parallel data filter` 和 `sentence alignment`，透過將訓練資料品質加強，就可以提升翻譯水準。

論文採用了 `back-translation` (Edunov et al., 2018)，將中文翻譯成日文，擴充了 `Japanese-Chinese training corpus` 的大小，是運用 `monolingual datasets` 的好方法。

最後引入了 `Relative Position Attention` (Shaw et al., 2018)，還比較了 `shared embeddings` 和 `exclusive embeddings` 是否有差；在 C-J 的方向中使用 `shared embeddings` 效果較好，而 J-C 方向的效果則相反。

## Dataset

Dataset 由 `Japanese-Chinese bidirectional machine translation competition (Ansari et al., 2020)` 而來，為主辦單位提供的大量但有 noise 的 Japanese-Chinese pairs，這些資料從網頁上抓取而來，主要分成四個部分:

1. 對已有的 `Japanese-Chinese parallel datasets` 進行清理，得到雖然小但乾淨的資料集
2. 主辦者透過對網路爬蟲資料，進一步 `align`, `filter` 所得到的 `pre-filtered dataset`
3. Unfiltered parallel web crawled corpus
4. Huge file of the unaligned scraped web pages with the document boundaries

在之後的方法中會介紹怎麼處理這四個部分，而論文團隊也有使用 `back-translation` 在額外的 `monolingual data` 增加資料數量 (data augmentation)。

## Method

### Data pre-processing



#### Parallel Data Filter



#### Web Crawled Sentence Alignment



### Back-translation



### Model



## Result


