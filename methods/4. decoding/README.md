# The Search Problem in NMT

NMT 要計算的 <img src="https://latex.codecogs.com/png.latex?P(y\mid%20x)"/> (translation probability) 需要給定 `x` 和 `y`，但實作中並不會知道 y，得出這個 y 可說是機器翻譯的最終目標

利用 source sentence `x` 來求出最接近的 <img src="https://latex.codecogs.com/png.latex?\hat{y}"/> 的這個任務叫做 **decoding** 或是 **inference**

<img src="https://latex.codecogs.com/png.latex?\hat {y}=\argmax_{y\in\sum}P(y\mid%20x)"/>

NMT decoding 有兩大重點:

1. Search space 隨著字典大小增加而指數成長
   * 例如當字典大小為 32000 時 <img src="https://latex.codecogs.com/png.latex?\lvert\sum_{trg}\rvert=32000"/>
   * 要翻譯出 20 個字的時候，就需要在 32000^20 個可能性中搜尋 (大於宇宙中的原子 10^82)

2. NMT 模型犯錯是時常的事情，Search 越深反而會產生較差的翻譯
   * `On NMT search errors and model errors: Cat got your tongue?`

# Greedy and Beam Search

NMT 最常用的 decoding 演算法是 `greedy search` 和 `beam search`

翻譯永遠是從左到右，隨著評分 <img src="https://latex.codecogs.com/png.latex?P(y_j\mid%20y_1^{j-1},x)"/> 來進行翻譯，也就是說翻譯都是與時間同步的 (time-synchronous)

![](../../assets/nmt_decoding.png)

## Greedy Search (Green)

Greedy search 會在每一個 time step 選取最好的 (綠色線所示)

* j=1 時選取 `c`
* j=2 時選取 `a`
* j=3 時選取 `b`

Greedy search 的最大問題是 `garden-path problem` (`P. Koehn, Neural machine translation`)

若是在 j=1 選的 `c` 是錯誤的，那麼後面選擇的單字就會整組壞掉

## Beam Search (Orange)

Beam search 就是為了緩解 greedy search 的問題而出現的，每次會選取 `n` 個分數最好的 `translation prefix`

例如上圖就是以 n=2 下去找出最好的翻譯，被選到的 n 個當前最好的前綴句子稱為 `active hypotheses`

儘管 beam search 看起來已經解決了 greedy search 的缺點，但事實上依然存在 `garden-path problem` (`On NMT search errors and model errors: Cat got your tongue?`)

## Formal Description of Decoding for the RNNsearch Model

解釋 OneStepRNNsearch P(y|x) 用於 greedy & beam search 

## Ensembling

ensembling 用 K 個 nmt model 並使用 arith, geo 來合併結果

Sarith, Sgeo 可以取代 equ.5

Sarith 合理，但 Sgeo 較快，因為 log 在合併不用轉換

最先進的 NMT 都使用 ensembling，例如 tencent 72 model

Ensembling 缺點：

1. worse speed
2. difficult to imple

在 13. model size 中有講到 `knowledge distillation` 可以用於減緩 ensembling 缺點

通常在 ensembling 的所有 model 都是使用相同 size, training data，只有改變 random weight initialization 和 randomized order of training samples

每個 ensembling model 會犯不同的錯，但又可以被其他 model 省略掉 (156)
這很合理因為 NMT 的翻譯品質在不同訓練會差距很大 (157)

NMT loss surface 往往是 highly non-convex 無法到達 local optima
而 ensembling 可以緩解這個問題，甚至能達到 regularization (158)

### Checkpoint Averaging

checkpoint averaging 常被和 ensembling 一起討論

checkpoint averaging 會追蹤訓練時的 checkpoint 將 weight matrices 平均作為最終矩陣，不增加 decoding 時間，在 NMT 常被使用 (76, 126, 161)

和 ensembling 處理不同的問題，主要是修正 training curve 的 minor fluctuation
造成原因是

1. optimizer's update rule
2. mini-batch training 下 gradient estimation 的 noise

因為個別獨立的 model 相差很大，Checkpoint averaing 無法用在 independently trained models

## Decoding Direction




<img src="https://latex.codecogs.com/png.latex?"/>