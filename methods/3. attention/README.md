# Attention

單純的 encoder-decoder model 在產生長句子的翻譯時效果很爛，原因來自 `fixed-length source sentence encoding`

* 固定長度的 source encoding 不能給 decoder 足夠資訊來翻譯
* 一開始有人把句子拆成多個子句，翻譯後再合併，但效果不好
  * `Overcoming the curse of sentence length for neural machine translation using automatic segmentation`
* 另一個想法是 attention 不再使用固定長度的 encoding <img src="https://latex.codecogs.com/png.latex?c(x)"/>
  * Attentional decoder 可以只看需要的 encoder 資訊來翻譯
  * <img src="https://latex.codecogs.com/png.latex?c(x)"/> 變成一系列的 <img src="https://latex.codecogs.com/png.latex?c_j(x)"/> 其中 j 代表時間點

## Intuitive

Attention 的概念就像: 

* Input: n `query vectors`
* Mapping: Mapping Table (m `key-value pairs`)
* Output: n `output vectors`

每個 vector 都是 d 維，然後可以堆疊起來，所以有三大矩陣: 

* <img src="https://latex.codecogs.com/png.latex?Q\in\mathbb{R}^{n\times%20d}"/>
* <img src="https://latex.codecogs.com/png.latex?K\in\mathbb{R}^{m\times%20d}"/>
* <img src="https://latex.codecogs.com/png.latex?V\in\mathbb{R}^{m\times%20d}"/>

直覺上，每個 `query vector` 在算出 `output vector` 前會先變成 `value vector` 的權重，這個權重由 query 跟所有 keys 計算 `similarity score` 得到

* <img src="https://latex.codecogs.com/png.latex?\underbrace{\text{Attention}(K,V,Q)}_{n\times%20d}=\text{Softmax}(\underbrace{\text{score}(Q,K)}_{n\times%20m})\underbrace{V}_{m\times%20d}"/>

* `score(Q, K)` 是一個 similarity score matrix (n, m)
* 會使用 softmax 將每個 column 都一般化，代表每個 `query vector` 的權重加總為一
* `score()` 最常使用的計算方式是 dot product

## Method

* Decoder hidden state (<img src="https://latex.codecogs.com/png.latex?s_j"/>) 是 `query vectors`
* Encoder hidden state (<img src="https://latex.codecogs.com/png.latex?h_i"/>) 是 `key, value vectors`

如此一來:

* <img src="https://latex.codecogs.com/png.latex?Q=s_j"/> 為 `query vectors`
  * <img src="https://latex.codecogs.com/png.latex?n=J"/> 為 target sentence length
* <img src="https://latex.codecogs.com/png.latex?K=V=h_i"/> 為 `key, value vectors`
  * <img src="https://latex.codecogs.com/png.latex?m=I"/> 為 source sentence length

Attention layer 產出來的 output 為 `time-dependent context vectors` <img src="https://latex.codecogs.com/png.latex?c_j(x)"/>

* 每個時間點 j 我們會 query 原句子一次
* 然後最終能得出一個 `attention matrix` <img src="https://latex.codecogs.com/png.latex?A\in\mathbb{R}^{J\times%20I}"/>
* 從 A 可以看出翻譯句子和原句子之間的關係

以下是一個 H=3 的 multi-head model:

![](../../assets/attention_matrix.png)

## Multi-head Attention

> * Attention is all you need

Multi-head attention 是由 `H` (通常為 8) 個 attention head 組合而成

* 一個 attention head 的 `query, key, value vectors` 都是 Q, K, V 的 linear transforms
* Multi-head attention 的輸出就是這 H 個 attention heads 的 concatenation
* Attention heads 的維度通常會除以 H 避免參數過多

<img src="https://latex.codecogs.com/png.latex?\text{MultiHeadAttention}(K,V,Q)=\text{Concat}(\text{head}_1,\cdots,\text{head}_H)W^O"/>


* 其中的 weight matrix 是 <img src="https://latex.codecogs.com/png.latex?W^O\in\mathbb{R}^{d\times%20d}"/>
* 每個 head 是 <img src="https://latex.codecogs.com/png.latex?\text{Attention}(KW_h^K,VW_h^V,QW_h^Q)"/>
  * <img src="https://latex.codecogs.com/png.latex?W_h^K,W_h^V,W_h^Q\in\mathbb{R}^{d\times\frac{d}{H}}\text{%20for%20}h\in[1,H]"/> 都是權重，由網路訓練

![](../../assets/multi-head_attention.png)

Multi-head attention 雖然效能較好，但無法像 attention 輕鬆產出 attention matrix，所以較難解釋

# Attention Masks and Padding

通常我們會將句子裝成 batches 加快訓練速度、減少 gradient 的 noise

而格式為 tensor 的句子，因為需要固定長度，所以在較短句子的後面會加入 `<pad>` 來補足

![](../../assets/sentence_batch_padded_attention.png)

為了不讓 `<pad>` 被納入 attention 的計算，所以會加入 mask 的系統

* 紅色的 `<pad>` 的 mask 都會是 0
* 綠色的單字的 mask 都會是 1

計算中的 attention weight 就會乘上 mask 來忽略 `<pad>`

# Recurrent Neural Machine Translation

RNNsearch 是第一個使用 attention 的 NMT model，預測 <img src="https://latex.codecogs.com/png.latex?P(y\mid%20x)"/> 中的 <img src="https://latex.codecogs.com/png.latex?c(x)"/> 改變成 <img src="https://latex.codecogs.com/png.latex?c_j(x)"/>:

<img src="https://latex.codecogs.com/png.latex?P(y\mid%20x)=P(y_j\mid%20y_1^{j-1},x)=g(y_j\mid%20s_j,y_{j-1},c_j(x))"/>

為了預測 <img src="https://latex.codecogs.com/png.latex?y_j"/> 的分布，使用了以下的資訊:

* 前一次產生的 token <img src="https://latex.codecogs.com/png.latex?y_{j-1}"/>
* RNN decoder state <img src="https://latex.codecogs.com/png.latex?s_j\in\mathbb{R}^n"/>
* Context vector <img src="https://latex.codecogs.com/png.latex?c_j(x)\in\mathbb{R}^m"/>
  * 代表 target token 在 source sentence 關注 (attention) 某一部分的 distributed representation
  * 解決了 encoder-decoder 只使用一個 context vector 的缺點

![](../../assets/rnnsearch_model.png)

## Context Vector

Context vector <img src="https://latex.codecogs.com/png.latex?c_j(x)"/> 是 source sentence 的 multi-head <img src="https://latex.codecogs.com/png.latex?\mathbf{h}=(h_1,\cdots,h_I)"/> 的加總 

也就是 encoder 將 input sequence `x` 轉換成一系列長度相等的 `h`，而 <img src="https://latex.codecogs.com/png.latex?h_i\in\mathbb{R}^m"/> 代表了 `x` 的資訊並且是以第 `i` 個字為重點表示

RNNsearch 是使用 bidirectional RNN 來產生 annotation (`h`)，`f(.)` 通常是 LSTM 或 GRU

<img src="https://latex.codecogs.com/png.latex?\begin{aligned}h_i&=\left[\overrightarrow{h_i};\,\overleftarrow{h_i}\right]\\\overrightarrow{h_i}&=\overrightarrow{f}(x_i,\overrightarrow{h_{i-1}})\\\overleftarrow{h_i}&=\overleftarrow{f}(x_i, \overleftarrow{h_{i+1}})\end{aligned}"/>

Context vector <img src="https://latex.codecogs.com/png.latex?c_j(x)"/> 就是 `h` (annotations as weighted sum) 和 weights <img src="https://latex.codecogs.com/png.latex?\alpha_j\in\left[0,1\right]^I"/> 的相乘:

<img src="https://latex.codecogs.com/png.latex?c_j(x)=\sum_{i=1}^I\alpha_{j,i}h_i"/>

每個 weights <img src="https://latex.codecogs.com/png.latex?\alpha"/> 由 alignment model `a(.)` 計算出來

<img src="https://latex.codecogs.com/png.latex?\alpha_{j,i}=\frac{1}{Z}\exp(a(s_{j-1},h_i))\text{%20with%20}Z=\sum_{k=1}^I\exp(a(s_{j-1},h_k))"/>

其中的 <img src="https://latex.codecogs.com/png.latex?a(s_{j-1},h_i)"/> 是一個 feedforward neural network

* 在給定當下 decoder state <img src="https://latex.codecogs.com/png.latex?s_{j-1}\in\mathbb{R}^n"/> 情況下
* 估計 annotation <img src="https://latex.codecogs.com/png.latex?h_i"/> 對第 `j` 個 target token 的重要性有多大

用 Pure Attention 的術語來表示的話:

* <img src="https://latex.codecogs.com/png.latex?h_i"/> 代表 keys, values
* <img src="https://latex.codecogs.com/png.latex?s_j"/> 代表 queries
* `a(.)` 代表 attention scoring function

## g Function

<img src="https://latex.codecogs.com/png.latex?P(y\mid%20x)=P(y_j\mid%20y_1^{j-1},x)=g(y_j\mid%20s_j,y_{j-1},c_j(x))"/>

現在可以回到最前面了解 g function 做的事情:

* 吃了前一個 target token <img src="https://latex.codecogs.com/png.latex?y_{j-1}"/>
* 吃了 context vector <img src="https://latex.codecogs.com/png.latex?c_j"/>
* 吃了 decoder hidden state <img src="https://latex.codecogs.com/png.latex?s_j"/>
  * <img src="https://latex.codecogs.com/png.latex?s_j=f(s_{j-1},y_{j-1},c_j)"/>
  * 其中 `f(.)` 可以是 GRU, LSTM cell model

整個 g 可以表達成下列式子:

<img src="https://latex.codecogs.com/png.latex?\begin{aligned}&g(y_j\mid%20y_{j-1},s_j,c_j)\propto\exp(W_o\max(t_j,u_j))\\t_j&=T_ss_j+T_yEy_{j-1}+T_cc_j\\u_j&=U_ss_j+U_yEy_{j-1}+U_cc_j\end{aligned}"/>

其中的大 `T, U, W, E` 都是 weight matrices

![](../../assets/rnnsearch_attention_model.png)

`g(.)` 的定義可以看作是將 `output of the recurrent layer` 和 `previous target token` (k-dimensional embedding) 還有 `context vector`，用一個 size l 的 maxout layer 連接起來

# Convolutional Neural Machine Translation

CNN 非常適合用於影像處理任務上，原因有二:

1. High degree of `weight tying` 減少參數量
2. Automatically learn space `invariant features` 找出影像中的物件

CNN 一樣可以用於 NLP 的一維資料，但我們會省略 pooling, strides (不常用於一維資料)


input u1...uI 個 M-dimensional vectors (i 代表空間維度 i-axis,  M 代表 channels)

convolution 將 input 轉換成 N-dimensional vectors
做法是用長度為 K 的 kernel 跑過 input 一遍
kernel 將 k-gram ui, ... ui+k-1 轉成 vi






<img src="https://latex.codecogs.com/png.latex?"/>

# 補充

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