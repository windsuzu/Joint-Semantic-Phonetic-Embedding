# NMT Model Size

NMT 模型通常會有上千萬個 `parameters`，而能使用的 GPU 數量卻有限制，所以只好從模型大小下手了

較小的模型不但能降低 `computational complexity`，而且能透過增加 `batch size` 來更好的運用 `GPU parallelism`

另外模型的檔案需要非常大的硬碟空間來存取，例如一個有效的方式是 `neural architecture search`，其他用來解決硬體空間限制的方法還有:

1. 利用 **systematic neural architecture search** 來找到 `computationally efficient Transformer hyper-parameters`
   * `A survey of multilingual neural machine translation`
2. 將運算中的 32-bit 浮點數調整為 8 或 16 bits
   * `Fast neural machine translation implementation`
   * `Scaling neural machine translation`
   * `Pieces of eight: 8-bit neural machine translation`
   * `Fast and accurate neural machine translation decoding on the CPU`
3. 使用 `vector quantization`
   * `Quantized convolutional neural networks for mobile devices`

> * 什麼是 neural architecture search ?
> * [Neural architecture search](https://en.wikipedia.org/wiki/Neural_architecture_search)
> * [提煉再提煉濃縮再濃縮：Neural Architecture Search 介紹](https://medium.com/ai-academy-taiwan/%E6%8F%90%E7%85%89%E5%86%8D%E6%8F%90%E7%85%89%E6%BF%83%E7%B8%AE%E5%86%8D%E6%BF%83%E7%B8%AE-neural-architecture-search-%E4%BB%8B%E7%B4%B9-ef366ffdc818)

很多將模型修剪縮小的方法在 30 年前就有了，也被證實過網路中的權重有大多數是多餘的:

* `Optimal brain damage`
* `Pruning algorithms of neural networks`
* `Compression of neural machine translation models via pruning`

其中一個研究是 `remove umimportant network connections`，這些 connections 可以用以下方法選擇:

1. second-derivative of the training error with respect to the weight
   * `Second order derivatives for network pruning: Optimal brain surgeon`
2. threshold criterion on its magnitude
   * `Learning both weights and connections for efficient neural network`

其他改良 model size 的方法還有:

1. 移除重複性較大且權重較小的 `neurons`
   * `Data-free parameter pruning for deep neural networks`
2. 在訓練時合併相似度較大的 `neurons`
   * `A simple way to prune neural networks`
3. 利用 `low rank matrics` 來壓縮網路，特別是使用 `SVD` 來近似
   * `Predicting parameters in deep learning`
   * `Exploiting linear structure within convolutional networks for efficient evaluation`
   * `Restructuring of deep neural network acoustic models with singular value decomposition`
   * `On the compression of recurrent neural networks with an application to LVCSR acoustic modeling for embedded speech recognition`
   * `Learning compact recurrent neural networks`

最後一個方法是 `knowledge distillation`，利用一個較大的網路 (teacher) 來產生 `soft training labels` 給予另一個較小的網路 (student)，而 `student network` 是通過最小化 teacher 的 `cross-entropy` 來訓練

* `Model compression`
* `Distilling the knowledge in a neural network`

這個方法已經被使用在多種 `sequence modelling task` 上面，例如翻譯和語音辨識:

* `Sequence student-teacher training of deep neural networks`
* `Ensemble distillation for neural machine translation`
* `Analyzing knowledge distillation in neural machine translation`
* `Knowledge distillation using output errors for self-attention end-to-end models`
* `End-to-end speech translation with knowledge distillation`

> 什麼是 knowledge distillation ?
> * [Knowledge Distillation : Simplified](https://towardsdatascience.com/knowledge-distillation-simplified-dd4973dbc764)
> * [Noisy Student: Knowledge Distillation強化Semi-supervise Learning](https://medium.com/%E8%BB%9F%E9%AB%94%E4%B9%8B%E5%BF%83/deep-learning-noisy-student-knowledge-distillation%E5%BC%B7%E5%8C%96semi-supervise-learning-4e0c2d11520a)