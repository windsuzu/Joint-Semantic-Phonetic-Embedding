# Multimodal NMT

因為語言本身就有模糊性，所以研究者就想要找到更多上下文 (context) 提供給翻譯系統，例如 source sentence 剛好在描述一張圖片，
該圖片就可能帶有隱藏的資訊能夠幫助翻譯

Multimodal MT 就是想使用 source + image 來同時翻譯 target sentence

目前大多的 multimodal MT 主要是一個 text-to-text 的系統，並且在 input 加入了從額外的 `computer vision model` 或 `visual attention` 所獲得的
 `global image features` 作為輔助

* `Multilingual image description with neural sequence models`
* `Multimodal pivots for image caption translation`
* `Attention-based multimodal neural machine translation`

目前已被證實使用圖片 (visual clues) 來幫助翻譯是有效的:

* `An error analysis for image-based multi-modal neural machine translation`

# Tree-based NMT

目前主要在 NMT 的處理單位大多為 `characters`, `subword` 為主流，這個設計並不是依照語言學，而是因為將 NMT 擴展到 `open vocabulary` 是非常困難的

若要依照語言學設計，翻譯應該被看作是更大元素之間的轉換，例如句子中的 `words`, `phrases`, `syntactic structures`

所以目前有許多研究嘗試在 source/target 端放入 `syntactic constituency trees` 或是 `dependency trees`，或是保持 sequence-to-sequence 的架構，並將架構線性化 (linearize) 為 `tree-structure`，方法有:

1. Bracket expressions
   * `Multi-source syntactic neural machine translation`
2. Sequence of rules
   * `Multi-representation ensembles and delayed SGD updates improve syntax-based NMT`
3. CCG supertags
   * `Predicting target language CCG supertags improves neural machine translation`
4. Packed forests represent multiple source sentence parses
   * `Forest-based neural machine translation`
5. Recurrent neural network grammars
   * `Recurrent neural network grammars`
6. Tree-LSTMs
   * `Improved semantic representations from tree-structured long short-term memory networks`
7. Convolutional encoders to represent a dependency graph
   * `Graph convolutional encoders for syntax-aware neural machine translation`
8. Biased encoder-decoder attention weights with syntactic clues
   * `Syntax-directed attention for neural machine translation`
9. Unsupervised tree-based methods
   * `Unsupervised recurrent neural network grammars`

# NMT with Graph Structured Input

更多的研究改良了基於 `tree-based NMT` 的 source sentence input:

1. Lattice-based NMT
   * `Lattices` 代表 `upstream components` (e.g., speech recognizer, tokenizer) 的不確定性
   * `Neural lattice-to-sequence models for uncertain inputs`
   * `Lattice-based recurrent neural network encoders for neural machine translation`
   * `Lattice-to-sequence attentional neural machine translation models` 
2. Factors
   * `Factors` 則利用 tuple 來代表更多單字的資訊 (e.g., lemma, prefix, suffix, POS, etc)
   * `Factored translation models`
   * `Linguistic input features improve neural machine translation`
   * `Factored neural machine translation architectures`
   * `Neural machine translation by generating multiple linguistic factors`

# Document-level Translation

某個研究中 (`Has machine translation achieved human parity?`) 提到，當人類取得較多上下文內容時，例如 `full-document` 時就能翻譯的比 SOTA sentence-level NMT 還要來得好，原因是人類在尋找跨句子的上下文比較優秀，因此許多技巧被用來尋找跨句子的上下文 (`intersentential context`):

1. Initializing encoder or decoder states
   * `Exploiting cross-sentence context for neural machine translation`
2. Multisource encoders as additional decoder input
   * `Does neural machine translation benefit from larger context?`
3. Memory-augmented neural networks
   * `Learning to remember translation history with a continuous cache`
   * `Document context neural machine translation with memory networks`
   * `Cache-based document-level neural machine translation`
4. Document-level LM
   * `CUED@WMT19:EWC&LMs`
5. Hierarchical attention
   * ` Document-level neural machine translation with hierarchical attention networks`
   * `Selective attention for context-aware neural machine translation`
6. Deliberation networks
   * `Modeling coherence for discourse neural machine translation`
7. Simply concatenating multiple source and/or target sentences
   * `Neural machine translation with extended context`
8. Context-aware extensions to Transformer encoders
   * `Context-aware neural machine translation learns anaphora resolution`
   * `Improving the Transformer translation model with document-level context`
