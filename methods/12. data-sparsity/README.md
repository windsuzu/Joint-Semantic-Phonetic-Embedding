Deep Learning 最需要的還是資料，缺少資料的情況下大部分的 SMT 還是勝過 NMT 的，在這個章節將會研究各種 `data sparsity` 的問題，例如 `noise reducing`, `different domain data`, `less or no parallel data`

# Corpus Filtering

因為 MT training data 往往是自動或半自動從爬蟲而來，所以會有很多 noise，例如會抓到:

1. Sentence fragments
2. Wrong languages
3. Misaligned sentence pairs
4. MT output (並非真正的 parallel text)

雖然在 back-translation 的合成資料中加入 noise 可以提升效能，而 noise 也能幫助產生多樣性的翻譯結果，但這邊指的原生 noise 是完全沒有幫助的髒東西，已經有許多研究證明這些 noise 對 training 和 testing 都有不好的影響:

1. `On the impact of various types of noise on neural machine translation`
2. `Synthetic and natural noise both break neural machine translation`
3. `A testbed for machine translation of noisy text`
4. `Towards robust neural machine translation`
5. `Assessing the tolerance of neural machine translation systems against speech recognition errors`

在不改善 noise 情況下想要增加效能的一種方法，是讓 model 在包含 synthetic noise 的情況下訓練。

1. `Training on synthetic noise improves robustness to natural noise in machine translation`
2. `Improving robustness of machine translation with synthetic noise`

而要在 training 就解決 noise 問題，有一種做法在 SMT 就已經在研究，稱為 `corpus filtering`，常與 `domain adaptation` 一起實作，但 SMT 的 `corpus filtering` 無法有效的在 NMT 上被實踐，所以開始有人針對 NMT 設計 `corpus filtering`:

1. Based on semantic analysis
   1. `Detecting cross-lingual semantic divergence for neural machine translation`
2. (Most effective in WMT18) Combination of `likelihood scores` from `neural translation models` and `neural language models` which trained on `clean data`
   1. `Findings of the WMT 2018 shared task on parallel corpus filtering`
   2. `Microsoft’s submission to the WMT2018 news translation task`
   3. `Dual conditional cross-entropy filtering of noisy parallel corpora`
   4. `The RWTH Aachen University filtering system for the WMT 2018 parallel corpus filtering task`

這些做法最終都是希望 sentence pairs 為 translation model 所翻譯出的最接近的結果，但有一些人反對這個說法，他們認為 'difficult' training samples 才對 NMT 模型有幫助 (samples with low translation probability)

這種 hard data filtering 的做法又稱為 `curriculum learning`: 

1. `Curriculum learning`
2. `Dynamic data selection for neural machine translation`
3. `Denoising neural machine translation training with trusted data and online data selection`
4. `Reinforcement learning based curriculum optimization for neural machine translation`
5. `Competence-based curriculum learning for neural machine translation`

# Domain Adaptation

`Domain adaptation` 是 MT 中非常重要的領域，目標是在非常大的 `out-of-domain corpus` 中挑選 (select) 或加權 (weight) 樣本，做法有好幾種:

1. Back-translation 透過將 `in-domain monolingual corpus` 翻譯回去，得到新的 `in-domain` 資料 
2. 聯合訓練 `in-domain` 和 `out-domain` sentences，中間透過 `domain-tags` 來幫助訓練
   1. `Domain control for neural machine translation`
   2. `Multi-domain neural machine translation`
   3. `Effective domain mixing for neural machine translation`
3. 簡單的 `in-domain` 和 `out-domain` 串聯，就能提升 NMT 的實力和一般化效果
   1. `Neural machine translation training in a multi-domain scenario`
4. 將分開訓練的 `in-domain` 及 `general-domain` 的模型透過 ensembling 合併
   1. `Fast domain adaptation for neural machine translation`
5. Constraining an NMT system to SMT lattices
   1. `Neural lattice search for domain adaptation in machine translation`

另一類的方法是先訓練一個 `general-domain model` 然後再利用 `in-domain corpus` 來繼續 fine-tune，這方法有兩個缺陷:

1. Catastrophic forgetting
   * 在 `in-domain` 的訓練很成功，但 `general-domain` 的表現卻大幅降低
2. Over-fitting
   * 因為 `in-domain` 的資料過少，導致模型變成 `over-fitting`

可以利用在 `fine-tuning` 階段手動設定**一些限制**來降低兩個缺陷的影響:

1. 凍結 sub-networks
   1. `Freezing subnetworks to analyze domain adaptation in neural machine translation`
2. 不更新所有 weights，而是只學習一些 hidden units 的 `scaling factors`
   1. `Learning hidden unit contributions for unsupervised speaker adaptation of neural network acoustic models`
   2. `Learning hidden unit contribution for adapting neural machine translation models`
3. 應用 regularizers 來讓 weights 保留在接近原本的權重
   1. 用 `knowledge distillation` 來 regularize `output distributions`
      1. `Regularized training objective for continued training for domain adaptation in neural machine translation`
      2. `Fine-tuning for neural machine translation with limited degradation across in-and out-of-domain data`
   2. 用 `L2 regularization` 和 `dropout` 來執行 `domain adaptation`
      1. `Regularization techniques for finetuning in neural machine translation`
   3. Elastic weight consolidation (EWC) 將 `importance of weights` 納入考量，像是 L2 的一般化做法
      2. `Overcoming catastrophic forgetting during domain adaptation of neural machine translation`
      3. `Domain adaptive inference for neural machine translation`

EWC does **not only reduce catastrophic forgetting** but even **yields gains on the general domain** when used for fine-tuning on a related domain.

# Low-resource NMT

雖然前面有提到 SMT 在 `low-resource` 的情況下通常都贏過 NMT，但近期也有非常多的研究旨在提升 NMT 於 `low-resource` 的表現:

1. `Monolingual data` with `back-translation`
2. Translation from `source` into `third resource-rich language`, then into `target`
   * `Triangular architecture for rare language translation`
3. Transfer learning 首先在 `resource-rich language` 上訓練一個 `parent model` (e.g., French-English)，再用該 model 來訓練 `low-resource language pairs` (e.g., Uzbek-English)
   * `Transfer learning for low-resource neural machine translation`
   * 這個方法需要依賴語言的相關性:
     * `Transfer learning across low-resource, related languages for neural machine translation`
     * `Addressing word-order divergence in multilingual neural machine translation for extremely low resource languages`
     * `An empirical study of language relatedness for transfer learning in neural machine translation`
4. Multilingual NMT system 可以良好適應 `low-resource language pairs`
   * `Rapid adaptation of neural machine translation to new languages`
5. Supervised the generation order of an insertion-based low-resource translation model with word alignments
   * `Neural machine translation for low-resource languages`

A series of NIST evaluation campaigns called `LoReHLT` focuses on `low-resource MT`, and recent `WMT editions` also contain `low-resource language pairs`.

* `Overview of the NIST 2016 LoReHLT evaluation`
* `Findings of the 2017 conference on machine translation (WMT17)`
* `Findings of the 2018 conference on machine translation (WMT18)`
* `Findings of the 2019 conference on machine translation (WMT19)`

# Unsupervised NMT

`Unsupervised NMT` 不使用任何 `cross-lingual data` 而是完全只用 `(unrelated) monolingual data` 來訓練，通常一開始會有一個 `unsupervised cross-lingual word embedding model`

* `Word translation without parallel data`
* `Learning bilingual word embeddings with (almost) no bilingual data`
* `Non-adversarial unsupervised word translation`

該 model 負責將 source 和 target 的 embedding 映射到 `joint embedding space`

* `Unsupervised machine translation using monolingual corpora only`
* `Unsupervised neural machine translation`

該 model 再透過遞迴的 back-translation 來更新並完善

* `Phrase-based & neural unsupervised machine translation`
* `Unsupervised neural machine translation with SMT as posterior regularization`

Back-translation 有一個替代方案為 `extract-edit scheme`，用編輯來取代合成 monolingual corpus

* `An alternative to back-translation for unsupervised neural machine translation`

近年來 unsupervised NMT 已經在 WMT 中受到關注

* `Findings of the 2018 conference on machine translation (WMT18)`
* `Findings of the 2019 conference on machine translation (WMT19)`
