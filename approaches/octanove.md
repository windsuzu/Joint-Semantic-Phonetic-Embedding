# Octanove Labs’ Japanese-Chinese Open Domain Translation System

## Abstract

這篇論文使用了兩種方法來提升 open domain 下的 `Japanese-Chinese translation system`:

1. Parallel corpus filtering
2. Back-translation

第一種方法使用一些 `heuristic rules` 和 `learned classifiers`，能將 parallel data 削減 70% 至 90%，而且不降低翻譯表現；第二種方法利用 `back-translation` 來產生人造資料，可以增加 17% 至 27% 的表現。

## Introduction

