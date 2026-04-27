# EFFACE: Top-1 Compression Suffices for Federated Unlearning with the Help of Adaptive Error Feedback

[![Venue: ICASSP 2026](https://img.shields.io/badge/Venue-ICASSP%202026-blue)](https://2026.ieeeicassp.org/event/about-conference/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/RadiumStar/EFFACE?tab=MIT-1-ov-file)
[![GitHub top language](https://img.shields.io/github/languages/top/RadiumStar/EFFACE)](https://github.com/RadiumStar/EFFACE)
[![GitHub repo size](https://img.shields.io/github/repo-size/RadiumStar/EFFACE)](https://github.com/RadiumStar/EFFACE)
[![GitHub stars](https://img.shields.io/github/stars/RadiumStar/EFFACE)](https://github.com/RadiumStar/EFFACE)

This is the official code repository for the ICASSP 2026 paper [Top-1 Compression Suffices for Federated Unlearning with the Help of Adaptive Error Feedback](https://ieeexplore.ieee.org/abstract/document/11462604)

## 🐱‍💻 Code 
See [EFFACE code overview](https://github.com/RadiumStar/EFFACE/blob/main/code/README.md) for more details. 

## 🔍 Clarification on the Selection Mechanism
The selection mechanism presented in the ICASSP paper was optimized for empirical performance. However, our subsequent theoretical analysis indicates that incorporating a **correction term** is necessary to strictly guarantee convergence bounds.

### Formulation Comparison

| Version | Formula |
| :--- | :--- |
| **Original** (Used in Paper Experiments) | ![](https://github.com/RadiumStar/EFFACE/blob/main/previous_implement_selection.png) |
| **Corrected** (Current Repo Implementation) | ![](https://github.com/RadiumStar/EFFACE/blob/main/current_implement_selection.png) |

> **Implementation Update**: We have updated the corresponding code to implement the theoretically corrected version. Our empirical verification confirms that this update yields performance comparable to the original implementation, ensuring both theoretical rigor and practical effectiveness.

## 🚀 Upcoming Extended Version
We are preparing a comprehensive extension of this work that will include:
1. **Convergence Analysis**: Detailed analysis of the convergence order for EFFACE, including the selection of the error compensation strength coefficient $\eta$
2. **Federated Unlearning Bounds**: Specific theoretical guarantees within the federated unlearning context.

*This extended manuscript is currently in preparation and will be made available soon. We appreciate your interest and patience.*

## 📖 Citation
If you find this work useful, please consider citing:
```bibtex
@inproceedings{xiao2026efface,
  title={Top-1 Compression Suffices for Federated Unlearning with the Help of Adaptive Error Feedback},
  author={Xiao, Boxu and Liu, Sijia and Ling, Qing},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2171--2175},
  year={2026},
  organization={IEEE}
}
