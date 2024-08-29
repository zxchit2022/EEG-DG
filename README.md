# EEG-DG：A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification

This is the offical repository of the paper "EEG-DG：A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification" in IEEE Journal of Biomedical and Health Informatics [https://ieeexplore.ieee.org/abstract/document/10076833](https://ieeexplore.ieee.org/document/10609514).

In this work,
1) We consider a more practical and challenging scenario: domain-generalized motor imagery EEG classification where the target domain EEG data cannot be accessed during the training process.
2) To address the variations in EEG signals across sessions and subjects, we propose a multi-source domain generalization framework (EEG-DG) that learns domain-invariant
features with strong representation by optimizing both P(X) and P(Y |X) towards minimizing the discrepancy across a variety of source domains.
3) Systematic experiments on a simulated dataset and three motor imagery EEG datasets demonstrate that our proposed EEG-DG can deliver a competitive performance compared to other methods. Particularly, the proposed EEG-DG has the potential to achieve performance comparable to or even outperform domain adaptation methods that can access the test data during training.

All experiments are conducted with Windows 11 on an Intel Core i7 CPU and an NVIDIA RTX 4080 16GB GPU. Our proposed EEG-DG framework is implemented on Python 3.8 with the PyTorch package.

# Citing
If you find our work is useful for your research, please consider citing it:

@article{zhong2024eeg,
  title={EEG-DG: A Multi-Source Domain Generalization Framework for Motor Imagery EEG Classification},
  author={Zhong, Xiao-Cong and Wang, Qisong and Liu, Dan and Chen, Zhihuang and Liao, Jing-Xiao and Sun, Jinwei and Zhang, Yudong and Fan, Feng-Lei},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  publisher={IEEE}
}
