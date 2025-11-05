This codebase has developed into a new project that is well-maintained and includes more SOTA methods. Please refer to [RNKS-for-CIL](https://www.sciencedirect.com/science/article/pii/S0031320324002577) for more information.
# Implementation of continual learning methods
This repository implements some continual / incremental / lifelong learning methods by PyTorch.

Especially the methods based on **memory replay**.

- [x] iCaRL: Incremental Classifier and Representation Learning. [[paper](https://arxiv.org/abs/1611.07725)]
- [x] End2End: End-to-End Incremental Learning. [[paper](https://arxiv.org/abs/1807.09536)]
- [x] DR: Lifelong Learning via Progressive Distillation and Retrospection. [[paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Saihui_Hou_Progressive_Lifelong_Learning_ECCV_2018_paper.html)]
- [x] UCIR: Learning a Unified Classifier Incrementally via Rebalancing. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html)]
- [x] BiC: Large Scale Incremental Learning. [[paper](https://arxiv.org/abs/1905.13260)]
- [ ] LwM: Learning without Memorizing. [[paper](https://arxiv.org/abs/1811.08051)]

## Dependencies
1. torch 1.4.0
2. torchvision 0.5.0
3. tqdm
4. numpy
5. scipy

## Usage
1. Edit the *config.json* file.
2. Run:
```bash
python main.py
```


## References
https://github.com/arthurdouillard/incremental_learning.pytorch

## Citation
@article{song2024rebalancing,
  title={Rebalancing network with knowledge stability for class incremental learning},
  author={Song, Jialun and Chen, Jian and Du, Lan},
  journal={Pattern Recognition},
  volume={153},
  pages={110506},
  year={2024},
  publisher={Elsevier}
}
