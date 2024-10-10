<p align="center">
  <h1 align="center">Learning Commonality, Divergence and Variety for Unsupervised Visible-Infrared Person Re-identification</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Go9q2jsAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Jiangming Shi*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?hl=zh-CN&pli=1&user=H1rqfM4AAAAJ" rel="external nofollow noopener" target="_blank"><strong>Xiangbo Yin*</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=a-I8c8EAAAAJ&hl=zh-CN&oi=sra" target="_blank"><strong>Yachao Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=CXZciFAAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Zhizhong Zhang</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=RN1QMPgAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Yuan Xie† </strong></a>    
    ·
    <a href="https://scholar.google.com/citations?user=idiP90sAAAAJ&hl=zh-CN&oi=sra" rel="external nofollow noopener" target="_blank"><strong>Yanyun Qu† </strong></a>       
  </p>
<p align="center">
  <a href="https://arxiv.org/pdf/2402.19026" rel="external nofollow noopener" target="_blank">Paper Link</a>

![PCLHD](figs/framework.png)
This is an official code implementation of "Learning Commonality, Divergence and Variety for Unsupervised Visible-Infrared Person Re-identification", Which is accepted by NeurIPS 2024.


## Requirements
- python 3.8.13
- torch 1.8.0
- torchvision 0.9.0
- scikit-learn 1.2.2
  

## Dataset Preprocessing
```shell
python prepare_sysu.py   # for SYSU-MM01
python prepare_regdb.py  # for RegDB
```
You need to change the dataset path to your own path in the `prepare_sysu.py` and `prepare_regdb.py`.


## Training
```shell
sh run_train_sysu.sh     # for SYSU-MM01
sh run_train_regdb.sh    # for RegDB
```

## Testing
```shell
sh test_sysu.sh          # for SYSU-MM01
sh test_regdb.sh         # for RegDB
```

## Citation
If our work is helpful for your research, please consider citing:
```
@article{shi2024progressive,
  title={Progressive Contrastive Learning with Multi-Prototype for Unsupervised Visible-Infrared Person Re-identification},
  author={Shi, Jiangming and Yin, Xiangbo and Wang, Yaoxing and Liu, Xiaofeng and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2402.19026},
  year={2024}
}

@article{yin2024robust,
  title={Robust Pseudo-label Learning with Neighbor Relation for Unsupervised Visible-Infrared Person Re-Identification},
  author={Yin, Xiangbo and Shi, Jiangming and Zhang, Yachao and Lu, Yang and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2405.05613},
  year={2024}
}

@article{shi2024multi,
  title={Multi-Memory Matching for Unsupervised Visible-Infrared Person Re-Identification},
  author={Shi, Jiangming and Yin, Xiangbo and Chen, Yeyun and Zhang, Yachao and Zhang, Zhizhong and Xie, Yuan and Qu, Yanyun},
  journal={arXiv preprint arXiv:2401.06825},
  year={2024}
}

@inproceedings{shi2023dpis,
  title={Dual pseudo-labels interactive self-training for semi-supervised visible-infrared person re-identification},
  author={Shi, Jiangming and Zhang, Yachao and Yin, Xiangbo and Xie, Yuan and Zhang, Zhizhong and Fan, Jianping and Shi, Zhongchao and Qu, Yanyun},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11218--11228},
  year={2023}
```

## Contact
xiangboyin@stu.xmu.edu.cn; jiangming.shi@outlook.com.
