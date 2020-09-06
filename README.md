# Faster-RCNN with DPP

This code is part of the paper: [Determinantal Point Process as an alternative to NMS](https://arxiv.org/abs/2008.11451) published at BMVC 2020.

The code is forked from and based on [A Faster Pytorch Implementation of Faster R-CNN](https://github.com/jwyang/faster-rcnn.pytorch) and as such can be used in a similar fashion.

## Prerequisites
- Python 3.7.5
- PyTorch 1.5.0
- Boost 1.71.0
- Eigen 3.3.7
- CUDA 10.0

## Installation    
- Clone the repository and checkout the relevant branch.
```bash
git clone https://github.com/samiksome/faster-rcnn.pytorch
cd faster-rcnn.pytorch
git checkout dpp
```

- Install requirements
```bash
pip install -r requirements.txt
```

- Build
```bash
cd lib
python setup.py build develop
cd model
make
```

- Replace pycocotools
```bash
cd ../../..
git clone "https://github.com/cocodataset/cocoapi"
cd cocoapi/PythonAPI

make all

cd ../../faster-rcnn.pytorch/lib
rm -rf pycocotools
cp -r ../../cocoapi/PythonAPI/pycocotools ./
cd ..
```

## Datasets and Pretrained models
- Get datasets (PASCAL trainval and COCO train datasets can be skipped if training is not needed)
```bash
mkdir data
cd data

wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
wget "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar"
tar -xvf VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
tar -xvf VOCdevkit_08-Jun-2007.tar
mv VOCdevkit VOCdevkit2007

mkdir coco
cd coco

wget "http://images.cocodataset.org/zips/train2014.zip"
wget "http://images.cocodataset.org/zips/val2014.zip"
wget "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
wget "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip"
wget "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip"
unzip train2014.zip
unzip val2014.zip
unzip annotations_trainval2014.zip
unzip instances_minival2014.json.zip
unzip instances_valminusminival2014.json.zip
mkdir images
mv train2014 images/
mv val2014 images/
mv instances_minival2014.json annotations/
mv instances_valminusminival2014.json annotations/

cd ../..
```

- Download trained models (ours)
```bash
wget "https://www.dropbox.com/s/dnxsdhkhcj9jvn0/models.zip"
unzip models.zip
```

## Running the code
- Test on PASCAL VOC (with our trained models) using NMS
```bash
python test_net.py --dataset pascal_voc \
                   --net vgg16 \
                   --checksession 1 --checkepoch 6 --checkpoint 10021 \
                   --cuda
```

- Test on PASCAL VOC (with our trained models) using DPP
```bash
python test_net.py --dataset pascal_voc \
                   --net vgg16 \
                   --checksession 1 --checkepoch 6 --checkpoint 10021 \
                   --use_dpp --dpp_alpha 5 \
                   --cuda
```

Replace `--dataset pascal_voc` with `--dataset coco` and `--checkpoint 10021` with `--checkpoint 58632` to test on COCO dataset.

By default the maximum number of windows selected (`k`) is `300`. To use a different `k` change `line 196` in `faster-rcnn.pytorch/lib/model/utils/config.py` to required value.
```python
__C.TEST.RPN_POST_NMS_TOP_N = 300
```

## Citation
Please cite the following paper if you use this code:
```
@misc{some2020determinantal,
    title={Determinantal Point Process as an alternative to NMS},
    author={Samik Some and Mithun Das Gupta and Vinay P. Namboodiri},
    year={2020},
    eprint={2008.11451},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
