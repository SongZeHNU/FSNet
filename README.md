# FSNet: Focus Scanning Network for Camouflaged Object Detection
> **Authors:** 
> [*Ze Song*](https://scholar.google.com/citations?user=uatSii8AAAAJ&hl=zh-CN&oi=sra),
> [*Xudong Kang*](https://scholar.google.com/citations?user=5XOeLZYAAAAJ&hl=en),
> [*Xiaohui Wei*](https://scholar.google.co.il/citations?user=Uq50h3gAAAAJ&hl=zh-CN),
> Haibo Liu,
> [*Renwei Dian*](https://scholar.google.com/citations?user=EoTrH5UAAAAJ&hl=en),
> and [*Shutao Li*](https://scholar.google.com/citations?user=PlBq8n8AAAAJ&hl=en).


Code implementation of "_**FSNet: Focus Scanning Network for Camouflaged Object Detection**_".  IEEE TIP 2023.[Paper](https://ieeexplore.ieee.org/document/10103836/)

## Prerequisites
Install Prerequisites with the following command:
```bash
conda create -n FSNet python = 3.7
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```
## Usage
### 1. Download pre-trained Swin transformer model
Please download model from the official websites: 
* [github](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth) or [baidu](https://pan.baidu.com/share/init?surl=vwJxnJcVqcLZAw9HaqiR6g) with the fetch code: swin.
* move it into ``` ./pretrained_ckpt/ ```

### 2. Prepare data

We use data from publicly available datasets:
+ downloading testing dataset and move it into `./Dataset/TestDataset/`, 
    which can be found in [Google Drive](https://drive.google.com/file/d/1SLRB5Wg1Hdy7CQ74s3mTQ3ChhjFRSFdZ/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1F3tVEWYzKYp5NBv3cjiaAg) (extraction code: fapn). 

+ downloading training/validation dataset and move it into `./Dataset/TrainValDataset/`, 
    which can be found in [Google Drive](https://drive.google.com/file/d/1Kifp7I0n9dlWKXXNIbN7kgyokoRY4Yz7/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1uyQz0b_r_5yCee0orSw7EA) (extraction code: fapn). 

### 3. Train

To train FSNet with costumed path:

```bash
python MyTrain_Val.py --save_path './snapshot/FSNet/'
```
### 4. Test

To test with trained model:

```bash
python MyTesting.py --pth_path './snapshot/FSNet/Net_epoch_best.pth'
```

downloading our weights and move it into `./snapshot/FSNet/`, 
    which can be found from [Google Drive](https://drive.google.com/file/d/1Bgi8MThe1eEE9fYyaHuLHacO1Cs_e9Ta/view?usp=share_link).
    
 You can also download prediction maps from [Google Drive](https://drive.google.com/file/d/1kT9l9NrwWCffP4EQ7ITBd834389xY8iV/view?usp=share_link).

### 4. Evaluation 

We use public one-key evaluation, which is written in MATLAB code ([link](https://drive.google.com/file/d/1_h4_CjD5GKEf7B1MRuzye97H0MXf2GE9/view?usp=sharing)). 
Please follow this the instructions in `./eval/main.m` and just run it to generate the evaluation results in `./res/`.


## Citation

Please cite our paper if you find the work useful, thanks!

	@article{song2023fsnet,
	   title={FSNet: Focus Scanning Network for Camouflaged Object Detection},
  	   author={Song, Ze and Kang, Xudong and Wei, Xiaohui and Liu, Haibo and Dian, Renwei and Li, Shutao},
  	   journal={IEEE Transactions on Image Processing},
  	   year={2023},
  	   publisher={IEEE}
	}
	

**[⬆ back to top](#1-preface)**
