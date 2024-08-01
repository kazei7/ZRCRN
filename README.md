# Fast, Zero Reference Low Light Image Enhancement with Camera Response Model

**The implementation of ZRCRN is for non-commercial use only.**

## Folder structure
```
├─input
├─output
├─snapshots
├─train_data
│  └─train
└─valid_data
   ├─Low
   └─Normal
```

## Requirements
```
pip install -r requirements.txt
```

## Test:
```
python test.py 
```
The script will process the images in the "input" folder and make a new folder "output". 
You can find the enhanced images in the "output" folder.

## Train: 
1. Download the training data 
   <a href="https://drive.google.com/file/d/1dRBvjzzW3PbQIQeN6bXBl9pKDuyxsm1D/view?usp=sharing">google drive</a> or <a href="https://pan.baidu.com/s/1QrE2bXcHvCixdPm2VeJpZw?pwd=a5ms">baidu cloud [password: a5ms]</a>
2. Unzip and put the downloaded "train" folder to "train_data" folder
3. Run the following script. New checkpoints will be saved in "snapshots" folder.
```
python train.py 
```


## References
1. https://github.com/Li-Chongyi/Zero-DCE
2. https://github.com/RenYurui/LECARM