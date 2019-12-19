# bidirectional_mocogan

pytroch impl. of MoCoGAN and BiGAN(ALI).

### Usage

1.make original data for training the model
```python
# make custom movingMNIST dataset
$ python3 make_data.py
```

2. train the model
```python
# train GAN model
$ python3 -u train.py | tee stdout.log
```
