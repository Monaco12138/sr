# 易复用的SR模型框架
本工作实现了一个易于复用的SR模型框架，目标是在测试使用新的SR模型时只需做最少量的修改。
下面以EDSR模型为例阐述本框架的运行逻辑。
***
## models
里面包含了相应的SR模型，使用新模型时将相应的模型文件添加进入。  
___需注意的部分：___ 
1. 需要在返回SR模型类声明前添加一个修饰器。

    ```python
    @register('edsr')
    class EDSR(nn.Module):
        def __init__(self, args):
            pass
    ```

2. 需要在models.\_\_init\_\_.py文件中import对应的模型文件

***
## datasets
包含了数据集的处理工作。  
___需注意的部分：___
1. image_folder.py 包含了两个读取图像的类，通常我们使用PairedImageFolders类
2. wrappers.py 包含了对图像处理的一些操作，关于SR的放大倍数是隐式定义的，取决于你给出的LR数据集和HR数据集。
3. 如果添加了自定义的数据处理函数，需在声明前添加修饰器
    ```python
    @register('sr_paired')
    class SRImplicitPaired(Dataset):
        def __init__(self, args):
            pass
    ```
4. 如果添加了新的文件，需在datasets.\_\_init\_\_.py文件中import
***
## configs
相应的训练参数设置  
___需注意的部分：___
1. 以训练集为例，每一个name对应的名字都要在register('name')中存在，即可调用相应的类，args后的参数为对应类的初始化参数
    ```python
    train_dataset:
        dataset:
            name: paired-image-folders
            args:
                root_path_1: DIV2K/DIV2K_train_LR_bicubic/X2_1-800
                root_path_2: DIV2K/DIV2K_train_HR1-800
                repeat: 20
                cache: in_memory
        wrapper:
            name: sr-implicit-paired
            args:
                inp_size: 48
                augment: true
        batch_size: 32
    ```
2. 相关参数含义:

    + __dataset.args__
        + _root_path_1_: 训练用低清图片(lr)的路径
        + _root_path_2_: 训练用高清图片(hr)的路径 
        + _repeat_: 表示1个epoch内重复遍历数据集次数，如果数据集图片较少可以提高 _repeat_
        + _cache_: 
            + in_memory：表示将数据集存储在内存中，如果内存够大可以用这个，会加快训练速度
            + none: 在训练时才从磁盘中读取数据，训练速度较慢
    + __wrapper.args__
        + _inp_size_: 表示训练时输入图片的大小，一般SR网络训练时会对输入图片进行一定程度的裁剪，常见的裁剪大小是(48,48)
        + _augment_: 表示是否对训练图片加入旋转，上下颠倒，左右颠倒等变化，适当的增加训练图片的变化可以提高模型泛化行
***
## train&test
### train: 
1. 训练时选择好对应的config文件，和保存的文件夹位置即可。
2. 默认的保存文件树如下：  
    * save
        * \_train_edsr\_
            * config.yaml
            * epoch-100.pth
            * epoch-xxx.pth
            * epoch-last.pth
            * epoch-best.pth
            * log.txt
            * tensorboard
        * \_train_rdn\_
            * ...
3. 若模型forward参数有多个，需在tran.py的train函数中添加其余输入参数
4. loss默认为L1, 可在train.py的train函数中设置。
```python
def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    ...
    for batch in tqdm(train_loader, leave=False, desc='train'):        
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = (batch['inp'] - inp_sub) / inp_div            
        inp_kwargs = { 'x': inp } #可增加model.forward()参数                      
        pred = model( **inp_kwargs )
        gt = (batch['gt'] - gt_sub) / gt_div
        loss = loss_fn(pred, gt)
        ...

```
### test: 
1. 测试时选择好对应的config文件和保存的模型位置。
2. 测试结果只在终端print。
***

