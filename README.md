# Vision KAN
We try to explore the application of KAN in visual tasks


we can use "KanMLPMixer","KanPermutator" to get a  classification model:
```bash
python train_cifar10.py
```

We circumvent the problem that the classification model can only support fixed-size inputs by applying kan to the channel dimension or using windows in the spatial dimension.

we can use "(channel)kanSSR","SwinPermutatorKan","SwinConvKan" to get a Hyperspectral Image Restoration model / spectral super resolution model / semantic segmentation model:
```bash
cd ./fastkan/HyperSpectralmodel
python kan_fit_test.py.py
```
super spectral image task's training code is coming soon!

# To do

semantic segmentation

# thansk:
[KAN](https://github.com/KindXiaoming/pykan),
[fast kan](https://github.com/ZiyaoLi/fast-kan),
[MLP Mixer](https://github.com/lucidrains/mlp-mixer-pytorch)
