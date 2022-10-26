# Workable MoveNet on PyTorch

Movenet is Google's next generation fast pose detection model. However, there seems not PyTorch version.

I ported original tensorflow weights to pytorch, using CenterNet code base to inference. The result is fully right.

Hopefully, you can even train it on pytorch if you have time.

If you are interested on Movenet, open an issue and I will guide you have to contribute.


## Demo

Run:

```
python demo.py single_pose --dataset active --arch movenet --demo data/input_image.jpeg --load_model ./weights/movenet.pth --K 1 --gpus -1 --debug 2
```


![](https://raw.githubusercontent.com/jinfagang/public_images/master/20221017164046.png)

Runing on WNN:

![](https://raw.githubusercontent.com/jinfagang/public_images/master/Kapture%202022-10-25%20at%2022.47.27.gif)