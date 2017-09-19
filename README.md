# CaptchaRecognition
End-to-end variable length Captcha recognition using  CNN+RNN+Attention.  端到端的不定长验证码识别

目前encoder端可以选择使用CNN+RNN或CNN；decoder端有两种attention方式+不使用attention。

## TODO
- RNN + CTC loss

## Usage
把字体文件放入fonts文件夹，并修改GenCaptcha.py中第173行的字体文件名。

运行 python GenCaptcha.py ，在data/下生成数据集captcha.npz和captcha.vocab_dict。

（GenCaptcha.py中还提供了生成tfrecord文件的函数。）

运行 python main.py训练。
