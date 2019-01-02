# webcam-detect-monoh

Object detection using menoh that is DNN inference library.

## Usage

```
Usage of webcam-detect-menoh:
  -input-image string
    	input image path (default "data/Light_sussex_hen.jpg")
  -model string
    	ONNX model path (default "data/vgg16.onnx")
  -synset-words string
    	synset words file path (default "data/synset_words.txt")
```

## Requirements

* menoh
* Go

## Installation

```
$ go get github.com/mattn/webcam-detect-menoh
```

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)
