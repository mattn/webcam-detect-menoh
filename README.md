# webcam-detect-menoh

Example App detecting objects from video capture using menoh that is DNN inference library.

## Usage

```
Usage of webcam-detect-menoh:
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
