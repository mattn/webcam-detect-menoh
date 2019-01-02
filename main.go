package main

import (
	"bufio"
	"bytes"
	"context"
	"flag"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	"github.com/pfnet-research/go-menoh"

	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"github.com/faiface/pixel/text"
	"gocv.io/x/gocv"
	"golang.org/x/image/colornames"
	"golang.org/x/image/font/basicfont"
)

const (
	batch   = 1
	channel = 3
	width   = 224
	height  = 224

	conv1_1InName  = "Input_0"
	fc6OutName     = "Gemm_0"
	softmaxOutName = "Softmax_0"
)

var (
	onnxModelPath   = flag.String("model", "data/vgg16.onnx", "ONNX model path")
	synsetWordsPath = flag.String("synset-words", "data/synset_words.txt", "synset words file path")
)

type result struct {
	classes []float32
	img     image.Image
}

func encodeFloats(img image.Image, channel int, bgrMean []float32) []float32 {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	floats := make([]float32, channel*h*w)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			floats[0*(w*h)+y*w+x] = float32(r/257) - bgrMean[2]
			floats[1*(w*h)+y*w+x] = float32(g/257) - bgrMean[1]
			floats[2*(w*h)+y*w+x] = float32(b/257) - bgrMean[0]
		}
	}
	return floats
}

func loadCategoryList(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return []string{}, err
	}
	defer file.Close()

	categories := []string{}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		categories = append(categories, strings.SplitN(scanner.Text(), " ", 2)[1])
	}
	if err := scanner.Err(); err != nil {
		return []string{}, err
	}
	return categories, nil
}

func extractTopKIndexList(values []float32, k int) []int {
	type pair struct {
		index int
		value float32
	}
	pairs := make([]pair, len(values))
	for i, f := range values {
		pairs[i] = pair{
			index: i,
			value: f,
		}
	}
	sort.SliceStable(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})
	topKIndices := make([]int, k)
	for i := 0; i < k; i++ {
		topKIndices[i] = pairs[i].index
	}
	return topKIndices
}

func capture(wg *sync.WaitGroup, cam *gocv.VideoCapture, frames chan []byte, ctx context.Context) {
	defer wg.Done()

	frame := gocv.NewMat()
	defer frame.Close()

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		if ok := cam.Read(&frame); !ok {
			log.Fatal("failed reading cam")
		}

		// Encode Mat as a bmp (uncompressed)
		buf, err := gocv.IMEncode(".bmp", frame)
		if err != nil {
			log.Fatalf("Error encoding frame: %v", err)
		}

		// Push the frame to the channel
		frames <- buf
	}
}

func run() {
	// Setup Pixel window
	cfg := pixelgl.WindowConfig{
		Title:  "Thinger",
		Bounds: pixel.R(0, 0, 500, 500),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		log.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup
	wg.Add(2)

	cam, err := gocv.OpenVideoCapture(0)
	if err != nil {
		log.Fatal("failed reading cam", err)
	}
	defer cam.Close()

	// Start up the background capture
	framesChan := make(chan []byte, 1)
	resultsChan := make(chan result, 1)
	go capture(&wg, cam, framesChan, ctx)
	go detect(&wg, resultsChan, framesChan, ctx)

	// Setup Pixel requirements for drawing boxes and labels
	mat := pixel.IM
	mat = mat.Moved(win.Bounds().Center())

	atlas := text.NewAtlas(basicfont.Face7x13, text.ASCII)
	imd := imdraw.New(nil)

	// Some local vars to calculate frame rate
	var (
		frames = 0
		second = time.Tick(time.Second)
	)

	categories, err := loadCategoryList(*synsetWordsPath)
	if err != nil {
		log.Fatal(err)
	}

	for !win.Closed() {
		// Run inference if we have a new frame to read
		result := <-resultsChan

		topKIndices := extractTopKIndexList(result.classes, 5)
		img := result.img

		pic := pixel.PictureDataFromImage(img)
		bounds := pic.Bounds()
		sprite := pixel.NewSprite(pic, bounds)

		imd.Clear()
		win.Clear(colornames.Black)
		sprite.Draw(win, mat)

		for i, idx := range topKIndices {
			s := fmt.Sprintf("%d %.5f %s\n", idx, result.classes[idx], categories[idx])
			txt := text.New(pixel.V(10.0, 470.0-float64(30*i)), atlas)
			txt.Color = color.White
			txt.WriteString(s)
			txt.Draw(win, pixel.IM.Scaled(txt.Orig, 2))
		}

		imd.Draw(win)
		win.Update()

		// calculate frame rate
		frames++
		select {
		case <-second:
			win.SetTitle(fmt.Sprintf("%s | FPS: %d", cfg.Title, frames))
			frames = 0
		default:
		}
	}

	cancel()
	<-framesChan
	<-resultsChan
	wg.Wait()
	close(framesChan)
	close(resultsChan)
}

func detect(wg *sync.WaitGroup, results chan<- result, frames chan []byte, ctx context.Context) {
	defer wg.Done()

	bgrMean := []float32{103.939, 116.779, 123.68}

	// build model runner
	runner, err := menoh.NewRunner(menoh.Config{
		ONNXModelPath: *onnxModelPath,
		Backend:       menoh.TypeMKLDNN,
		BackendConfig: "",
		Inputs: []menoh.InputConfig{
			{
				Name:  conv1_1InName,
				Dtype: menoh.TypeFloat,
				Dims:  []int32{batch, channel, height, width},
			},
		},
		Outputs: []menoh.OutputConfig{
			{
				Name:         fc6OutName,
				Dtype:        menoh.TypeFloat,
				FromInternal: true,
			},
			{
				Name:         softmaxOutName,
				Dtype:        menoh.TypeFloat,
				FromInternal: false,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer runner.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case frame := <-frames:
			img, _, err := image.Decode(bytes.NewReader(frame))
			if err != nil {
				continue
			}

			resizedImg := imaging.Resize(img, width, height, imaging.Linear)
			resizedImgTensor := &menoh.FloatTensor{
				Dims:  []int32{batch, channel, height, width},
				Array: encodeFloats(resizedImg, channel, bgrMean),
			}
			err = runner.RunWithTensor(conv1_1InName, resizedImgTensor)
			if err != nil {
				log.Println(err.Error())
				continue
			}
			softmaxOutTensor, err := runner.GetOutput(softmaxOutName)
			if err != nil {
				log.Println(err.Error())
				continue
			}
			softmaxOutData, err := softmaxOutTensor.FloatArray()
			if err != nil {
				log.Println(err.Error())
				continue
			}
			results <- result{
				classes: softmaxOutData,
				img:     img,
			}
		default:
		}
	}
}

func main() {
	flag.Parse()
	pixelgl.Run(run)
}
