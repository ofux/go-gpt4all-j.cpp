package gptj

// #cgo CFLAGS: -I./gpt4all-j/ggml/include/ggml/ -I./gpt4all-j/ggml/examples/
// #cgo CXXFLAGS: -I./gpt4all-j/ggml/include/ggml/ -I./gpt4all-j/ggml/examples/
// #cgo darwin LDFLAGS: -framework Accelerate
// #cgo darwin CXXFLAGS: -std=c++17
// #cgo LDFLAGS: -lgptj -lm -lstdc++
// #include <gptj.h>
import "C"
import (
	"fmt"
	"strings"
	"time"
	"unicode"
	"unsafe"
)

type GPTJ struct {
	state unsafe.Pointer
}

func New(model string) (*GPTJ, error) {
	state := C.gptj_allocate_state()
	modelPath := C.CString(model)
	result := C.gptj_bootstrap(modelPath, state)
	if result != 0 {
		return nil, fmt.Errorf("failed loading model")
	}

	return &GPTJ{state: state}, nil
}

func (l *GPTJ) Predict(responseStream chan<- string, text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)

	input := C.CString(text)
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}
	out := make([]byte, po.Tokens)

	params := C.gptj_allocate_params(input, C.int(po.Seed), C.int(po.Threads), C.int(po.Tokens), C.int(po.TopK),
		C.float(po.TopP), C.float(po.Temperature), C.int(po.Batch))

	stopStreaming := make(chan struct{})
	go func() {
		for {
			select {
			case <-stopStreaming:
				return
			default:
				time.Sleep(200 * time.Millisecond)
				str := C.GoString((*C.char)(unsafe.Pointer(&out[0])))
				str = sanitizeResponse(str, text)
				responseStream <- str
			}
		}
	}()

	ret := C.gptj_predict(params, l.state, (*C.char)(unsafe.Pointer(&out[0])))
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}
	res := C.GoString((*C.char)(unsafe.Pointer(&out[0])))

	res = sanitizeResponse(res, text)
	C.gptj_free_params(params)

	stopStreaming <- struct{}{}
	return res, nil
}

func (l *GPTJ) Free() {
	C.gptj_free_model(l.state)
}

func sanitizeResponse(res string, text string) string {
	res = trimLeadingSpaces(res)
	res = strings.TrimPrefix(res, trimLeadingSpaces(text))
	res = strings.TrimPrefix(res, "\n")
	res = strings.TrimSuffix(res, "<|endoftext|>")
	return res
}

func trimLeadingSpaces(s string) string {
	for i, c := range s {
		if !unicode.IsSpace(c) {
			return s[i:]
		}
	}
	return ""
}
