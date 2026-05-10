package handlers

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sync/atomic"
	"time"
)

// RequestLogger writes per-request JSON payloads to timestamped files.
// Created only when debug logging is enabled; nil otherwise.
type RequestLogger struct {
	dir     string
	counter atomic.Int64
}

// loggedRequest is the metadata wrapper written to each file.
type loggedRequest struct {
	Timestamp string `json:"timestamp"`
	Model     string `json:"model"`
	Streaming bool   `json:"streaming"`
	BodyJSON  any    `json:"body,omitempty"`
	BodyText  string `json:"body_text,omitempty"`
}

// NewRequestLogger creates the log directory at $HOME/.config/oc-go-cc/logs/<ts>/
// and returns a RequestLogger. Returns nil if the directory cannot be created.
func NewRequestLogger() *RequestLogger {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil
	}

	ts := time.Now().Format("2006-01-02-15-04")
	dir := filepath.Join(home, ".config", "oc-go-cc", "logs", ts)

	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil
	}

	return &RequestLogger{dir: dir}
}

// Log writes a single JSON file for one request/response payload.
// counter is the per-request incrementing ID, isStream controls the _stream suffix,
// model is the upstream model name, and data is the JSON payload.
// suffix distinguishes multiple payloads from the same request (e.g. "req", "resp", "transformed").
func (l *RequestLogger) Log(counter int, isStream bool, model string, data []byte, suffix string) {
	if l == nil {
		return
	}

	var filename string
	if isStream {
		filename = fmt.Sprintf("%d_stream_%s.json", counter, suffix)
	} else {
		filename = fmt.Sprintf("%d_%s.json", counter, suffix)
	}

	var bodyJSON any
	var bodyText string
	if json.Valid(data) {
		bodyJSON = json.RawMessage(data)
	} else {
		bodyText = string(data)
	}

	entry := loggedRequest{
		Timestamp: time.Now().Format(time.RFC3339Nano),
		Model:     model,
		Streaming: isStream,
		BodyJSON:  bodyJSON,
		BodyText:  bodyText,
	}

	out, err := json.MarshalIndent(entry, "", "  ")
	if err != nil {
		slog.Error("requestlog marshal failed", "filename", filename, "error", err, "data_len", len(data))
		return
	}
	slog.Debug("requestlog writing file", "filename", filename, "size", len(out), "body_len", len(data))
	if err := os.WriteFile(filepath.Join(l.dir, filename), out, 0644); err != nil {
		slog.Error("requestlog write failed", "filename", filename, "error", err)
	}
}
