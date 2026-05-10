// Package handlers contains HTTP request handlers for API endpoints.
package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"oc-go-cc/internal/client"
	"oc-go-cc/internal/config"
	"oc-go-cc/internal/metrics"
	"oc-go-cc/internal/middleware"
	"oc-go-cc/internal/router"
	"oc-go-cc/internal/token"
	"oc-go-cc/internal/transformer"
	"oc-go-cc/pkg/types"
)

// streamResponseCache caches Anthropic-format responses from streaming requests
// so that subsequent non-streaming duplicates can be served from cache.
type streamResponseCache struct {
	mu       sync.Mutex
	entries  map[string]*cacheEntry
}

type cacheEntry struct {
	response []byte
	done     chan struct{}
}

func newStreamResponseCache() *streamResponseCache {
	return &streamResponseCache{
		entries: make(map[string]*cacheEntry),
	}
}

// getOrCreate returns an existing cache entry or creates a new one.
// If the entry already exists and is complete, returns (response, true, true).
// If the entry is still pending, waits for it and returns (response, true, false).
// If no entry exists, creates one and returns (nil, false, false).
func (c *streamResponseCache) getOrCreate(key string) ([]byte, bool) {
	c.mu.Lock()
	entry, exists := c.entries[key]
	if exists {
		c.mu.Unlock()
		<-entry.done
		return entry.response, true
	}
	entry = &cacheEntry{done: make(chan struct{})}
	c.entries[key] = entry
	c.mu.Unlock()
	return nil, false
}

// store stores a response and signals waiters. The entry is removed after ttl.
func (c *streamResponseCache) store(key string, response []byte, ttl time.Duration) {
	c.mu.Lock()
	entry, exists := c.entries[key]
	if !exists {
		c.mu.Unlock()
		return
	}
	entry.response = response
	close(entry.done)

	// Schedule cleanup after ttl
	go func() {
		time.Sleep(ttl)
		c.mu.Lock()
		delete(c.entries, key)
		c.mu.Unlock()
	}()
	c.mu.Unlock()
}

// MessagesHandler handles /v1/messages requests.
type MessagesHandler struct {
	client              *client.OpenCodeClient
	modelRouter         *router.ModelRouter
	fallbackHandler     *router.FallbackHandler
	requestTransformer  *transformer.RequestTransformer
	responseTransformer *transformer.ResponseTransformer
	streamHandler       *transformer.StreamHandler
	tokenCounter        *token.Counter
	logger              *slog.Logger
	rateLimiter         *middleware.RateLimiter
	requestDedup        *middleware.RequestDeduplicator
	requestIDGen        *middleware.RequestIDGenerator
	metrics             *metrics.Metrics
	autoRoute           bool
	requestLogger       *RequestLogger
	requestCounter      atomic.Int64
	responseCache       *streamResponseCache
	timeout             time.Duration
}

// responseWriter wraps http.ResponseWriter to track if headers were written.
type responseWriter struct {
	http.ResponseWriter
	wroteHeader bool
	tee         io.Writer // if set, Write also copies to this writer
}

func (w *responseWriter) WriteHeader(code int) {
	if !w.wroteHeader {
		w.wroteHeader = true
		w.ResponseWriter.WriteHeader(code)
	}
}

func (w *responseWriter) Write(b []byte) (int, error) {
	if !w.wroteHeader {
		w.WriteHeader(http.StatusOK)
	}
	if w.tee != nil {
		n, _ := w.tee.Write(b)
		slog.Debug("responseWriter tee write", "written", n, "total", len(b))
	}
	return w.ResponseWriter.Write(b)
}

// Flush implements http.Flusher for SSE streaming support.
func (w *responseWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// NewMessagesHandler creates a new messages handler.
func NewMessagesHandler(
	openCodeClient *client.OpenCodeClient,
	modelRouter *router.ModelRouter,
	fallbackHandler *router.FallbackHandler,
	tokenCounter *token.Counter,
	metrics *metrics.Metrics,
	autoRoute bool,
	requestLogger *RequestLogger,
	atomic *config.AtomicConfig,
) *MessagesHandler {
	cfg := atomic.Get()
	timeout := time.Duration(cfg.OpenCodeGo.TimeoutMs) * time.Millisecond
	if timeout == 0 {
		timeout = 5 * time.Minute
	}
	return &MessagesHandler{
		client:              openCodeClient,
		modelRouter:         modelRouter,
		fallbackHandler:     fallbackHandler,
		requestTransformer:  transformer.NewRequestTransformer(),
		responseTransformer: transformer.NewResponseTransformer(),
		streamHandler:       transformer.NewStreamHandler(),
		tokenCounter:        tokenCounter,
		logger:              slog.Default(),
		rateLimiter:         middleware.NewRateLimiter(100, time.Minute),
		requestDedup:        middleware.NewRequestDeduplicator(500 * time.Millisecond),
		requestIDGen:        middleware.NewRequestIDGenerator(),
		metrics:             metrics,
		autoRoute:           autoRoute,
		requestLogger:       requestLogger,
		responseCache:       newStreamResponseCache(),
		timeout:             timeout,
	}
}

// HandleMessages handles POST /v1/messages.
func (h *MessagesHandler) HandleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Generate or get request ID for correlation
	requestID := r.Header.Get("X-Request-ID")
	if requestID == "" {
		requestID = h.requestIDGen.Generate()
	}
	w.Header().Set("X-Request-ID", requestID)

	// Rate limiting
	clientIP := middleware.GetClientIP(r)
	if !h.rateLimiter.Allow(clientIP) {
		h.metrics.RecordRateLimited()
		h.logger.Warn("rate limited", "client", clientIP, "request_id", requestID)
		http.Error(w, "rate limited", http.StatusTooManyRequests)
		return
	}

	// Read the raw request body for debug logging
	var rawBody json.RawMessage
	if err := json.NewDecoder(r.Body).Decode(&rawBody); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	// Deduplicate - skip duplicate requests
	if _, ok := h.requestDedup.TryAcquire(rawBody); !ok {
		h.metrics.RecordDeduplicated()
		h.logger.Info("duplicate request skipped", "request_id", requestID)
		return
	}

	// Parse into Anthropic request
	var anthropicReq types.MessageRequest
	if err := json.Unmarshal(rawBody, &anthropicReq); err != nil {
		h.sendError(w, http.StatusBadRequest, "invalid request body", err)
		return
	}

	// Validate request
	if err := anthropicReq.Validate(); err != nil {
		h.sendError(w, http.StatusBadRequest, err.Error(), nil)
		return
	}

	// Record metrics
	isStreaming := anthropicReq.Stream != nil && *anthropicReq.Stream
	h.metrics.RecordRequest(isStreaming)

	// Increment request counter for file logging
	reqNum := int(h.requestCounter.Add(1))

	h.logger.Info("received request",
		"model", anthropicReq.Model,
		"streaming", isStreaming,
		"messages", len(anthropicReq.Messages),
		"tools", len(anthropicReq.Tools),
		"max_tokens", anthropicReq.MaxTokens,
		"timeout_ms", h.timeout.Milliseconds(),
	)

	// Build message content for routing and token counting.
	var routerMessages []router.MessageContent
	var tokenMessages []token.MessageContent
	systemText := anthropicReq.SystemText()

	for _, msg := range anthropicReq.Messages {
		blocks := msg.ContentBlocks()
		content := extractTextFromBlocks(blocks)
		mc := router.MessageContent{
			Role:    msg.Role,
			Content: content,
		}
		routerMessages = append(routerMessages, mc)
		tokenMessages = append(tokenMessages, token.MessageContent{
			Role:    msg.Role,
			Content: content,
		})
	}

	// Count tokens.
	tokenCount, err := h.tokenCounter.CountMessages(systemText, tokenMessages)
	if err != nil {
		h.logger.Warn("failed to count tokens", "error", err)
		tokenCount = 0
	}

	// Route to appropriate model.
	var routeResult router.RouteResult
	if h.autoRoute {
		// Scenario-based auto-routing (long_context > complex > think > background > default)
		if isStreaming && !h.modelRouter.IsStreamingScenarioRoutingEnabled() {
			routeResult = h.modelRouter.RouteForStreaming(routerMessages, tokenCount)
		} else {
			var err error
			routeResult, err = h.modelRouter.Route(routerMessages, tokenCount)
			if err != nil {
				h.sendError(w, http.StatusInternalServerError, "routing failed", err)
				return
			}
		}
	} else {
		// Direct model lookup: use the model from Claude Code's request.
		// Looks up config.Models[modelName], falls back to passthrough if not found.
		if isStreaming {
			routeResult = h.modelRouter.RouteForStreamingByModel(anthropicReq.Model)
		} else {
			routeResult = h.modelRouter.RouteByModel(anthropicReq.Model)
		}
	}

	h.logger.Info("routing request",
		"scenario", routeResult.Scenario,
		"model", routeResult.Primary.ModelID,
		"tokens", tokenCount,
	)

	// Build fallback chain.
	modelChain := routeResult.GetModelChain()
	primaryModel := modelChain[0].ModelID

	// Compute cache key from rawBody with stream field stripped.
	// This groups streaming and non-streaming duplicates of the same request.
	cacheKey := normalizeForCache(rawBody)

	// Log the Anthropic request body and normalized cache key for debugging.
	h.requestLogger.Log(reqNum, isStreaming, primaryModel, rawBody, "anthropic_req")
	h.requestLogger.Log(reqNum, isStreaming, primaryModel, []byte(cacheKey), "cache_key")

	// Create or get the cache entry BEFORE processing either path.
	// This ensures that a streaming request creates the entry (so store() can
	// find it later) and a non-streaming request that arrives after streaming
	// completes will hit the cache.
	if cached, ok := h.responseCache.getOrCreate(cacheKey); ok {
		// Cache hit -- the streaming response was already cached.
		h.logger.Info("serving request from streaming cache")
		if isStreaming {
			// Replay the cached response as SSE (unlikely but handle it).
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(cached)
		} else {
			h.requestLogger.Log(reqNum, false, primaryModel, cached, "transformed")
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write(cached)
		}
		return
	}

	if isStreaming {
		// Streaming: use ProxyStream for real-time SSE transformation
		h.handleStreaming(w, r, &anthropicReq, modelChain, rawBody, reqNum, primaryModel, cacheKey, tokenCount)
	} else {
		// Non-streaming: execute with fallback and return full response
		h.handleNonStreaming(w, r, &anthropicReq, modelChain, rawBody, reqNum, primaryModel, cacheKey)
	}
}

// normalizeForCache strips fields that differ between streaming and non-streaming
// variants of the same request (the "stream" flag and the billing header cch value),
// so they produce the same cache key.
func normalizeForCache(rawBody json.RawMessage) string {
	var m map[string]any
	if err := json.Unmarshal(rawBody, &m); err != nil {
		return string(rawBody)
	}
	delete(m, "stream")

	// Strip the billing-header cch value from the system prompt.
	// CC embeds "x-anthropic-billing-header: ...; cch=XXXXX;" at the start
	// of the system string, and the cch value changes between requests,
	// preventing cache hits. We strip the cch=<hex> part.
	if sys, ok := m["system"]; ok {
		m["system"] = stripBillingCCH(sys)
	}

	normalized, _ := json.Marshal(m)
	return string(normalized)
}


func stripBillingCCH(sys any) any {
	switch v := sys.(type) {
	case string:
		// Remove "; cch=XXXXX" from the billing header prefix.
		// Format: "...; cch=HEX;You are Claude..."
		cchIdx := strings.Index(v, "; cch=")
		if cchIdx == -1 {
			return v
		}
		// Skip past "; cch=" to find the hex value
		rest := v[cchIdx+len("; cch="):]
		hexEnd := 0
		for hexEnd < len(rest) && (rest[hexEnd] >= '0' && rest[hexEnd] <= '9' ||
			rest[hexEnd] >= 'a' && rest[hexEnd] <= 'f') {
			hexEnd++
		}
		return v[:cchIdx] + rest[hexEnd:]
	case []any:
		result := make([]any, len(v))
		for i, item := range v {
			result[i] = stripBillingCCH(item)
		}
		return result
	case map[string]any:
		result := make(map[string]any, len(v))
		for k, val := range v {
			result[k] = stripBillingCCH(val)
		}
		return result
	}
	return sys
}

// countOutputTokensFromBlocks sums the token count of all text in content blocks.
func (h *MessagesHandler) countOutputTokensFromBlocks(blocks []types.ContentBlock) int {
	var allText strings.Builder
	for _, block := range blocks {
		switch block.Type {
		case "text":
			allText.WriteString(block.Text)
		case "thinking":
			allText.WriteString(block.Thinking)
		case "tool_use":
			allText.WriteString(block.Name)
			allText.WriteString(string(block.Input))
		}
	}
	text := allText.String()
	if text == "" {
		return 0
	}
	count, err := h.tokenCounter.CountTokens(text)
	if err != nil {
		return 0
	}
	return count
}

// injectUsageIfMissing sets input and output tokens on a cached response when the
// upstream streaming didn't include usage (e.g. DeepSeek doesn't send usage in
// streaming chunks even with include_usage=true).
func (h *MessagesHandler) injectUsageIfMissing(cached *types.MessageResponse, inputTokens int) {
	if cached.Usage.InputTokens != 0 || cached.Usage.OutputTokens != 0 {
		return
	}
	cached.Usage.InputTokens = inputTokens
	cached.Usage.OutputTokens = h.countOutputTokensFromBlocks(cached.Content)
}

// parseSSEToMessageResponse extracts a non-streaming MessageResponse from the
// buffered Anthropic SSE stream captured during streaming.
func parseSSEToMessageResponse(sseData []byte) *types.MessageResponse {
	var resp types.MessageResponse
	var contentBlocks []types.ContentBlock
	var currentText string
	var currentToolUse *types.ContentBlock
	var currentThinking string
	currentBlockIndex := -1

	lines := strings.Split(string(sseData), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, ":") {
			// Skip empty lines and SSE comments (keepalives)
			continue
		}
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "" || data == "[DONE]" {
			continue
		}

		var event types.MessageEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		switch event.Type {
		case "message_start":
			if event.Message != nil {
				resp.ID = event.Message.ID
				resp.Model = event.Message.Model
				resp.Type = "message"
				resp.Role = "assistant"
			}
		case "content_block_start":
			if event.ContentBlock != nil {
				currentBlockIndex = *event.Index
				cb := *event.ContentBlock
				switch cb.Type {
				case "text":
					currentText = ""
				case "thinking":
					currentThinking = ""
				case "tool_use":
					tc := cb
					currentToolUse = &tc
				}
			}
		case "content_block_delta":
			if event.Delta != nil {
				switch event.Delta.Type {
				case "text_delta":
					currentText += event.Delta.Text
				case "thinking_delta":
					currentThinking += event.Delta.Thinking
				case "input_json_delta":
					if currentToolUse != nil {
						currentToolUse.Input = append(currentToolUse.Input, []byte(event.Delta.PartialJSON)...)
					}
				}
			}
		case "content_block_stop":
			var cb types.ContentBlock
			if currentToolUse != nil {
				cb = *currentToolUse
				currentToolUse = nil
			} else if currentThinking != "" {
				cb = types.ContentBlock{Type: "thinking", Thinking: currentThinking}
				currentThinking = ""
			} else {
				cb = types.ContentBlock{Type: "text", Text: currentText}
				currentText = ""
			}
			contentBlocks = append(contentBlocks, cb)
			_ = currentBlockIndex
		case "message_delta":
			if event.Delta != nil && event.Delta.StopReason != "" {
				resp.StopReason = event.Delta.StopReason
			}
			if event.Usage != nil {
				resp.Usage = types.Usage{
					InputTokens:              event.Usage.InputTokens,
					OutputTokens:             event.Usage.OutputTokens,
					CacheCreationInputTokens: event.Usage.CacheCreationInputTokens,
					CacheReadInputTokens:     event.Usage.CacheReadInputTokens,
				}
			}
		case "message_stop":
			// If we have an open text/thinking block that wasn't closed,
			// include it in the response.
			if currentText != "" {
				contentBlocks = append(contentBlocks, types.ContentBlock{Type: "text", Text: currentText})
				currentText = ""
			}
			if currentThinking != "" {
				contentBlocks = append(contentBlocks, types.ContentBlock{Type: "thinking", Thinking: currentThinking})
				currentThinking = ""
			}
			if currentToolUse != nil {
				contentBlocks = append(contentBlocks, *currentToolUse)
				currentToolUse = nil
			}
		}
	}

	resp.Content = contentBlocks
	if resp.ID == "" {
		return nil
	}
	return &resp
}

// handleStreaming handles a streaming request with real-time SSE proxying.
func (h *MessagesHandler) handleStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
	reqNum int,
	primaryModel string,
	cacheKey string,
	tokenCount int,
) {
	// Each fallback attempt needs its own context with timeout.
	// Don't share r.Context() across fallbacks - when Claude Code retries,
	// the original context gets canceled and kills all fallbacks.
	clientCtx := r.Context()

	var streamBuf bytes.Buffer
	rw := &responseWriter{ResponseWriter: w, tee: &streamBuf}

	// Set SSE headers immediately so Claude Code knows the stream is alive.
	// This prevents client-side timeouts before we even start sending data.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	rw.WriteHeader(http.StatusOK)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	// Start heartbeat to keep connection alive while waiting for upstream.
	// Claude Code times out after ~6 seconds of no data, so we send pings every 3 seconds
	// (frequent enough to prevent timeout, not so frequent as to cause overhead).
	heartbeatDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				// Send SSE comment (ignored by client but keeps connection alive)
				_, _ = fmt.Fprintf(rw, ":keepalive\n\n")
				if f, ok := w.(http.Flusher); ok {
					f.Flush()
				}
			case <-heartbeatDone:
				return
			case <-clientCtx.Done():
				return
			}
		}
	}()
	// Stop heartbeat when streaming completes
	defer close(heartbeatDone)

	streamStart := time.Now()

	for _, model := range modelChain {
		// Check if client already disconnected before trying this model
		select {
		case <-clientCtx.Done():
			h.logger.Info("client disconnected, stopping streaming fallbacks")
			return
		default:
		}

		h.logger.Info("attempting streaming model", "model", model.ModelID)

		// Create a fresh context with timeout for THIS attempt only.
		// Don't use r.Context() directly - it gets canceled when Claude Code retries.
		ctx, cancel := context.WithTimeout(context.Background(), h.timeout)

		// Check if this is an Anthropic-native model (MiniMax)
		if client.IsAnthropicModel(model.ModelID) {
			// For MiniMax models, send raw Anthropic request to Anthropic endpoint
			// But we need to replace the model name in the raw body
			modelBody := replaceModelInRawBody(rawBody, model.ModelID)
			if err := h.handleAnthropicStreaming(ctx, rw, modelBody, model.ModelID); err != nil {
				cancel()
				// Check if this was a client disconnect
				if clientCtx.Err() == context.Canceled {
					h.logger.Info("client disconnected during anthropic stream")
					return
				}
				h.logger.Warn("anthropic streaming failed", "model", model.ModelID, "error", err)
				continue
			}
			cancel()
			h.requestLogger.Log(reqNum, true, primaryModel, streamBuf.Bytes(), "resp")
			if cached := parseSSEToMessageResponse(streamBuf.Bytes()); cached != nil {
				h.injectUsageIfMissing(cached, tokenCount)
				if cachedJSON, err := json.Marshal(cached); err == nil {
					h.responseCache.store(cacheKey, cachedJSON, 5*time.Second)
				}
			}
			latency := time.Since(streamStart)
			h.metrics.RecordSuccess(model.ModelID, latency)
			h.logger.Info("streaming completed", "model", model.ModelID, "latency", latency)
			return
		}

		// For OpenAI-compatible models, transform and send to OpenAI endpoint
		openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
		if err != nil {
			cancel()
			h.logger.Warn("request transform failed", "model", model.ModelID, "error", err)
			continue
		}
		h.logger.Info("thinking transform",
			append([]any{"model", model.ModelID}, thinkingLogAttrs(anthropicReq, openaiReq, model)...)...,
		)

		// Write outgoing OpenAI streaming request body to file
		if reqJSON, err := json.Marshal(openaiReq); err == nil {
			h.requestLogger.Log(reqNum, true, primaryModel, reqJSON, "req")
		}

		// Get streaming body from upstream
		streamBody, err := h.client.GetStreamingBody(ctx, model.ModelID, openaiReq)
		if err != nil {
			cancel()
			// Check if this was a client disconnect (context canceled)
			if clientCtx.Err() == context.Canceled {
				h.logger.Info("client disconnected during upstream request")
				return
			}
			h.logger.Warn("streaming request failed", "model", model.ModelID, "error", err)
			continue
		}

		// Proxy the stream: transform OpenAI SSE → Anthropic SSE in real-time.
		// Tee the upstream body to capture the raw OpenAI streaming chunks.
		var rawOpenAIBuf bytes.Buffer
		proxyErr := h.streamHandler.ProxyStream(rw, io.NopCloser(io.TeeReader(streamBody, &rawOpenAIBuf)), model.ModelID, clientCtx, tokenCount, h.tokenCounter.CountTokens)
		_ = streamBody.Close()
		h.requestLogger.Log(reqNum, true, primaryModel, rawOpenAIBuf.Bytes(), "openai_resp")

		if proxyErr != nil {
			cancel()
			if proxyErr == transformer.ErrClientDisconnected {
				h.logger.Info("client disconnected during stream")
				return
			}
			if clientCtx.Err() == context.Canceled {
				h.logger.Info("client disconnected during stream (context canceled)")
				return
			}
			h.logger.Warn("stream proxy failed", "model", model.ModelID, "error", proxyErr)
			continue
		}

		cancel()
		h.requestLogger.Log(reqNum, true, primaryModel, streamBuf.Bytes(), "resp")
		if cached := parseSSEToMessageResponse(streamBuf.Bytes()); cached != nil {
			h.injectUsageIfMissing(cached, tokenCount)
			if cachedJSON, err := json.Marshal(cached); err == nil {
				h.responseCache.store(cacheKey, cachedJSON, 5*time.Second)
			}
		}
		latency := time.Since(streamStart)
		h.metrics.RecordSuccess(model.ModelID, latency)
		h.logger.Info("streaming completed", "model", model.ModelID, "latency", latency)
		return
	}

	// All models failed
	h.metrics.RecordFailure()
	if !rw.wroteHeader {
		h.sendError(w, http.StatusBadGateway, "all streaming models failed", nil)
	} else {
		// Headers already sent - send error as SSE event
		h.sendStreamError(rw, "all upstream models failed")
	}
}

// replaceModelInRawBody replaces the model field in raw JSON body with the actual model ID.
// This is needed for Anthropic endpoint which validates the model name.
func replaceModelInRawBody(rawBody json.RawMessage, modelID string) json.RawMessage {
	// Simple string replacement - find "model":"..." and replace with "model":"actual-model"
	bodyStr := string(rawBody)

	// Try to find and replace the model field
	// Pattern: "model":"claude-..." or "model":"any-model-name"
	if idx := strings.Index(bodyStr, `"model":"`); idx != -1 {
		start := idx + len(`"model":"`)
		if end := strings.Index(bodyStr[start:], `"`); end != -1 {
			oldModel := bodyStr[start : start+end]
			// Replace the model value
			newBody := bodyStr[:start] + modelID + bodyStr[start+end:]
			slog.Debug("replaced model in request body",
				"old_model", oldModel,
				"new_model", modelID,
				"success", true)
			return json.RawMessage(newBody)
		}
	}

	slog.Warn("could not find model field in request body, using original",
		"body_preview", bodyStr[:min(len(bodyStr), 200)])
	// If we couldn't parse, return original (will likely fail upstream but that's ok)
	return rawBody
}

// handleAnthropicStreaming sends a raw Anthropic request to the Anthropic endpoint.
func (h *MessagesHandler) handleAnthropicStreaming(
	ctx context.Context,
	w http.ResponseWriter,
	rawBody json.RawMessage,
	modelID string,
) error {
	// Debug: Log what we're sending
	h.logger.Debug("sending anthropic streaming request",
		"model_id", modelID,
		"body_preview", string(rawBody)[:min(len(rawBody), 200)])

	// Send raw Anthropic request to Anthropic endpoint
	// Use ctx so cancellation propagates when client disconnects
	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, true)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()

	// Copy the response directly (already in Anthropic format)
	// SSE headers already set by handleStreaming
	// Use io.Copy which handles streaming efficiently
	_, err = io.Copy(w, resp.Body)
	if err != nil {
		// Check if this was a client disconnect
		if ctx.Err() == context.Canceled {
			return transformer.ErrClientDisconnected
		}
		return fmt.Errorf("failed to copy response: %w", err)
	}

	return nil
}

// sendStreamError sends an error event in the SSE stream.
// Use this when headers have already been written.
func (h *MessagesHandler) sendStreamError(w http.ResponseWriter, message string) {
	h.logger.Error("sending stream error", "message", message)

	errorEvent := map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    "api_error",
			"message": message,
		},
	}

	data, _ := json.Marshal(errorEvent)
	_, _ = fmt.Fprintf(w, "event: error\ndata: %s\n\n", string(data))

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

// handleNonStreaming handles a non-streaming request with fallback.
func (h *MessagesHandler) handleNonStreaming(
	w http.ResponseWriter,
	r *http.Request,
	anthropicReq *types.MessageRequest,
	modelChain []config.ModelConfig,
	rawBody json.RawMessage,
	reqNum int,
	primaryModel string,
	cacheKey string,
) {
	ctx := r.Context()
	startTime := time.Now()

	result, responseBody, err := h.fallbackHandler.ExecuteWithFallback(
		ctx,
		modelChain,
		func(ctx context.Context, model config.ModelConfig) ([]byte, error) {
			// Check if this is an Anthropic-native model (MiniMax)
			if client.IsAnthropicModel(model.ModelID) {
				return h.executeAnthropicRequest(ctx, rawBody, model)
			}
			// Otherwise use OpenAI transformation
			return h.executeOpenAIRequest(ctx, anthropicReq, model, reqNum, primaryModel)
		},
	)

	if err != nil {
		h.metrics.RecordFailure()
		h.sendError(w, http.StatusBadGateway, "all models failed", err)
		return
	}

	latency := time.Since(startTime)
	h.metrics.RecordSuccess(result.ModelID, latency)

	h.logger.Info("request completed",
		"model", result.ModelID,
		"attempts", result.Attempted,
		"latency", latency,
	)

	// Cache the response so subsequent non-streaming duplicates can reuse it.
	h.responseCache.store(cacheKey, responseBody, 5*time.Second)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(responseBody)
}

// executeAnthropicRequest executes a request to the Anthropic endpoint (for MiniMax models).
func (h *MessagesHandler) executeAnthropicRequest(
	ctx context.Context,
	rawBody json.RawMessage,
	model config.ModelConfig,
) ([]byte, error) {
	// Send raw Anthropic request to Anthropic endpoint
	resp, err := h.client.SendAnthropicRequest(ctx, rawBody, false)
	if err != nil {
		return nil, fmt.Errorf("anthropic request failed: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Read the response (already in Anthropic format)
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	h.logger.Debug("anthropic response", "body", string(body))

	return body, nil
}

// executeOpenAIRequest executes a request to the OpenAI endpoint with transformation.
func (h *MessagesHandler) executeOpenAIRequest(
	ctx context.Context,
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
	reqNum int,
	primaryModel string,
) ([]byte, error) {
	// Transform request to OpenAI format.
	openaiReq, err := h.requestTransformer.TransformRequest(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("request transform failed: %w", err)
	}
	h.logger.Info("thinking transform",
		append([]any{"model", model.ModelID}, thinkingLogAttrs(anthropicReq, openaiReq, model)...)...,
	)

	// Write outgoing OpenAI request body to file
	reqJSON, _ := json.Marshal(openaiReq)
	h.requestLogger.Log(reqNum, false, primaryModel, reqJSON, "req")

	// Handle non-streaming.
	resp, err := h.client.ChatCompletionNonStreaming(ctx, model.ModelID, openaiReq)
	if err != nil {
		return nil, fmt.Errorf("chat completion failed: %w", err)
	}

	// Write raw OpenAI response body to file
	respJSON, _ := json.Marshal(resp)
	h.requestLogger.Log(reqNum, false, primaryModel, respJSON, "resp")

	// Transform response to Anthropic format.
	anthropicResp, err := h.responseTransformer.TransformResponse(resp, model.ModelID)
	if err != nil {
		return nil, fmt.Errorf("response transform failed: %w", err)
	}

	// Write final Anthropic-format response to file
	anthropicJSON, _ := json.Marshal(anthropicResp)
	h.requestLogger.Log(reqNum, false, primaryModel, anthropicJSON, "transformed")

	return anthropicJSON, nil
}

// extractTextFromBlocks extracts plain text from Anthropic content blocks.
func extractTextFromBlocks(blocks []types.ContentBlock) string {
	var content string
	for _, block := range blocks {
		switch block.Type {
		case "text":
			content += block.Text
		case "tool_use":
			content += fmt.Sprintf("[Tool Use: %s]", block.Name)
		case "tool_result":
			content += block.TextContent()
		case "thinking":
			// Skip thinking blocks for text extraction
		case "image":
			content += "[Image]"
		}
	}
	return content
}

// sendError sends an error response in Anthropic format.
// Safe to call multiple times - subsequent calls are no-ops.
func (h *MessagesHandler) sendError(w http.ResponseWriter, statusCode int, message string, err error) {
	h.logger.Error("request error",
		"status", statusCode,
		"message", message,
		"error", err,
	)

	// Use the wrapped writer if available to prevent duplicate WriteHeader calls
	if rw, ok := w.(*responseWriter); ok && rw.wroteHeader {
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	errorResp := transformer.TransformErrorResponse(statusCode, message)
	_ = json.NewEncoder(w).Encode(errorResp)
}

func thinkingLogAttrs(anthropicReq *types.MessageRequest, openaiReq *types.ChatCompletionRequest, model config.ModelConfig) []any {
	anthroType := "not_set"
	var anthroBudgetTokens int
	if len(anthropicReq.Thinking) > 0 {
		var m struct {
			Type         string `json:"type"`
			BudgetTokens int    `json:"budget_tokens"`
		}
		if json.Unmarshal(anthropicReq.Thinking, &m) == nil {
			anthroType = m.Type
			anthroBudgetTokens = m.BudgetTokens
		}
	}

	openaiThinkingType := "not_set"
	if len(openaiReq.Thinking) > 0 {
		var m struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(openaiReq.Thinking, &m) == nil {
			openaiThinkingType = m.Type
		}
	}

	reasoningEffort := "not_set"
	if openaiReq.ReasoningEffort != nil {
		reasoningEffort = *openaiReq.ReasoningEffort
	}

	configThinkingType := "not_set"
	if len(model.Thinking) > 0 {
		var m struct {
			Type string `json:"type"`
		}
		if json.Unmarshal(model.Thinking, &m) == nil {
			configThinkingType = m.Type
		}
	}

	anthroEffort := "not_set"
	if anthropicReq.OutputConfig != nil && anthropicReq.OutputConfig.Effort != "" {
		anthroEffort = anthropicReq.OutputConfig.Effort
	}

	return []any{
		"anthro_thinking", anthroType,
		"anthro_budget_tokens", anthroBudgetTokens,
		"anthro_effort", anthroEffort,
		"config_thinking", configThinkingType,
		"openai_thinking", openaiThinkingType,
		"reasoning_effort", reasoningEffort,
		"has_thinking_history", transformer.HasThinkingBlocks(anthropicReq.Messages),
	}
}
