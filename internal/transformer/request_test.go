package transformer

import (
	"bytes"
	"encoding/json"
	"testing"

	"oc-go-cc/internal/config"
	"oc-go-cc/pkg/types"
)

// TestTransformRequestRoundTripReasoning verifies that a DeepSeek response with
// reasoning_content survives the full round-trip (OpenAI response → Anthropic
// response → Anthropic request → OpenAI request) so that on the next turn
// DeepSeed receives the reasoning_content it expects.
func TestTransformRequestRoundTripReasoning(t *testing.T) {
	// Step 1: Simulate a DeepSeek response with reasoning_content.
	deepSeekReasoning := "Let me think step by step"
	openaiResp := &types.ChatCompletionResponse{
		ID:     "resp_123",
		Object: "chat.completion",
		Model:  "deepseek-v4-flash",
		Choices: []types.Choice{{
			Index: 0,
			Message: types.ChatMessage{
				Role:             "assistant",
				Content:          "The answer is 42",
				ReasoningContent: &deepSeekReasoning,
			},
			FinishReason: "stop",
		}},
		Usage: types.UsageInfo{
			PromptTokens:     10,
			CompletionTokens: 20,
		},
	}

	// Step 2: Transform to Anthropic format (what Claude Code receives).
	rt := NewResponseTransformer()
	anthropicResp, err := rt.TransformResponse(openaiResp, "deepseek-v4-flash")
	if err != nil {
		t.Fatalf("TransformResponse error: %v", err)
	}

	// Verify Anthropic response has a thinking block.
	if len(anthropicResp.Content) != 2 {
		t.Fatalf("expected 2 content blocks, got %d", len(anthropicResp.Content))
	}
	if anthropicResp.Content[0].Type != "thinking" {
		t.Fatalf("expected first block to be thinking, got %s", anthropicResp.Content[0].Type)
	}
	if anthropicResp.Content[0].Thinking != deepSeekReasoning {
		t.Fatalf("thinking text = %q, want %q", anthropicResp.Content[0].Thinking, deepSeekReasoning)
	}

	// Step 3: Simulate Claude Code sending the conversation back on the next turn.
	// It includes the previous assistant message with the thinking block.
	anthropicReq := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"What is the answer?"`)},
			{
				Role:    "assistant",
				Content: mustJSONBytes(t, anthropicResp.Content),
			},
			{Role: "user", Content: json.RawMessage(`"Explain why?"`)},
		},
	}

	// Step 4: Transform back to OpenAI request.
	qt := NewRequestTransformer()
	openaiReq, err := qt.TransformRequest(anthropicReq, config.ModelConfig{ModelID: "deepseek-v4-flash"})
	if err != nil {
		t.Fatalf("TransformRequest error: %v", err)
	}

	// Find the assistant message.
	var assistantMsg *types.ChatMessage
	for i := range openaiReq.Messages {
		if openaiReq.Messages[i].Role == "assistant" {
			assistantMsg = &openaiReq.Messages[i]
			break
		}
	}
	if assistantMsg == nil {
		t.Fatal("assistant message not found in transformed request")
	}

	// Step 5: Verify reasoning_content is preserved.
	if assistantMsg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil after round-trip")
	}
	if got, want := *assistantMsg.ReasoningContent, deepSeekReasoning; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}

	// Also verify the JSON serialization includes the field.
	body, err := json.Marshal(openaiReq)
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}
	if !bytes.Contains(body, []byte(`"reasoning_content"`)) {
		t.Fatalf("serialized request missing reasoning_content field: %s", body)
	}
}

func TestTransformRequestPreservesThinkingAsReasoningContent(t *testing.T) {
	transformer := NewRequestTransformer()
	stream := true

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Stream:    &stream,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Need to look up the weather first","signature":"sig_123"},
					{"type":"tool_use","id":"toolu_123","name":"get_weather","input":{"city":"Kigali"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 1; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	msg := openaiReq.Messages[0]
	if got, want := msg.Role, "assistant"; got != want {
		t.Fatalf("Role = %q, want %q", got, want)
	}
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil")
	}
	if got, want := *msg.ReasoningContent, "Need to look up the weather first"; got != want {
		t.Fatalf("ReasoningContent = %q, want %q", got, want)
	}
	if got, want := len(msg.ToolCalls), 1; got != want {
		t.Fatalf("len(ToolCalls) = %d, want %d", got, want)
	}
	if got, want := msg.ToolCalls[0].ID, "toolu_123"; got != want {
		t.Fatalf("ToolCalls[0].ID = %q, want %q", got, want)
	}
	if got, want := msg.ToolCalls[0].Function.Name, "get_weather"; got != want {
		t.Fatalf("ToolCalls[0].Function.Name = %q, want %q", got, want)
	}
	if got, want := msg.ToolCalls[0].Function.Arguments, `{"city":"Kigali"}`; got != want {
		t.Fatalf("ToolCalls[0].Function.Arguments = %q, want %q", got, want)
	}
}

func TestTransformRequestIncludesStreamUsageOptions(t *testing.T) {
	transformer := NewRequestTransformer()
	stream := true

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Stream:    &stream,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.StreamOptions == nil {
		t.Fatal("StreamOptions = nil, want include_usage enabled")
	}
	if !openaiReq.StreamOptions.IncludeUsage {
		t.Fatal("StreamOptions.IncludeUsage = false, want true")
	}
}

func TestTransformRequestOmitsStreamUsageOptionsWhenStreamingDisabled(t *testing.T) {
	transformer := NewRequestTransformer()
	stream := false

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Stream:    &stream,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.StreamOptions != nil {
		t.Fatalf("StreamOptions = %v, want nil when streaming is disabled", openaiReq.StreamOptions)
	}
}

func TestTransformRequestOmitsPlaceholderWhenThinkingInactive(t *testing.T) {
	transformer := NewRequestTransformer()

	// Without thinking history and without config thinking, tool-call-only
	// messages should NOT get a placeholder — providers don't require
	// reasoning_content when thinking mode is inactive.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_456","name":"search_docs","input":{"query":"figma api"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[0]
	if msg.ReasoningContent != nil {
		t.Fatalf("ReasoningContent = %q, want nil (no thinking history, no config thinking)", *msg.ReasoningContent)
	}
}

func TestTransformRequestSerializesAssistantToolCallContent(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_456","name":"search_docs","input":{"query":"figma api"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	body, err := json.Marshal(openaiReq)
	if err != nil {
		t.Fatalf("json.Marshal() error = %v", err)
	}

	var payload struct {
		Messages []map[string]json.RawMessage `json:"messages"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		t.Fatalf("json.Unmarshal() error = %v", err)
	}
	if _, ok := payload.Messages[0]["content"]; !ok {
		t.Fatalf("serialized assistant tool-call message omitted content: %s", body)
	}
	if got, want := string(payload.Messages[0]["content"]), `""`; got != want {
		t.Fatalf("serialized content = %s, want %s", got, want)
	}
}

func TestTransformRequestAppliesReasoningEffortAndThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// When the conversation history already contains thinking blocks,
	// reasoning_effort and thinking should be applied.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"solve this carefully"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"The answer is 42"}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-pro",
		ReasoningEffort: "max",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want max")
	}
	if got, want := *openaiReq.ReasoningEffort, "max"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
}

func TestTransformRequestForwardsReasoningEffortWhenNoThinkingHistory(t *testing.T) {
	transformer := NewRequestTransformer()

	// When config defines reasoning_effort, it is always forwarded regardless
	// of thinking history. Config thinking is also forwarded.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"solve this carefully"`)},
			{Role: "assistant", Content: json.RawMessage(`"The answer is 42"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-pro",
		ReasoningEffort: "max",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want max (config always forwarded)")
	}
	if got, want := *openaiReq.ReasoningEffort, "max"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
}

func TestTransformRequestPreservesSystemCacheControl(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		System: json.RawMessage(`[
			{"type":"text","text":"You are helpful","cache_control":{"type":"ephemeral"}}
		]`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 2; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	systemMsg := openaiReq.Messages[0]
	if got, want := systemMsg.Role, "system"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if got, want := systemMsg.Content, "You are helpful"; got != want {
		t.Fatalf("Messages[0].Content = %q, want %q", got, want)
	}
	if systemMsg.CacheControl == nil {
		t.Fatal("Messages[0].CacheControl = nil, want non-nil")
	}
	if got, want := systemMsg.CacheControl.Type, "ephemeral"; got != want {
		t.Fatalf("Messages[0].CacheControl.Type = %q, want %q", got, want)
	}
}

func TestTransformRequestOmitsCacheControlWhenAbsent(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		System:    json.RawMessage(`"You are helpful"`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 2; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	systemMsg := openaiReq.Messages[0]
	if got, want := systemMsg.Role, "system"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if systemMsg.CacheControl != nil {
		t.Fatalf("Messages[0].CacheControl = %v, want nil", systemMsg.CacheControl)
	}
}

func TestTransformRequestPlacesToolResultsBeforeUserText(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_123","name":"create_file","input":{"name":"draft.fig"}}
				]`),
			},
			{
				Role: "user",
				Content: json.RawMessage(`[
					{"type":"tool_result","tool_use_id":"toolu_123","content":"created"},
					{"type":"text","text":"now continue"}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "kimi-k2.6"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := len(openaiReq.Messages), 3; got != want {
		t.Fatalf("len(Messages) = %d, want %d", got, want)
	}

	if got, want := openaiReq.Messages[0].Role, "assistant"; got != want {
		t.Fatalf("Messages[0].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[1].Role, "tool"; got != want {
		t.Fatalf("Messages[1].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[1].ToolCallID, "toolu_123"; got != want {
		t.Fatalf("Messages[1].ToolCallID = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[2].Role, "user"; got != want {
		t.Fatalf("Messages[2].Role = %q, want %q", got, want)
	}
	if got, want := openaiReq.Messages[2].Content, "now continue"; got != want {
		t.Fatalf("Messages[2].Content = %q, want %q", got, want)
	}
}

func TestTransformRequestOmitsReasoningEffortWhenThinkingDisabled(t *testing.T) {
	transformer := NewRequestTransformer()

	// reasoning_effort is omitted when thinking is explicitly disabled in config,
	// because providers (DeepSeek) reject the combination.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"think carefully"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"The answer is 42"}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-pro",
		ReasoningEffort: "max",
		Thinking:        json.RawMessage(`{"type":"disabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort != nil {
		t.Fatal("ReasoningEffort should be nil when thinking is disabled in config")
	}
	if got, want := string(openaiReq.Thinking), `{"type":"disabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
}

func TestTransformRequestOmitsPlaceholderForDeepSeek(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_456","name":"search_docs","input":{"query":"figma api"}}
				]`),
			},
		},
	}

	// DeepSeek should NOT get a placeholder when there's no thinking history
	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[1] // assistant message
	if msg.ReasoningContent != nil {
		t.Fatalf("ReasoningContent = %q, want nil (DeepSeek without thinking history should not get placeholder)", *msg.ReasoningContent)
	}
}

func TestTransformRequestDeepSeekPlaceholderWithThinkingHistory(t *testing.T) {
	transformer := NewRequestTransformer()

	// When thinking history exists, DeepSeek assistant messages with tool_calls
	// but no thinking block MUST get a placeholder reasoning_content, because
	// DeepSeek requires ALL assistant messages to have reasoning_content in
	// thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"think about this"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"I considered it"}
				]`),
			},
			{Role: "user", Content: json.RawMessage(`"now use a tool"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"search","input":{"q":"test"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-flash",
		ReasoningEffort: "high",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	// Find the second assistant message (tool_call only, no thinking)
	var toolCallAssistant *types.ChatMessage
	for i := range openaiReq.Messages {
		if openaiReq.Messages[i].Role == "assistant" && len(openaiReq.Messages[i].ToolCalls) > 0 {
			toolCallAssistant = &openaiReq.Messages[i]
			break
		}
	}
	if toolCallAssistant == nil {
		t.Fatal("no assistant message with tool_calls found")
	}
	if toolCallAssistant.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for DeepSeek with thinking history")
	}
	if *toolCallAssistant.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *toolCallAssistant.ReasoningContent)
	}
}

func TestTransformRequestDeepSeekPlaceholderTextOnlyWithThinkingHistory(t *testing.T) {
	transformer := NewRequestTransformer()

	// D3 fix: text-only assistant messages (no tool_calls, no thinking)
	// should also get a placeholder when thinking history exists, because
	// DeepSeek requires reasoning_content on ALL assistant messages in
	// thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"think about this"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"I considered it"}
				]`),
			},
			{Role: "user", Content: json.RawMessage(`"ok, what's next"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"text","text":"Next step is..."}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-flash",
		ReasoningEffort: "high",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	// Second assistant message is text-only, no tool_calls, no thinking
	var textOnlyAssistant *types.ChatMessage
	for i := len(openaiReq.Messages) - 1; i >= 0; i-- {
		if openaiReq.Messages[i].Role == "assistant" && len(openaiReq.Messages[i].ToolCalls) == 0 {
			textOnlyAssistant = &openaiReq.Messages[i]
			break
		}
	}
	if textOnlyAssistant == nil {
		t.Fatal("no text-only assistant message found")
	}
	if textOnlyAssistant.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for DeepSeek text-only message with thinking history")
	}
	if *textOnlyAssistant.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *textOnlyAssistant.ReasoningContent)
	}
}

func TestTransformRequestDeepSeekPlaceholderWithConfigThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// D4 fix: when config enforces thinking via thinking: enabled but the
	// conversation history has no thinking blocks, tool-call assistant
	// messages should still get a placeholder because the provider requires
	// reasoning_content in thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"search","input":{"q":"test"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:  "deepseek-v4-flash",
		Thinking: json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[1]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for DeepSeek with config-enforced thinking")
	}
	if *msg.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *msg.ReasoningContent)
	}
}

func TestTransformRequestKimiPlaceholderWithThinkingHistory(t *testing.T) {
	transformer := NewRequestTransformer()

	// When thinking history exists, Kimi assistant messages with tool_calls
	// but no thinking block MUST get a placeholder reasoning_content, because
	// Moonshot requires ALL assistant messages to have reasoning_content in
	// thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"think about this"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"I considered it"}
				]`),
			},
			{Role: "user", Content: json.RawMessage(`"now use a tool"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"search","input":{"q":"test"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "kimi-k2.6",
		ReasoningEffort: "high",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	// Find the second assistant message (tool_call only, no thinking)
	var toolCallAssistant *types.ChatMessage
	for i := range openaiReq.Messages {
		if openaiReq.Messages[i].Role == "assistant" && len(openaiReq.Messages[i].ToolCalls) > 0 {
			toolCallAssistant = &openaiReq.Messages[i]
			break
		}
	}
	if toolCallAssistant == nil {
		t.Fatal("no assistant message with tool_calls found")
	}
	if toolCallAssistant.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for Kimi with thinking history")
	}
	if *toolCallAssistant.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *toolCallAssistant.ReasoningContent)
	}
}

func TestTransformRequestKimiPlaceholderTextOnlyWithThinkingHistory(t *testing.T) {
	transformer := NewRequestTransformer()

	// Text-only assistant messages (no tool_calls, no thinking) should also
	// get a placeholder when thinking history exists, because Moonshot
	// requires reasoning_content on ALL assistant messages in thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"think about this"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"thinking","thinking":"Let me think..."},
					{"type":"text","text":"I considered it"}
				]`),
			},
			{Role: "user", Content: json.RawMessage(`"ok, what's next"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"text","text":"Next step is..."}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "kimi-k2.6",
		ReasoningEffort: "high",
		Thinking:        json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	// Second assistant message is text-only, no tool_calls, no thinking
	var textOnlyAssistant *types.ChatMessage
	for i := len(openaiReq.Messages) - 1; i >= 0; i-- {
		if openaiReq.Messages[i].Role == "assistant" && len(openaiReq.Messages[i].ToolCalls) == 0 {
			textOnlyAssistant = &openaiReq.Messages[i]
			break
		}
	}
	if textOnlyAssistant == nil {
		t.Fatal("no text-only assistant message found")
	}
	if textOnlyAssistant.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for Kimi text-only message with thinking history")
	}
	if *textOnlyAssistant.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *textOnlyAssistant.ReasoningContent)
	}
}

func TestTransformRequestKimiPlaceholderWithConfigThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// When config enforces thinking via thinking: enabled but the
	// conversation history has no thinking blocks, assistant
	// messages should still get a placeholder because Moonshot requires
	// reasoning_content in thinking mode.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"search","input":{"q":"test"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:  "kimi-k2.6",
		Thinking: json.RawMessage(`{"type":"enabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[1]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for Kimi with config-enforced thinking")
	}
	if *msg.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *msg.ReasoningContent)
	}
}

func TestTransformRequestKimiPlaceholderWithRequestThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// When the incoming Anthropic request has thinking enabled but there is
	// no model config thinking and no thinking history, assistant messages
	// should still get a placeholder because the provider requires
	// reasoning_content whenever thinking mode is active.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"enabled","budget_tokens":32000}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
			{
				Role: "assistant",
				Content: json.RawMessage(`[
					{"type":"tool_use","id":"toolu_789","name":"search","input":{"q":"test"}}
				]`),
			},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "kimi-k2.6",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	msg := openaiReq.Messages[1]
	if msg.ReasoningContent == nil {
		t.Fatal("ReasoningContent = nil, want non-nil placeholder for Kimi with request thinking enabled")
	}
	if *msg.ReasoningContent != " " {
		t.Fatalf("ReasoningContent = %q, want placeholder space", *msg.ReasoningContent)
	}
}

func TestAnthropicThinkingToReasoningEffort(t *testing.T) {
	tests := []struct {
		name     string
		thinking string
		want     string
	}{
		{"empty", "", ""},
		{"disabled", `{"type":"disabled","budget_tokens":1000}`, ""},
		{"max budget", `{"type":"enabled","budget_tokens":64000}`, "max"},
		{"max budget boundary", `{"type":"enabled","budget_tokens":128000}`, "max"},
		{"xhigh budget", `{"type":"enabled","budget_tokens":32000}`, "xhigh"},
		{"xhigh budget boundary", `{"type":"enabled","budget_tokens":48000}`, "xhigh"},
		{"high budget", `{"type":"enabled","budget_tokens":16000}`, "high"},
		{"high budget boundary", `{"type":"enabled","budget_tokens":24000}`, "high"},
		{"medium budget", `{"type":"enabled","budget_tokens":8000}`, "medium"},
		{"medium budget boundary", `{"type":"enabled","budget_tokens":4000}`, "medium"},
		{"low budget", `{"type":"enabled","budget_tokens":2000}`, "low"},
		{"low budget boundary", `{"type":"enabled","budget_tokens":1}`, "low"},
		{"enabled no budget", `{"type":"enabled"}`, "low"},
		{"invalid json", `not json`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := anthropicThinkingToReasoningEffort(json.RawMessage(tt.thinking))
			if got != tt.want {
				t.Errorf("got %q, want %q", got, tt.want)
			}
		})
	}
}

func TestAnthropicThinkingToOpenAIThinking(t *testing.T) {
	tests := []struct {
		name     string
		thinking string
		want     string
	}{
		{"empty", "", ""},
		{"disabled", `{"type":"disabled","budget_tokens":1000}`, `{"type":"disabled"}`},
		{"enabled", `{"type":"enabled","budget_tokens":32000}`, `{"type":"enabled"}`},
		{"enabled no budget", `{"type":"enabled"}`, `{"type":"enabled"}`},
		{"adaptive no budget", `{"type":"adaptive"}`, `{"type":"enabled"}`},
		{"adaptive budget zero", `{"type":"adaptive","budget_tokens":0}`, `{"type":"enabled"}`},
		{"adaptive with budget", `{"type":"adaptive","budget_tokens":8000}`, `{"type":"enabled"}`},
		{"invalid json", `not json`, ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := anthropicThinkingToOpenAIThinking(json.RawMessage(tt.thinking))
			if tt.want == "" {
				if got != nil {
					t.Errorf("got %s, want nil", string(got))
				}
			} else {
				if got == nil {
					t.Fatal("got nil, want non-nil")
				}
				if string(got) != tt.want {
					t.Errorf("got %s, want %s", string(got), tt.want)
				}
			}
		})
	}
}

func TestTransformRequestDerivesReasoningEffortFromAnthropicThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// When model config has no reasoning_effort but the Anthropic request has
	// thinking with budget_tokens, reasoning_effort should be derived.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"enabled","budget_tokens":32000}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "kimi-k2.6",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want xhigh (derived from budget_tokens=32000)")
	}
	if got, want := *openaiReq.ReasoningEffort, "xhigh"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
	// No thinking history, no config thinking — thinking should be set from Anthropic request
	if openaiReq.Thinking == nil {
		t.Fatal("Thinking = nil, want enabled (derived from Anthropic request)")
	}
	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
}

func TestTransformRequestConfigReasoningEffortOverridesAnthropicThinking(t *testing.T) {
	transformer := NewRequestTransformer()

	// Config reasoning_effort takes precedence over Anthropic thinking derivation.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"enabled","budget_tokens":32000}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "kimi-k2.6",
		ReasoningEffort: "medium",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want medium")
	}
	if got, want := *openaiReq.ReasoningEffort, "medium"; got != want {
		t.Fatalf("ReasoningEffort = %q, want %q", got, want)
	}
}

func TestTransformRequestSkipsReasoningEffortWhenConfigThinkingDisabled(t *testing.T) {
	transformer := NewRequestTransformer()

	// When config has thinking=disabled, reasoning_effort must NOT be set
	// from ANY source (config OR derived from incoming request), because
	// DeepSeek rejects the combination.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"enabled","budget_tokens":32000}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID:         "deepseek-v4-flash",
		ReasoningEffort: "max",
		Thinking:        json.RawMessage(`{"type":"disabled"}`),
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if openaiReq.ReasoningEffort != nil {
		t.Fatalf("ReasoningEffort = %q, want nil (config thinking is disabled, must block all sources)", *openaiReq.ReasoningEffort)
	}
	if got, want := string(openaiReq.Thinking), `{"type":"disabled"}`; got != want {
		t.Fatalf("Thinking = %s, want %s", got, want)
	}
}

func TestTransformRequestDefaultsThinkingDisabled(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{ModelID: "deepseek-v4-pro"})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"disabled"}`; got != want {
		t.Fatalf("Thinking = %s, want disabled when nothing enables it", got)
	}
	if openaiReq.ReasoningEffort != nil {
		t.Fatalf("ReasoningEffort = %q, want nil", *openaiReq.ReasoningEffort)
	}
}

func TestTransformRequestAdaptiveThinkingMapsToEnabled(t *testing.T) {
	transformer := NewRequestTransformer()

	// adaptive type (CC's "low"/"medium"/"high") always maps to thinking=enabled.
	// The effort level comes from output_config.effort, not budget_tokens.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"adaptive"}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "deepseek-v4-pro",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want enabled for adaptive", got)
	}
	// No reasoning_effort without output_config.effort or config
	if openaiReq.ReasoningEffort != nil {
		t.Fatalf("ReasoningEffort = %q, want nil (no effort source)", *openaiReq.ReasoningEffort)
	}
}

func TestTransformRequestOutputConfigEffortMapsToReasoningEffort(t *testing.T) {
	transformer := NewRequestTransformer()

	// CC effort level from output_config.effort maps to reasoning_effort.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"adaptive"}`),
		OutputConfig: &types.OutputConfig{
			Effort: "medium",
		},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "deepseek-v4-pro",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want enabled", got)
	}
	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want medium from output_config.effort")
	}
	if got, want := *openaiReq.ReasoningEffort, "medium"; got != want {
		t.Fatalf("ReasoningEffort = %q, want medium", got)
	}
}

func TestTransformRequestSkipsReasoningEffortWhenThinkingNotSet(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:        "claude-test",
		MaxTokens:    256,
		OutputConfig: &types.OutputConfig{Effort: "medium"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "deepseek-v4-pro",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"disabled"}`; got != want {
		t.Fatalf("Thinking = %s, want disabled (nothing enables it)", got)
	}
	if openaiReq.ReasoningEffort != nil {
		t.Fatalf("ReasoningEffort = %q, want nil (thinking disabled)", *openaiReq.ReasoningEffort)
	}
}

func TestTransformRequestSkipsReasoningEffortWhenThinkingDisabled(t *testing.T) {
	transformer := NewRequestTransformer()

	req := &types.MessageRequest{
		Model:        "claude-test",
		MaxTokens:    256,
		Thinking:     json.RawMessage(`{"type":"disabled"}`),
		OutputConfig: &types.OutputConfig{Effort: "medium"},
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "deepseek-v4-pro",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"disabled"}`; got != want {
		t.Fatalf("Thinking = %s, want disabled", got)
	}
	if openaiReq.ReasoningEffort != nil {
		t.Fatalf("ReasoningEffort = %q, want nil (thinking disabled)", *openaiReq.ReasoningEffort)
	}
}

func TestTransformRequestAdaptiveThinkingWithBudgetMapsToEnabled(t *testing.T) {
	transformer := NewRequestTransformer()

	// CC "medium"/"high" setting: adaptive with budget > 0
	// should map to thinking=enabled with reasoning_effort derived.
	req := &types.MessageRequest{
		Model:     "claude-test",
		MaxTokens: 256,
		Thinking:  json.RawMessage(`{"type":"adaptive","budget_tokens":8000}`),
		Messages: []types.Message{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}

	openaiReq, err := transformer.TransformRequest(req, config.ModelConfig{
		ModelID: "deepseek-v4-pro",
	})
	if err != nil {
		t.Fatalf("TransformRequest() error = %v", err)
	}

	if got, want := string(openaiReq.Thinking), `{"type":"enabled"}`; got != want {
		t.Fatalf("Thinking = %s, want enabled for adaptive with budget", got)
	}
	if openaiReq.ReasoningEffort == nil {
		t.Fatal("ReasoningEffort = nil, want medium (derived from budget_tokens=8000)")
	}
	if got, want := *openaiReq.ReasoningEffort, "medium"; got != want {
		t.Fatalf("ReasoningEffort = %q, want medium", got)
	}
}

func mustJSONBytes(t *testing.T, v any) json.RawMessage {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return json.RawMessage(b)
}
