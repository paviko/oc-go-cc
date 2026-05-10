// Package transformer handles request and response format conversion
// between Anthropic Messages API and OpenAI Chat Completions API.
package transformer

import (
	"encoding/json"
	"fmt"
	"strings"

	"oc-go-cc/internal/config"
	"oc-go-cc/pkg/types"
)

// RequestTransformer converts Anthropic requests to OpenAI format.
type RequestTransformer struct{}

// NewRequestTransformer creates a new request transformer.
func NewRequestTransformer() *RequestTransformer {
	return &RequestTransformer{}
}

// anthropicThinkingToReasoningEffort maps Anthropic's thinking config (which uses
// budget_tokens) to an OpenAI reasoning_effort value.
// Returns empty string if thinking is disabled or absent.
func anthropicThinkingToReasoningEffort(thinking json.RawMessage) string {
	if len(thinking) == 0 {
		return ""
	}
	var m struct {
		Type         string `json:"type"`
		BudgetTokens *int   `json:"budget_tokens"`
	}
	if err := json.Unmarshal(thinking, &m); err != nil {
		return ""
	}
	if m.Type == "disabled" {
		return ""
	}
	if m.BudgetTokens != nil {
		switch {
		case *m.BudgetTokens >= 64000:
			return "max"
		case *m.BudgetTokens >= 32000:
			return "xhigh"
		case *m.BudgetTokens >= 16000:
			return "high"
		case *m.BudgetTokens >= 4000:
			return "medium"
		default:
			return "low"
		}
	}
	if m.Type == "enabled" {
		return "low"
	}
	return ""
}

// anthropicThinkingToOpenAIThinking maps Anthropic's thinking config to the
// OpenAI/DeepSeek thinking format ({"type":"enabled"} or {"type":"disabled"}).
// "adaptive" with no budget or budget=0 means the model should think minimally,
// which maps to disabled on the OpenAI side.
func anthropicThinkingToOpenAIThinking(thinking json.RawMessage) json.RawMessage {
	if len(thinking) == 0 {
		return nil
	}
	var m struct {
		Type         string `json:"type"`
		BudgetTokens *int   `json:"budget_tokens"`
	}
	if err := json.Unmarshal(thinking, &m); err != nil {
		return nil
	}
	if m.Type == "disabled" {
		return json.RawMessage(`{"type":"disabled"}`)
	}
	if m.Type == "enabled" || m.Type == "adaptive" {
		return json.RawMessage(`{"type":"enabled"}`)
	}
	return nil
}

// isThinkingDisabled returns true when the OpenAI/DeepSeek thinking field
// is explicitly set to {"type":"disabled"}.
func isThinkingDisabled(thinking json.RawMessage) bool {
	if len(thinking) == 0 {
		return false
	}
	var m struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(thinking, &m); err != nil {
		return false
	}
	return m.Type == "disabled"
}

// isDeepSeekModel returns true for DeepSeek models that require thinking mode handling.
func isDeepSeekModel(modelID string) bool {
	return strings.HasPrefix(modelID, "deepseek-")
}

// effortToReasoningEffort maps Claude Code's output_config.effort to an
// OpenAI reasoning_effort value. Returns empty string if unknown.
func effortToReasoningEffort(effort string) string {
	switch effort {
	case "low", "medium", "high", "xhigh", "max":
		return effort
	default:
		return ""
	}
}

// needsPlaceholderReasoning returns true for providers whose validators require
// a non-empty reasoning_content field on assistant tool-call messages.
func needsPlaceholderReasoning(modelID string) bool {
	// Moonshot's validator treats an empty string as missing.
	return strings.HasPrefix(modelID, "kimi-")
}

// TransformRequest converts an Anthropic MessageRequest to OpenAI ChatCompletionRequest.
func (t *RequestTransformer) TransformRequest(
	anthropicReq *types.MessageRequest,
	model config.ModelConfig,
) (*types.ChatCompletionRequest, error) {
	// Transform messages
	messages, err := t.transformMessages(anthropicReq, model)
	if err != nil {
		return nil, fmt.Errorf("failed to transform messages: %w", err)
	}

	// Build OpenAI request
	openaiReq := &types.ChatCompletionRequest{
		Model:    model.ModelID,
		Messages: messages,
		Stream:   anthropicReq.Stream,
	}
	if anthropicReq.Stream != nil && *anthropicReq.Stream {
		openaiReq.StreamOptions = &types.StreamOptions{IncludeUsage: true}
	}

	// Copy optional parameters from Anthropic request
	if anthropicReq.Temperature != nil {
		openaiReq.Temperature = anthropicReq.Temperature
	}
	if anthropicReq.TopP != nil {
		openaiReq.TopP = anthropicReq.TopP
	}

	// Map max_tokens
	if anthropicReq.MaxTokens > 0 {
		maxTokens := anthropicReq.MaxTokens
		openaiReq.MaxTokens = &maxTokens
	}

	// Apply model-specific overrides
	if model.Temperature > 0 {
		openaiReq.Temperature = &model.Temperature
	}
	if model.MaxTokens > 0 {
		maxTokens := model.MaxTokens
		openaiReq.MaxTokens = &maxTokens
	}

	// thinking: config takes priority, then derive from history, then from incoming request.
	// thinking is the gate for reasoning_effort — only when thinking=enabled do we set effort.
	hasThinkingInHistory := HasThinkingBlocks(anthropicReq.Messages)
	if len(model.Thinking) > 0 {
		openaiReq.Thinking = model.Thinking
	} else if hasThinkingInHistory {
		if len(anthropicReq.Thinking) > 0 {
			openaiReq.Thinking = anthropicThinkingToOpenAIThinking(anthropicReq.Thinking)
		} else {
			openaiReq.Thinking = json.RawMessage(`{"type":"enabled"}`)
		}
	} else if len(anthropicReq.Thinking) > 0 {
		openaiReq.Thinking = anthropicThinkingToOpenAIThinking(anthropicReq.Thinking)
	} else {
		openaiReq.Thinking = json.RawMessage(`{"type":"disabled"}`)
	}

	// reasoning_effort: only set when thinking is explicitly enabled.
	// Config > budget_tokens > output_config.effort.
	if openaiReq.Thinking != nil && !isThinkingDisabled(openaiReq.Thinking) {
		if model.ReasoningEffort != "" {
			openaiReq.ReasoningEffort = &model.ReasoningEffort
		} else if effort := anthropicThinkingToReasoningEffort(anthropicReq.Thinking); effort != "" {
			openaiReq.ReasoningEffort = &effort
		} else if anthropicReq.OutputConfig != nil {
			if effort := effortToReasoningEffort(anthropicReq.OutputConfig.Effort); effort != "" {
				openaiReq.ReasoningEffort = &effort
			}
		}
	}

	// Transform tools if present
	if len(anthropicReq.Tools) > 0 {
		openaiReq.Tools = t.transformTools(anthropicReq.Tools)
	}

	return openaiReq, nil
}

// HasThinkingBlocks returns true if any assistant message contains a
// thinking content block.
func HasThinkingBlocks(messages []types.Message) bool {
	for _, msg := range messages {
		if msg.Role != "assistant" {
			continue
		}
		for _, block := range msg.ContentBlocks() {
			if block.Type == "thinking" {
				return true
			}
		}
	}
	return false
}

// transformMessages converts Anthropic messages to OpenAI format.
func (t *RequestTransformer) transformMessages(anthropicReq *types.MessageRequest, model config.ModelConfig) ([]types.ChatMessage, error) {
	hasThinking := HasThinkingBlocks(anthropicReq.Messages) ||
		(len(model.Thinking) > 0 && !isThinkingDisabled(model.Thinking))

	var result []types.ChatMessage

	// Add system message if present, preserving cache_control if available
	systemText := anthropicReq.SystemText()
	if systemText != "" {
		systemMsg := types.ChatMessage{
			Role:    "system",
			Content: systemText,
		}
		// Try to extract cache_control from system array blocks
		if len(anthropicReq.System) > 0 {
			var blocks []types.SystemContentBlock
			if err := json.Unmarshal(anthropicReq.System, &blocks); err == nil {
				for _, b := range blocks {
					if b.Type == "text" && b.CacheControl != nil {
						systemMsg.CacheControl = b.CacheControl
						break
					}
				}
			}
		}
		result = append(result, systemMsg)
	}

	// Transform each message
	for _, msg := range anthropicReq.Messages {
		openaiMsgs, err := t.transformMessage(msg, model.ModelID, hasThinking)
		if err != nil {
			return nil, err
		}
		result = append(result, openaiMsgs...)
	}

	return result, nil
}

// transformMessage converts a single Anthropic message to one or more OpenAI messages.
// Tool_use and tool_result require special handling to map to OpenAI's function calling format.
func (t *RequestTransformer) transformMessage(msg types.Message, modelID string, hasThinkingInHistory bool) ([]types.ChatMessage, error) {
	blocks := msg.ContentBlocks()

	switch msg.Role {
	case "user":
		return t.transformUserMessage(blocks)
	case "assistant":
		return t.transformAssistantMessage(blocks, modelID, hasThinkingInHistory)
	default:
		// Fallback: concatenate all text
		var text string
		for _, b := range blocks {
			if b.Type == "text" {
				text += b.Text
			}
		}
		return []types.ChatMessage{{Role: msg.Role, Content: text}}, nil
	}
}

// transformUserMessage converts a user message with potential tool_result blocks.
func (t *RequestTransformer) transformUserMessage(blocks []types.ContentBlock) ([]types.ChatMessage, error) {
	var result []types.ChatMessage
	var textParts []string

	for _, block := range blocks {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "tool_result":
			// In OpenAI, tool results are separate messages with role "tool"
			toolContent := block.TextContent()
			result = append(result, types.ChatMessage{
				Role:       "tool",
				Content:    toolContent,
				ToolCallID: block.GetToolID(),
			})
		case "image":
			// Images not supported in text-only models, skip
			textParts = append(textParts, "[Image]")
		}
	}

	// If there's text content, add it as a user message
	if len(textParts) > 0 {
		text := ""
		for _, p := range textParts {
			text += p
		}
		// OpenAI-compatible tool calling requires tool responses to appear
		// immediately after the assistant message that emitted tool_calls.
		// If the Anthropic user turn also includes free-form text, emit it as
		// a subsequent user message after all tool results.
		userMsg := types.ChatMessage{Role: "user", Content: text}
		result = append(result, userMsg)
	}

	return result, nil
}

// transformAssistantMessage converts an assistant message with potential tool_use blocks.
func (t *RequestTransformer) transformAssistantMessage(blocks []types.ContentBlock, modelID string, hasThinkingInHistory bool) ([]types.ChatMessage, error) {
	var textParts []string
	var thinkingParts []string
	var toolCalls []types.ToolCall

	for _, block := range blocks {
		switch block.Type {
		case "text":
			textParts = append(textParts, block.Text)
		case "thinking":
			// Preserve chain-of-thought so it can be forwarded back to providers
			// that require reasoning_content to be preserved across turns.
			if block.Thinking != "" {
				thinkingParts = append(thinkingParts, block.Thinking)
			}
		case "tool_use":
			// Map to OpenAI function call format
			arguments := "{}"
			if len(block.Input) > 0 {
				arguments = string(block.Input)
			}
			toolCalls = append(toolCalls, types.ToolCall{
				ID:   block.ID,
				Type: "function",
				Function: types.FunctionCall{
					Name:      block.Name,
					Arguments: arguments,
				},
			})
		}
	}

	// Build the assistant message
	content := ""
	for _, p := range textParts {
		content += p
	}
	reasoningContent := ""
	for _, p := range thinkingParts {
		reasoningContent += p
	}

	var reasoningContentPtr *string
	if reasoningContent != "" {
		reasoningContentPtr = &reasoningContent
	} else if hasThinkingInHistory && isDeepSeekModel(modelID) {
		placeholder := " "
		reasoningContentPtr = &placeholder
	} else if hasThinkingInHistory && len(toolCalls) > 0 && needsPlaceholderReasoning(modelID) {
		placeholder := " "
		reasoningContentPtr = &placeholder
	}

	msg := types.ChatMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoningContentPtr,
		ToolCalls:        toolCalls,
	}

	return []types.ChatMessage{msg}, nil
}

// transformTools converts Anthropic tools to OpenAI tools.
func (t *RequestTransformer) transformTools(tools []types.Tool) []types.ToolDef {
	var result []types.ToolDef

	for _, tool := range tools {
		// InputSchema is already json.RawMessage, use it directly
		schema := tool.InputSchema
		if len(schema) == 0 {
			schema = []byte(`{"type":"object","properties":{}}`)
		}

		result = append(result, types.ToolDef{
			Type: "function",
			Function: types.FunctionDef{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  json.RawMessage(schema),
			},
		})
	}

	return result
}
