# Configuration

## Config File

Location: `~/.config/oc-go-cc/config.json`

Override with `OC_GO_CC_CONFIG` environment variable.

## Full Config Reference

```json
{
  "api_key": "${OC_GO_CC_API_KEY}",
  "host": "127.0.0.1",
  "port": 3456,
  "hot_reload": false,

  "models": {
    "default": {
      "provider": "opencode-go",
      "model_id": "kimi-k2.6",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "background": {
      "provider": "opencode-go",
      "model_id": "qwen3.5-plus",
      "temperature": 0.5,
      "max_tokens": 2048
    },
    "think": {
      "provider": "opencode-go",
      "model_id": "glm-5.1",
      "temperature": 0.7,
      "max_tokens": 8192
    },
    "complex": {
      "provider": "opencode-go",
      "model_id": "glm-5.1",
      "temperature": 0.7,
      "max_tokens": 4096
    },
    "long_context": {
      "provider": "opencode-go",
      "model_id": "minimax-m2.7",
      "temperature": 0.7,
      "max_tokens": 16384,
      "context_threshold": 80000
    },
    "fast": {
      "provider": "opencode-go",
      "model_id": "qwen3.6-plus",
      "temperature": 0.7,
      "max_tokens": 4096
    }
  },

  "fallbacks": {
    "default": [
      { "provider": "opencode-go", "model_id": "glm-5" },
      { "provider": "opencode-go", "model_id": "qwen3.6-plus" }
    ],
    "think": [{ "provider": "opencode-go", "model_id": "glm-5" }],
    "complex": [{ "provider": "opencode-go", "model_id": "glm-5" }],
    "long_context": [{ "provider": "opencode-go", "model_id": "minimax-m2.5" }],
    "fast": [{ "provider": "opencode-go", "model_id": "qwen3.5-plus" }]
  },

  "opencode_go": {
    "base_url": "https://opencode.ai/zen/go/v1/chat/completions",
    "timeout_ms": 300000
  },

  "logging": {
    "level": "info",
    "requests": true
  }
}
```

## Environment Variables

Environment variables override config file values. Config values also support `${VAR}` interpolation.

| Variable                | Description                                 | Default                                          |
| ----------------------- | ------------------------------------------- | ------------------------------------------------ |
| `OC_GO_CC_API_KEY`      | OpenCode Go API key (**required**)          | —                                                |
| `OC_GO_CC_CONFIG`       | Custom config file path                     | `~/.config/oc-go-cc/config.json`                 |
| `OC_GO_CC_HOST`         | Proxy listen host                           | `127.0.0.1`                                      |
| `OC_GO_CC_PORT`         | Proxy listen port                           | `3456`                                           |
| `OC_GO_CC_OPENCODE_URL` | OpenCode Go API endpoint                    | `https://opencode.ai/zen/go/v1/chat/completions` |
| `OC_GO_CC_LOG_LEVEL`    | Log level: `debug`, `info`, `warn`, `error` | `info`                                           |

## Hot Reload

By default, config changes require a server restart. Enable hot reload to watch for config file changes and apply them automatically:

```json
{
  "hot_reload": true
}
```

When enabled, the proxy watches the config directory for changes (handling editors that save via rename/create) and reloads the config automatically. You can also trigger a manual reload by sending `SIGHUP` to the process:

```bash
kill -HUP <PID>
```

## Model Routing

### Direct Model Mapping (Default)

By default, the proxy uses the model from Claude Code's request directly. It looks up the incoming model name (e.g., `"claude-sonnet-4-6"`) in `config.models` and uses the mapped config if found. If no mapping exists, the model name is passed through as-is — so you only need to add entries for models you want to remap.

```json
{
  "models": {
    "claude-sonnet-4-6": {
      "provider": "opencode-go",
      "model_id": "deepseek-v4-pro",
      "temperature": 0.1,
      "max_tokens": 8192,
      "reasoning_effort": "max",
      "thinking": { "type": "enabled" }
    },
    "claude-opus-4-7": {
      "provider": "opencode-go",
      "model_id": "deepseek-v4-pro",
      "temperature": 0.1,
      "max_tokens": 16384,
      "reasoning_effort": "max",
      "thinking": { "type": "enabled" }
    }
  }
}
```

With this config, `claude-sonnet-4-6` and `claude-opus-4-7` both map to DeepSeek V4 Pro. Any other model (e.g., `claude-haiku-4-5`) would pass through as-is if no mapping is configured.

If you don't configure any model-name entries at all, every model passes through unchanged — the proxy just transforms the format.

### Auto-Route (Opt-in)

Enable scenario-based auto-routing with `--auto-route` or `OC_GO_CC_AUTO_ROUTE=true`. When enabled, the proxy automatically detects the type of request and routes to the appropriate model based on context size and content analysis:

| Scenario         | Trigger                                             | Model        | Why                                             |
| ---------------- | --------------------------------------------------- | ------------ | ----------------------------------------------- |
| **Long Context** | >80K tokens (configurable)                          | MiniMax M2.7 | 1M context window vs 128-256K for others        |
| **Complex**      | "architect", "refactor", "complex" in system prompt | GLM-5.1      | Best reasoning & architectural understanding    |
| **Think**        | "think", "plan", "reason" in system prompt          | GLM-5        | Good reasoning, cheaper than GLM-5.1            |
| **Background**   | "read file", "grep", "list directory"               | Qwen3.5 Plus | Cheapest (~10K req/5hr), perfect for simple ops |
| **Default**      | Everything else                                     | Kimi K2.6    | Best balance of quality & cost (~1.8K req/5hr)  |

**See [MODELS.md](MODELS.md) for detailed model capabilities, costs, and routing recommendations.**

DeepSeek V4 users can set any scenario model to `deepseek-v4-pro` or `deepseek-v4-flash`. For deterministic max thinking, add `reasoning_effort: "max"` and `thinking: {"type":"enabled"}` to that scenario's model config and fallback entries.

### Routing in Detail

| Scenario         | Trigger                                                                      | Config Key            | Default Model  |
| ---------------- | ---------------------------------------------------------------------------- | --------------------- | -------------- |
| **Default**      | Standard chat                                                                | `models.default`      | `kimi-k2.6`    |
| **Think**        | System prompt contains "think", "plan", "reason"; or thinking content blocks | `models.think`        | `glm-5.1`      |
| **Long Context** | Token count exceeds `context_threshold`                                      | `models.long_context` | `minimax-m2.7` |
| **Background**   | File read, directory list, grep patterns                                     | `models.background`   | `qwen3.5-plus` |

Routing priority: **Long Context** > **Think** > **Background** > **Default**

## Fallback Chains

When a model request fails (network error, rate limit, server error), the proxy tries the next model in the fallback chain:

```
Primary model -> Fallback 1 -> Fallback 2 -> ... -> Error (all failed)
```

Each model also has a **circuit breaker** that tracks consecutive failures. After 3 failures, the circuit opens and that model is skipped for 30 seconds, then tested again (half-open state).
