//go:build unit

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestForwardResponses_ForceChatCompletionsRoutesNonStreamingToChatCompletions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_resp_chat_json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_json","object":"chat.completion","model":"gpt-5.4","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1}}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
	require.Equal(t, HTTPUpstreamProfileOpenAI, HTTPUpstreamProfileFromContext(upstream.lastReq.Context()))
	require.Equal(t, "hello", gjson.GetBytes(upstream.lastBody, "messages.0.content").String())
	require.False(t, gjson.GetBytes(upstream.lastBody, "input").Exists())
	require.Equal(t, "response", gjson.Get(rec.Body.String(), "object").String())
	require.Equal(t, "ok", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
	require.Equal(t, 3, result.Usage.InputTokens)
	require.Equal(t, 2, result.Usage.OutputTokens)
	require.Equal(t, 1, result.Usage.CacheReadInputTokens)
	require.False(t, result.Stream)
}

func TestForwardResponses_MappedDeepSeekThinkingPreservesSequentialToolReasoning(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{
		"model":"sol",
		"reasoning":{"effort":"high"},
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"run it"}]},
			{"type":"reasoning","id":"rs_a","summary":[{"type":"summary_text","text":"first exact reasoning"}]},
			{"type":"function_call","call_id":"call_a","name":"exec_command","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_a","output":"ok"},
			{"type":"reasoning","id":"rs_b","summary":[{"type":"summary_text","text":"second exact reasoning"}]},
			{"type":"function_call","call_id":"call_b","name":"exec_command","arguments":"{}"},
			{"type":"function_call_output","call_id":"call_b","output":"ok"}
		],
		"stream":false
	}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_deepseek","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"fresh private reasoning","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2,"total_tokens":6,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}`,
		)),
	}}
	account := forceChatResponsesFallbackAccount()
	account.Credentials["model_mapping"] = map[string]any{"sol": "deepseek-v4-flash"}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "deepseek-v4-flash", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "high", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
	require.Equal(t, "assistant", gjson.GetBytes(upstream.lastBody, "messages.1.role").String())
	require.Equal(t, "first exact reasoning", gjson.GetBytes(upstream.lastBody, "messages.1.reasoning_content").String())
	require.Equal(t, "call_a", gjson.GetBytes(upstream.lastBody, "messages.1.tool_calls.0.id").String())
	require.Equal(t, "tool", gjson.GetBytes(upstream.lastBody, "messages.2.role").String())
	require.Equal(t, "assistant", gjson.GetBytes(upstream.lastBody, "messages.3.role").String())
	require.Equal(t, "second exact reasoning", gjson.GetBytes(upstream.lastBody, "messages.3.reasoning_content").String())
	require.Equal(t, "call_b", gjson.GetBytes(upstream.lastBody, "messages.3.tool_calls.0.id").String())
	require.Equal(t, "tool", gjson.GetBytes(upstream.lastBody, "messages.4.role").String())
	require.Equal(t, "reasoning", gjson.Get(rec.Body.String(), "output.0.type").String())
	require.Equal(t, "fresh private reasoning", gjson.Get(rec.Body.String(), "output.0.summary.0.text").String())
	require.Equal(t, "message", gjson.Get(rec.Body.String(), "output.1.type").String())
	require.Equal(t, "done", gjson.Get(rec.Body.String(), "output.1.content.0.text").String())
	require.Equal(t, int64(1), gjson.Get(rec.Body.String(), "usage.input_tokens_details.cached_tokens").Int())
	require.Equal(t, int64(1), gjson.Get(rec.Body.String(), "usage.output_tokens_details.reasoning_tokens").Int())
}

func TestForwardResponses_ForceChatCompletionsRoutesStreamingToChatCompletions(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl_stream","object":"chat.completion.chunk","model":"gpt-5.4","choices":[],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_resp_chat_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/chat/completions", upstream.lastReq.URL.String())
	require.True(t, gjson.GetBytes(upstream.lastBody, "stream_options.include_usage").Bool())
	require.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"he"`)
	require.Contains(t, rec.Body.String(), "event: response.completed")
	require.Contains(t, rec.Body.String(), `"input_tokens":4`)
	require.Contains(t, rec.Body.String(), "data: [DONE]")
	require.Equal(t, 4, result.Usage.InputTokens)
	require.Equal(t, 3, result.Usage.OutputTokens)
	require.True(t, result.Stream)
	require.NotNil(t, result.FirstTokenMs)
}

func TestForwardResponses_DeepSeekReasoningAndTextStreamPreservesUsageBreakdown(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"deepseek-v4-flash","input":"hello","reasoning":{"effort":"max"},"stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_reasoning_usage","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":""},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning_usage","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"reasoning_content":"private reasoning"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning_usage","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{"content":"FINAL"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning_usage","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning_usage","object":"chat.completion.chunk","model":"deepseek-v4-flash","choices":[],"usage":{"prompt_tokens":109,"completion_tokens":13,"total_tokens":122,"prompt_tokens_details":{"cached_tokens":4},"completion_tokens_details":{"reasoning_tokens":11}}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_deepseek_reasoning_usage"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "max", gjson.GetBytes(upstream.lastBody, "reasoning_effort").String())
	require.Contains(t, rec.Body.String(), "event: response.reasoning_summary_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"private reasoning"`)
	require.Contains(t, rec.Body.String(), "event: response.output_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"FINAL"`)
	for _, frame := range strings.Split(rec.Body.String(), "\n\n") {
		if strings.Contains(frame, "event: response.output_text.delta") {
			require.NotContains(t, frame, "private reasoning")
		}
	}
	terminalType, terminalPayload, ok := extractOpenAISSETerminalEvent(rec.Body.String())
	require.True(t, ok)
	require.Equal(t, "response.completed", terminalType)
	require.Equal(t, int64(109), gjson.GetBytes(terminalPayload, "response.usage.input_tokens").Int())
	require.Equal(t, int64(4), gjson.GetBytes(terminalPayload, "response.usage.input_tokens_details.cached_tokens").Int())
	require.Equal(t, int64(13), gjson.GetBytes(terminalPayload, "response.usage.output_tokens").Int())
	require.Equal(t, int64(11), gjson.GetBytes(terminalPayload, "response.usage.output_tokens_details.reasoning_tokens").Int())
	require.Equal(t, int64(122), gjson.GetBytes(terminalPayload, "response.usage.total_tokens").Int())
	require.Equal(t, 109, result.Usage.InputTokens)
	require.Equal(t, 13, result.Usage.OutputTokens)
}

func TestForwardResponses_DeepSeekReasoningOnlyStreamStaysOutOfVisibleText(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"deepseek-reasoner","input":"hello","stream":true}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstreamBody := strings.Join([]string{
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"role":"assistant","content":null,"reasoning_content":""},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"reasoning_content":"private reasoning"},"finish_reason":null}]}`,
		"",
		`data: {"id":"chatcmpl_reasoning","object":"chat.completion.chunk","model":"deepseek-reasoner","choices":[{"index":0,"delta":{"content":""},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":3,"total_tokens":7}}`,
		"",
		"data: [DONE]",
		"",
	}, "\n")
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}, "x-request-id": []string{"rid_deepseek_reasoning_responses_stream"}},
		Body:       io.NopCloser(strings.NewReader(upstreamBody)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Contains(t, rec.Body.String(), "event: response.reasoning_summary_text.delta")
	require.Contains(t, rec.Body.String(), `"delta":"private reasoning"`)
	require.NotContains(t, rec.Body.String(), "event: response.output_text.delta")
	require.Contains(t, rec.Body.String(), `"status":"incomplete"`)
	terminalType, terminalPayload, ok := extractOpenAISSETerminalEvent(rec.Body.String())
	require.True(t, ok)
	require.Equal(t, "response.completed", terminalType)
	require.Equal(t, int64(1), gjson.GetBytes(terminalPayload, "response.output.#").Int())
	require.Equal(t, "reasoning", gjson.GetBytes(terminalPayload, "response.output.0.type").String())
	require.Contains(t, rec.Body.String(), "data: [DONE]")
}

func TestForwardResponses_DeepSeekRemoteCompactionV2ProducesOneOpaqueCompaction(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{
		"model":"deepseek-v4-flash",
		"reasoning":{"effort":"max"},
		"stream":true,
		"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],
		"input":[
			{"type":"message","role":"user","content":[{"type":"input_text","text":"remember STATE=preserved"}]},
			{"type":"additional_tools","tools":[{"type":"function","name":"exec_command","parameters":{"type":"object"}}]},
			{"type":"compaction_trigger"}
		]
	}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("x-codex-beta-features", "remote_compaction_v2")
	c.Set("api_key", &APIKey{ID: 501})

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_deepseek_compact_v2"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_compact","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","reasoning_content":"private compact reasoning","content":"<summary>STATE=preserved; continue the task.</summary>"},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150,"completion_tokens_details":{"reasoning_tokens":19}}}`,
		)),
	}}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Equal(t, "application/json", upstream.lastReq.Header.Get("Accept"))
	require.False(t, gjson.GetBytes(upstream.lastBody, "stream").Bool())
	require.False(t, gjson.GetBytes(upstream.lastBody, "stream_options").Exists())
	require.False(t, gjson.GetBytes(upstream.lastBody, "tools").Exists())
	require.NotContains(t, string(upstream.lastBody), "compaction_trigger")
	require.Contains(t, string(upstream.lastBody), "successor assistant")

	var doneItems []string
	var completed []byte
	forEachOpenAISSEDataPayload(rec.Body.String(), func(data []byte) {
		switch gjson.GetBytes(data, "type").String() {
		case "response.output_item.done":
			doneItems = append(doneItems, gjson.GetBytes(data, "item").Raw)
		case "response.completed":
			completed = append([]byte(nil), data...)
		}
	})
	require.Len(t, doneItems, 1)
	require.Equal(t, "compaction", gjson.Get(doneItems[0], "type").String())
	encrypted := gjson.Get(doneItems[0], "encrypted_content").String()
	require.True(t, strings.HasPrefix(encrypted, chatFallbackCompactionPrefix))
	require.False(t, gjson.Get(doneItems[0], "summary").Exists())
	require.NotContains(t, rec.Body.String(), "private compact reasoning")
	require.NotEmpty(t, completed)
	require.Equal(t, "response.compaction", gjson.GetBytes(completed, "response.object").String())
	require.Equal(t, "compaction", gjson.GetBytes(completed, "response.output.0.type").String())
	require.Equal(t, int64(1), gjson.GetBytes(completed, "response.output.#").Int())
	require.Equal(t, int64(19), gjson.GetBytes(completed, "response.usage.output_tokens_details.reasoning_tokens").Int())
	summary, err := svc.decryptChatFallbackCompaction(c, encrypted)
	require.NoError(t, err)
	require.Equal(t, "STATE=preserved; continue the task.", summary)
}

func TestForwardResponses_DeepSeekCompactionRoundTripRestoresSummary(t *testing.T) {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set("api_key", &APIKey{ID: 502})
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig()}
	encrypted, err := svc.encryptChatFallbackCompaction(c, "STATE=round-trip; next command is build")
	require.NoError(t, err)

	replayBody, err := json.Marshal(map[string]any{
		"model": "deepseek-v4-flash",
		"input": []any{
			map[string]any{"type": "compaction", "encrypted_content": encrypted},
			map[string]any{"type": "message", "role": "user", "content": "continue now"},
		},
		"stream": false,
	})
	require.NoError(t, err)
	rec := httptest.NewRecorder()
	c, _ = gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(replayBody))
	c.Set("api_key", &APIKey{ID: 502})
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_after_compact","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"AFTER_COMPACT_OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":2,"total_tokens":22}}`,
		)),
	}}
	svc.httpUpstream = upstream

	result, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), replayBody)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Contains(t, gjson.GetBytes(upstream.lastBody, "messages.0.content").String(), "<conversation_summary>")
	require.Contains(t, gjson.GetBytes(upstream.lastBody, "messages.0.content").String(), "STATE=round-trip")
	require.Equal(t, "continue now", gjson.GetBytes(upstream.lastBody, "messages.1.content").String())
	require.NotContains(t, string(upstream.lastBody), chatFallbackCompactionPrefix)
	require.Equal(t, "AFTER_COMPACT_OK", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
}

func TestForwardResponses_DeepSeekExplicitCompactReturnsCanonicalResource(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"input":[
			{"role":"user","content":"remember first user message"},
			{"type":"message","role":"assistant","content":[{"type":"output_text","text":"assistant history"}]},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"remember second user message"}]}
		],
		"stream":false
	}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses/compact", bytes.NewReader(body))
	c.Set("api_key", &APIKey{ID: 503})
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_explicit_compact","object":"chat.completion","model":"deepseek-v4-flash","choices":[{"index":0,"message":{"role":"assistant","content":"explicit summary"},"finish_reason":"stop"}],"usage":{"prompt_tokens":8,"completion_tokens":3,"total_tokens":11}}`,
		)),
	}}
	account := forceChatResponsesFallbackAccount()
	account.Credentials["compact_model_mapping"] = map[string]any{"deepseek-v4-flash": "deepseek-v4-flash-compact"}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.False(t, result.Stream)
	require.Equal(t, "deepseek-v4-flash-compact", gjson.GetBytes(upstream.lastBody, "model").String())
	require.Equal(t, "response.compaction", gjson.Get(rec.Body.String(), "object").String())
	require.Greater(t, gjson.Get(rec.Body.String(), "created_at").Int(), int64(0))
	require.Equal(t, int64(3), gjson.Get(rec.Body.String(), "output.#").Int())
	require.Equal(t, "message", gjson.Get(rec.Body.String(), "output.0.type").String())
	require.Equal(t, "user", gjson.Get(rec.Body.String(), "output.0.role").String())
	require.Equal(t, "completed", gjson.Get(rec.Body.String(), "output.0.status").String())
	require.True(t, strings.HasPrefix(gjson.Get(rec.Body.String(), "output.0.id").String(), "msg_"))
	require.Equal(t, "remember first user message", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
	require.Equal(t, "message", gjson.Get(rec.Body.String(), "output.1.type").String())
	require.Equal(t, "remember second user message", gjson.Get(rec.Body.String(), "output.1.content.0.text").String())
	require.Equal(t, "compaction", gjson.Get(rec.Body.String(), "output.2.type").String())
	require.True(t, strings.HasPrefix(gjson.Get(rec.Body.String(), "output.2.encrypted_content").String(), chatFallbackCompactionPrefix))
	require.Equal(t, int64(8), gjson.Get(rec.Body.String(), "usage.input_tokens").Int())
	require.Equal(t, int64(3), gjson.Get(rec.Body.String(), "usage.output_tokens").Int())
	require.Equal(t, int64(11), gjson.Get(rec.Body.String(), "usage.total_tokens").Int())
	require.False(t, gjson.Get(rec.Body.String(), "model").Exists())
	require.False(t, gjson.Get(rec.Body.String(), "status").Exists())
}

func TestForwardResponses_DeepSeekExplicitCompactAlwaysIncludesUsage(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-flash","input":"remember explicit compact","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses/compact", bytes.NewReader(body))
	c.Set("api_key", &APIKey{ID: 506})
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"chatcmpl_no_usage","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"summary without usage"},"finish_reason":"stop"}]}`,
		)),
	}}
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig(), httpUpstream: upstream}

	_, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.NoError(t, err)
	require.True(t, gjson.Get(rec.Body.String(), "usage").Exists())
	require.True(t, gjson.Get(rec.Body.String(), "usage.input_tokens").Exists())
	require.True(t, gjson.Get(rec.Body.String(), "usage.output_tokens").Exists())
	require.True(t, gjson.Get(rec.Body.String(), "usage.total_tokens").Exists())
	require.Zero(t, gjson.Get(rec.Body.String(), "usage.total_tokens").Int())
}

func TestForwardResponses_DeepSeekCompactionPrepareErrorAfterKeepaliveEmitsFailed(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":true,
		"input":[
			{"type":"compaction","encrypted_content":"unreadable"},
			{"type":"compaction_trigger"}
		]
	}`)
	c, rec := newCompactBridgeTestContext(t, true)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Set("api_key", &APIKey{ID: 507})
	stop := StartOpenAICompactSSEKeepalive(c, keepaliveTestInterval)
	defer stop()
	waitForKeepaliveBeats()
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig()}

	_, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.ErrorContains(t, err, "prepare chat fallback compact request")
	events := parseCompactBridgeSSE(t, stripKeepaliveComments(rec.Body.String()))
	require.Len(t, events, 1)
	require.Equal(t, "response.failed", events[0][0])
	require.Equal(t, "failed", gjson.Get(events[0][1], "response.status").String())
	require.Equal(t, "invalid_request_error", gjson.Get(events[0][1], "response.error.code").String())
	require.True(t, strings.HasPrefix(stripKeepaliveComments(rec.Body.String()), "event: response.failed\n"))
}

func TestForwardResponses_DeepSeekCompactionTooLargeAfterKeepaliveEmitsFailed(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":true,
		"input":[
			{"type":"message","role":"user","content":"remember compact state"},
			{"type":"compaction_trigger"}
		]
	}`)
	c, rec := newCompactBridgeTestContext(t, true)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Set("api_key", &APIKey{ID: 508})
	stop := StartOpenAICompactSSEKeepalive(c, keepaliveTestInterval)
	defer stop()
	waitForKeepaliveBeats()
	cfg := rawChatCompletionsTestConfig()
	cfg.Gateway.UpstreamResponseReadMaxBytes = 16
	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(strings.Repeat("x", 32))),
	}}
	svc := &OpenAIGatewayService{cfg: cfg, httpUpstream: upstream}

	_, err := svc.Forward(context.Background(), c, forceChatResponsesFallbackAccount(), body)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrUpstreamResponseBodyTooLarge)
	events := parseCompactBridgeSSE(t, stripKeepaliveComments(rec.Body.String()))
	require.Len(t, events, 1)
	require.Equal(t, "response.failed", events[0][0])
	require.Equal(t, "failed", gjson.Get(events[0][1], "response.status").String())
	require.Contains(t, gjson.Get(events[0][1], "response.error.message").String(), "too large")
	require.True(t, strings.HasPrefix(stripKeepaliveComments(rec.Body.String()), "event: response.failed\n"))
}

func TestPrepareChatFallbackResponsesBodyRejectsTamperedAndCrossKeyCompaction(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := &OpenAIGatewayService{cfg: rawChatCompletionsTestConfig()}
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	c.Set("api_key", &APIKey{ID: 504})
	encrypted, err := svc.encryptChatFallbackCompaction(c, "bound summary")
	require.NoError(t, err)

	buildBody := func(value string) []byte {
		body, marshalErr := json.Marshal(map[string]any{
			"model": "deepseek-v4-flash",
			"input": []any{map[string]any{"type": "compaction", "encrypted_content": value}},
		})
		require.NoError(t, marshalErr)
		return body
	}
	replacement := byte('A')
	if encrypted[len(encrypted)-1] == replacement {
		replacement = 'B'
	}
	tampered := encrypted[:len(encrypted)-1] + string(replacement)
	_, err = svc.prepareChatFallbackResponsesBody(c, buildBody(tampered), false)
	require.ErrorContains(t, err, "authenticate compact envelope")

	other, _ := gin.CreateTestContext(httptest.NewRecorder())
	other.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
	other.Set("api_key", &APIKey{ID: 505})
	_, err = svc.prepareChatFallbackResponsesBody(other, buildBody(encrypted), false)
	require.ErrorContains(t, err, "different API key")
}

func TestForwardResponses_AutoSupportedAccountStillUsesResponsesEndpoint(t *testing.T) {
	gin.SetMode(gin.TestMode)

	body := []byte(`{"model":"gpt-5.4","input":"hello","stream":false}`)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}, "x-request-id": []string{"rid_resp_native"}},
		Body: io.NopCloser(strings.NewReader(
			`{"id":"resp_native","object":"response","model":"gpt-5.4","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}],"status":"completed"}],"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}`,
		)),
	}}
	svc := &OpenAIGatewayService{
		cfg:          rawChatCompletionsTestConfig(),
		httpUpstream: upstream,
	}
	account := rawChatCompletionsTestAccount()
	account.Extra = map[string]any{
		openai_compat.ExtraKeyResponsesMode:      string(openai_compat.ResponsesSupportModeAuto),
		openai_compat.ExtraKeyResponsesSupported: true,
	}

	result, err := svc.Forward(context.Background(), c, account, body)
	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "http://upstream.example/v1/responses", upstream.lastReq.URL.String())
	require.True(t, gjson.GetBytes(upstream.lastBody, "input").Exists())
	require.False(t, gjson.GetBytes(upstream.lastBody, "messages").Exists())
	require.Equal(t, "ok", gjson.Get(rec.Body.String(), "output.0.content.0.text").String())
}

func forceChatResponsesFallbackAccount() *Account {
	account := rawChatCompletionsTestAccount()
	account.Extra = map[string]any{
		openai_compat.ExtraKeyResponsesMode: string(openai_compat.ResponsesSupportModeForceChatCompletions),
	}
	return account
}
