//go:build unit

package service

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/openai_compat"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenAIResponsesRejectedFieldRetryStateRejectsDuplicateBodyAndCap(t *testing.T) {
	initialBody := []byte(`{"model":"gpt-5.5"}`)
	state := newOpenAIResponsesRejectedFieldRetryState(initialBody)

	require.False(t, state.Allow(initialBody))
	for attempt := 0; attempt < maxOpenAIResponsesRejectedFieldRetries; attempt++ {
		nextBody := []byte(fmt.Sprintf(`{"model":"gpt-5.5","variant":%d}`, attempt))
		require.True(t, state.Allow(nextBody))
		require.False(t, state.Allow(nextBody))
	}
	require.False(t, state.Allow([]byte(`{"model":"gpt-5.5","variant":"overflow"}`)))
}

func TestNormalizeOpenAIResponsesRejectedFieldRetryBodyRejectsAmbiguousErrors(t *testing.T) {
	tests := []struct {
		name         string
		body         []byte
		responseBody []byte
	}{
		{
			name:         "namespace belongs to message",
			body:         []byte(`{"input":[{"type":"message","namespace":"keep"}]}`),
			responseBody: []byte(`{"error":{"code":"unknown_parameter","message":"Unknown parameter: 'input[0].namespace'.","param":"input[0].namespace"}}`),
		},
		{
			name:         "max output tokens only mentioned",
			body:         []byte(`{"max_output_tokens":4096}`),
			responseBody: []byte(`{"error":{"code":"invalid_request_error","message":"max_output_tokens must be positive","param":"max_output_tokens"}}`),
		},
		{
			name:         "structured param overrides namespace mention",
			body:         []byte(`{"input":[{"type":"function_call","namespace":"keep","arguments":"{}"}]}`),
			responseBody: []byte(`{"error":{"code":"unknown_parameter","message":"Unknown parameter: 'input[0].namespace'.","param":"tools"}}`),
		},
		{
			name:         "nested max output tokens param is not top level",
			body:         []byte(`{"max_output_tokens":4096,"input":[{"type":"message","content":{"max_output_tokens":"keep"}}]}`),
			responseBody: []byte(`{"error":{"code":"unknown_parameter","message":"Unknown parameter: input[0].content.max_output_tokens","param":"input[0].content.max_output_tokens"}}`),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			retryBody, _, changed, err := normalizeOpenAIResponsesRejectedFieldRetryBody(http.StatusBadRequest, tt.body, tt.responseBody)
			require.NoError(t, err)
			require.False(t, changed)
			require.Nil(t, retryBody)
		})
	}
}

func TestNormalizeOpenAIResponsesRejectedFieldRetryBodyFindsNamespacePathInMessage(t *testing.T) {
	body := []byte(`{"input":[{"type":"function_call","namespace":"keep","arguments":"{}"},{"type":"function_call","namespace":"remove","arguments":"{}"}]}`)
	responseBody := []byte(`{"error":{"code":"unknown_parameter","message":"input[0] was accepted; Unknown parameter: 'input[1].namespace'."}}`)

	retryBody, _, changed, err := normalizeOpenAIResponsesRejectedFieldRetryBody(http.StatusBadRequest, body, responseBody)

	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, "keep", gjson.GetBytes(retryBody, "input.0.namespace").String())
	require.False(t, gjson.GetBytes(retryBody, "input.1.namespace").Exists())
}

func TestNormalizeOpenAIResponsesRejectedFieldRetryBodyBindsNamespacePathToRejectionPhrase(t *testing.T) {
	body := []byte(`{"input":[{"type":"function_call","namespace":"keep","arguments":"{}"},{"type":"function_call","namespace":"remove","arguments":"{}"}]}`)
	responseBody := []byte(`{"error":{"code":"unknown_parameter","message":"input[0].namespace is supported; Unknown parameter: input[1].namespace."}}`)

	retryBody, _, changed, err := normalizeOpenAIResponsesRejectedFieldRetryBody(http.StatusBadRequest, body, responseBody)

	require.NoError(t, err)
	require.True(t, changed)
	require.Equal(t, "keep", gjson.GetBytes(retryBody, "input.0.namespace").String())
	require.False(t, gjson.GetBytes(retryBody, "input.1.namespace").Exists())
}

func TestNormalizeOpenAIResponsesRejectedFieldRetryBodyDoesNotTreatMaxOutputTokensSuggestionAsRejection(t *testing.T) {
	body := []byte(`{"max_tokens":4096,"max_output_tokens":2048}`)
	responseBody := []byte(`{"error":{"code":"unknown_parameter","message":"Unknown parameter: max_tokens. Use max_output_tokens instead."}}`)

	retryBody, _, changed, err := normalizeOpenAIResponsesRejectedFieldRetryBody(http.StatusBadRequest, body, responseBody)

	require.NoError(t, err)
	require.False(t, changed)
	require.Nil(t, retryBody)
}

func TestNormalizeOpenAIResponsesRejectedFieldRetryBodyBindsMaxOutputTokensToRejectionPhrase(t *testing.T) {
	body := []byte(`{"max_output_tokens":2048}`)
	responseBody := []byte(`{"error":{"code":"unsupported_parameter","message":"Unsupported parameter: max_output_tokens."}}`)

	retryBody, _, changed, err := normalizeOpenAIResponsesRejectedFieldRetryBody(http.StatusBadRequest, body, responseBody)

	require.NoError(t, err)
	require.True(t, changed)
	require.False(t, gjson.GetBytes(retryBody, "max_output_tokens").Exists())
}

func TestOpenAIGatewayService_APIKeyStripsAllIndexedNamespacesBeforeFirstForward(t *testing.T) {
	body := []byte(`{"model":"gpt-5.5","stream":false,"input":[{"type":"function_call","name":"first","namespace":"remove-first","arguments":"{}"},{"type":"custom_tool_call","name":"second","namespace":"remove-second","input":"{}"}]}`)
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusOK, `{"output":[],"usage":{"input_tokens":1,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}`),
	}}

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(),
		newOpenAIRejectedFieldTestContext(body),
		newOpenAIRejectedFieldTestAccount(),
		body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.bodies, 1)
	require.False(t, gjson.GetBytes(upstream.bodies[0], "input.0.namespace").Exists())
	require.False(t, gjson.GetBytes(upstream.bodies[0], "input.1.namespace").Exists())
}

func TestOpenAIGatewayService_OpenAIHTTPStripsInputNamespacesBeforeFirstForward(t *testing.T) {
	accounts := []struct {
		name    string
		account *Account
	}{
		{name: "oauth", account: newOpenAIOAuthNamespaceTestAccount()},
		{name: "apikey", account: newOpenAIRejectedFieldTestAccount()},
	}
	for _, tt := range accounts {
		for _, path := range []string{"/v1/responses", "/v1/responses/compact"} {
			t.Run(tt.name+path, func(t *testing.T) {
				body := []byte(`{"model":"gpt-5.5","stream":false,"instructions":"test","input":[{"type":"message","role":"user","namespace":"remove","content":[{"type":"input_text","text":"hello","namespace":"nested-keep"}]}]}`)
				upstream := &httpUpstreamRecorder{responses: []*http.Response{
					newOpenAIRejectedFieldTestResponse(http.StatusOK, `{"id":"resp_namespace_ok","output":[],"usage":{"input_tokens":1,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}`),
				}}
				c := newOpenAIRejectedFieldTestContext(body)
				c.Request.URL.Path = path

				result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
					context.Background(),
					c,
					tt.account,
					body,
				)

				require.NoError(t, err)
				require.NotNil(t, result)
				require.Len(t, upstream.bodies, 1, "namespace must be removed before the first upstream request")
				require.False(t, gjson.GetBytes(upstream.bodies[0], "input.0.namespace").Exists())
				require.Equal(t, "nested-keep", gjson.GetBytes(upstream.bodies[0], "input.0.content.0.namespace").String())
			})
		}
	}
}

func TestOpenAIGatewayService_RetriesExplicitMaxOutputTokensRejection(t *testing.T) {
	body := []byte(`{"model":"gpt-5.5","stream":false,"max_output_tokens":4096,"input":[{"type":"message","role":"user","content":{"max_output_tokens":"keep"}}]}`)
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusBadRequest, `{"error":{"code":"unsupported_parameter","message":"Unsupported parameter: max_output_tokens","param":"max_output_tokens","type":"invalid_request_error"}}`),
		newOpenAIRejectedFieldTestResponse(http.StatusOK, `{"output":[],"usage":{"input_tokens":1,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}`),
	}}

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(),
		newOpenAIRejectedFieldTestContext(body),
		newOpenAIRejectedFieldTestAccount(),
		body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.bodies, 2)
	require.Equal(t, int64(4096), gjson.GetBytes(upstream.bodies[0], "max_output_tokens").Int())
	require.False(t, gjson.GetBytes(upstream.bodies[1], "max_output_tokens").Exists())
	require.Equal(t, "keep", gjson.GetBytes(upstream.bodies[1], "input.0.content.max_output_tokens").String())
}

func TestOpenAIGatewayService_RetriesUnsupportedCustomToolAsFunctionAndRestoresResponse(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash",
		"stream":false,
		"tools":[{"type":"custom","name":"exec","description":"Run a command","format":{"type":"text"}}],
		"tool_choice":{"type":"custom","name":"exec"},
		"input":[
			{"type":"custom_tool_call","call_id":"call_previous","name":"exec","input":"pwd"},
			{"type":"custom_tool_call_output","call_id":"call_previous","output":{"text":"/workspace"}},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`)
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusBadRequest, `{"error":{"type":"invalid_request_error","message":"Unsupported custom tool: 'exec'. Only 'apply_patch' is supported."}}`),
		newOpenAIRejectedFieldTestResponse(http.StatusOK, `{
			"id":"resp_custom_tool","object":"response","model":"deepseek-v4-flash","status":"completed",
			"output":[{"type":"function_call","id":"item_exec","call_id":"call_exec","name":"exec","arguments":"{\"input\":\"ls -la\"}"}],
			"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13,"input_tokens_details":{"cached_tokens":0}}
		}`),
	}}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "codex_vscode/test")

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(), c, newOpenAIRejectedFieldTestAccount(), body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.bodies, 2)
	require.Equal(t, "custom", gjson.GetBytes(upstream.bodies[0], "tools.0.type").String())
	require.Equal(t, "function", gjson.GetBytes(upstream.bodies[1], "tools.0.type").String())
	require.True(t, gjson.GetBytes(upstream.bodies[1], "tools.0.parameters.properties.input").Exists())
	require.False(t, gjson.GetBytes(upstream.bodies[1], "tools.0.format").Exists())
	require.Equal(t, "function", gjson.GetBytes(upstream.bodies[1], "tool_choice.type").String())
	require.Equal(t, "function_call", gjson.GetBytes(upstream.bodies[1], "input.0.type").String())
	require.JSONEq(t, `{"input":"pwd"}`, gjson.GetBytes(upstream.bodies[1], "input.0.arguments").String())
	require.Equal(t, "function_call_output", gjson.GetBytes(upstream.bodies[1], "input.1.type").String())
	require.JSONEq(t, `{"text":"/workspace"}`, gjson.GetBytes(upstream.bodies[1], "input.1.output").String())

	response := recorder.Body.Bytes()
	require.Equal(t, "custom_tool_call", gjson.GetBytes(response, "output.0.type").String())
	require.Equal(t, "ls -la", gjson.GetBytes(response, "output.0.input").String())
	require.False(t, gjson.GetBytes(response, "output.0.arguments").Exists())
}

func TestOpenAIGatewayService_RetriesUnsupportedCustomToolAndRestoresStreamingLifecycle(t *testing.T) {
	body := []byte(`{
		"model":"deepseek-v4-flash","stream":true,
		"tools":[{"type":"custom","name":"exec","description":"Run a command","format":{"type":"text"}}],
		"input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"list files"}]}]
	}`)
	stream := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","sequence_number":10,"response":{"id":"resp_custom_stream","model":"deepseek-v4-flash"}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","sequence_number":11,"output_index":0,"item":{"type":"function_call","id":"item_exec","call_id":"call_exec","name":"exec","arguments":"","status":"in_progress"}}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","sequence_number":12,"output_index":0,"item_id":"item_exec","delta":"{\"input\":\"ls"}`,
		``,
		`event: response.function_call_arguments.delta`,
		`data: {"type":"response.function_call_arguments.delta","sequence_number":13,"output_index":0,"item_id":"item_exec","delta":" -la\"}"}`,
		``,
		`event: response.function_call_arguments.done`,
		`data: {"type":"response.function_call_arguments.done","sequence_number":14,"output_index":0,"item_id":"item_exec","call_id":"call_exec","name":"exec","arguments":"{\"input\":\"ls -la\"}"}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","sequence_number":15,"output_index":0,"item":{"type":"function_call","id":"item_exec","call_id":"call_exec","name":"exec","arguments":"{\"input\":\"ls -la\"}","status":"completed"}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","sequence_number":16,"response":{"id":"resp_custom_stream","object":"response","model":"deepseek-v4-flash","status":"completed","output":[{"type":"function_call","id":"item_exec","call_id":"call_exec","name":"exec","arguments":"{\"input\":\"ls -la\"}"}],"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}}}`,
		``,
	}, "\n")
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusBadRequest, `{"error":{"type":"invalid_request_error","message":"Unsupported custom tool: 'exec'. Only 'apply_patch' is supported."}}`),
		{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(stream)),
		},
	}}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "codex_vscode/test")

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(), c, newOpenAIRejectedFieldTestAccount(), body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	require.Len(t, upstream.bodies, 2)
	require.Equal(t, "function", gjson.GetBytes(upstream.bodies[1], "tools.0.type").String())

	frames := parseGrokProtocolSSEFrames(t, recorder.Body.String())
	customAdded := requireGrokProtocolFrame(t, frames, "response.output_item.added", "item.type", "custom_tool_call")
	require.Equal(t, "exec", gjson.GetBytes(customAdded.data, "item.name").String())
	customInputDone := requireGrokProtocolFrame(t, frames, "response.custom_tool_call_input.done", "", "")
	require.Equal(t, "ls -la", gjson.GetBytes(customInputDone.data, "input").String())
	customDone := requireGrokProtocolFrame(t, frames, "response.output_item.done", "item.type", "custom_tool_call")
	require.Equal(t, "ls -la", gjson.GetBytes(customDone.data, "item.input").String())
	completed := requireGrokProtocolFrame(t, frames, "response.completed", "", "")
	require.Equal(t, "custom_tool_call", gjson.GetBytes(completed.data, "response.output.0.type").String())
	for _, frame := range frames {
		if gjson.GetBytes(frame.data, "item_id").String() == "item_exec" {
			require.NotContains(t, frame.event, "function_call_arguments")
		}
	}
}

func TestIsOpenAIResponsesUnsupportedCustomToolError(t *testing.T) {
	require.True(t, isOpenAIResponsesUnsupportedCustomToolError(http.StatusBadRequest, []byte(`{"error":{"message":"Unsupported custom tool: 'exec'."}}`)))
	require.False(t, isOpenAIResponsesUnsupportedCustomToolError(http.StatusUnprocessableEntity, []byte(`{"error":{"message":"Unsupported custom tool: 'exec'."}}`)))
	require.False(t, isOpenAIResponsesUnsupportedCustomToolError(http.StatusBadRequest, []byte(`{"error":{"message":"Unsupported parameter: tools."}}`)))
}

func TestOpenAIGatewayService_ComposesProactiveNamespaceStripWithRejectedFieldRetry(t *testing.T) {
	body := []byte(`{"model":"gpt-5.5","stream":false,"max_output_tokens":2048,"input":[{"type":"function_call","name":"first","namespace":"remove-first","arguments":"{}"},{"type":"custom_tool_call","name":"second","namespace":"remove-second","input":"{}"}]}`)
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusBadRequest, `{"error":{"code":"unsupported_parameter","message":"Unsupported parameter: max_output_tokens","param":"max_output_tokens"}}`),
		newOpenAIRejectedFieldTestResponse(http.StatusOK, `{"output":[],"usage":{"input_tokens":1,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}`),
	}}

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(),
		newOpenAIRejectedFieldTestContext(body),
		newOpenAIRejectedFieldTestAccount(),
		body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Len(t, upstream.bodies, 2)
	for _, forwardedBody := range upstream.bodies {
		require.False(t, gjson.GetBytes(forwardedBody, "input.0.namespace").Exists())
		require.False(t, gjson.GetBytes(forwardedBody, "input.1.namespace").Exists())
	}
	require.Equal(t, int64(2048), gjson.GetBytes(upstream.bodies[0], "max_output_tokens").Int())
	require.False(t, gjson.GetBytes(upstream.bodies[1], "max_output_tokens").Exists())
}

func newOpenAIRejectedFieldTestService(upstream *httpUpstreamRecorder) *OpenAIGatewayService {
	return &OpenAIGatewayService{
		cfg: &config.Config{Security: config.SecurityConfig{
			URLAllowlist: config.URLAllowlistConfig{Enabled: false},
		}},
		httpUpstream: upstream,
	}
}

func newOpenAIRejectedFieldTestContext(body []byte) *gin.Context {
	gin.SetMode(gin.TestMode)
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "curl/8.0")
	return c
}

func newOpenAIRejectedFieldTestAccount() *Account {
	return &Account{
		ID:          5107,
		Name:        "responses-compatible",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeAPIKey,
		Concurrency: 1,
		Credentials: map[string]any{
			"api_key":  "sk-test",
			"base_url": "https://compat.example",
		},
		Extra: map[string]any{
			openai_compat.ExtraKeyResponsesMode:      string(openai_compat.ResponsesSupportModeAuto),
			openai_compat.ExtraKeyResponsesSupported: true,
		},
		Status:      StatusActive,
		Schedulable: true,
	}
}

func newOpenAIOAuthNamespaceTestAccount() *Account {
	return &Account{
		ID:          5108,
		Name:        "openai-oauth-namespace",
		Platform:    PlatformOpenAI,
		Type:        AccountTypeOAuth,
		Concurrency: 1,
		Credentials: map[string]any{
			"access_token":       "oauth-token",
			"chatgpt_account_id": "chatgpt-account",
		},
		Status:      StatusActive,
		Schedulable: true,
	}
}

func newOpenAIRejectedFieldTestResponse(status int, body string) *http.Response {
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body:       io.NopCloser(strings.NewReader(body)),
	}
}
