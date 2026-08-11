package service

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestOpenCodeGoReasoningSummaryCompatPromotesNonStreamingResponse(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-flash","stream":false,"reasoning":{"effort":"max","summary":"auto"},"input":"hello"}`)
	upstream := &httpUpstreamRecorder{responses: []*http.Response{
		newOpenAIRejectedFieldTestResponse(http.StatusOK, `{
			"id":"resp_reasoning","object":"response","status":"completed","model":"deepseek-v4-flash",
			"output":[
				{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private reasoning"}]},
				{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer","annotations":[]}]}
			],
			"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5,"input_tokens_details":{"cached_tokens":0}}
		}`),
	}}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "codex_vscode/test")

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(), c, newOpenCodeGoReasoningSummaryTestAccount(), body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.Equal(t, "summary_text", gjson.GetBytes(recorder.Body.Bytes(), "output.0.summary.0.type").String())
	require.Equal(t, "private reasoning", gjson.GetBytes(recorder.Body.Bytes(), "output.0.summary.0.text").String())
	require.Equal(t, "reasoning_text", gjson.GetBytes(recorder.Body.Bytes(), "output.0.content.0.type").String())
}

func TestOpenCodeGoReasoningSummaryCompatPromotesStreamingLifecycle(t *testing.T) {
	body := []byte(`{"model":"deepseek-v4-flash","stream":true,"reasoning":{"effort":"max","summary":"detailed"},"input":"hello"}`)
	stream := strings.Join([]string{
		`event: response.created`,
		`data: {"type":"response.created","response":{"id":"resp_reasoning","status":"in_progress","output":[]}}`,
		``,
		`event: response.output_item.added`,
		`data: {"type":"response.output_item.added","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[],"content":[]}}`,
		``,
		`event: response.content_part.added`,
		`data: {"type":"response.content_part.added","output_index":0,"content_index":0,"item_id":"rs_1","part":{"type":"reasoning_text","text":""}}`,
		``,
		`event: response.reasoning_text.delta`,
		`data: {"type":"response.reasoning_text.delta","output_index":0,"content_index":0,"item_id":"rs_1","delta":"private "}`,
		``,
		`event: response.reasoning_text.delta`,
		`data: {"type":"response.reasoning_text.delta","output_index":0,"content_index":0,"item_id":"rs_1","delta":"reasoning"}`,
		``,
		`event: response.reasoning_text.done`,
		`data: {"type":"response.reasoning_text.done","output_index":0,"content_index":0,"item_id":"rs_1","text":"private reasoning"}`,
		``,
		`event: response.content_part.done`,
		`data: {"type":"response.content_part.done","output_index":0,"content_index":0,"item_id":"rs_1","part":{"type":"reasoning_text","text":"private reasoning"}}`,
		``,
		`event: response.output_item.done`,
		`data: {"type":"response.output_item.done","output_index":0,"item":{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private reasoning"}]}}`,
		``,
		`event: response.completed`,
		`data: {"type":"response.completed","response":{"id":"resp_reasoning","status":"completed","model":"deepseek-v4-flash","output":[{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"private reasoning"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		``,
	}, "\n")
	upstream := &httpUpstreamRecorder{responses: []*http.Response{{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
		Body:       io.NopCloser(strings.NewReader(stream)),
	}}}
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/responses", bytes.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	c.Request.Header.Set("User-Agent", "codex_vscode/test")

	result, err := newOpenAIRejectedFieldTestService(upstream).Forward(
		context.Background(), c, newOpenCodeGoReasoningSummaryTestAccount(), body,
	)

	require.NoError(t, err)
	require.NotNil(t, result)
	require.True(t, result.Stream)
	frames := parseGrokProtocolSSEFrames(t, recorder.Body.String())
	requireGrokProtocolFrame(t, frames, "response.reasoning_summary_part.added", "part.type", "summary_text")
	requireGrokProtocolFrame(t, frames, "response.reasoning_summary_text.delta", "delta", "private ")
	requireGrokProtocolFrame(t, frames, "response.reasoning_summary_text.done", "text", "private reasoning")
	requireGrokProtocolFrame(t, frames, "response.reasoning_summary_part.done", "part.type", "summary_text")
	done := requireGrokProtocolFrame(t, frames, "response.output_item.done", "item.type", "reasoning")
	require.Equal(t, "summary_text", gjson.GetBytes(done.data, "item.summary.0.type").String())
	completed := requireGrokProtocolFrame(t, frames, "response.completed", "", "")
	require.Equal(t, "private reasoning", gjson.GetBytes(completed.data, "response.output.0.summary.0.text").String())
	require.NotContains(t, recorder.Body.String(), "event: response.reasoning_text.delta")
}

func TestOpenCodeGoReasoningSummaryCompatLeavesSummaryNoneUntouched(t *testing.T) {
	body := []byte(`{"reasoning":{"summary":"none"}}`)
	require.False(t, shouldPromoteOpenCodeGoReasoningSummary(newOpenCodeGoReasoningSummaryTestAccount(), body))
}

func TestOpenCodeGoReasoningSummaryCompatDoesNotAffectOtherUpstreams(t *testing.T) {
	account := newOpenCodeGoReasoningSummaryTestAccount()
	account.Credentials["base_url"] = "https://api.example.com/v1/responses"
	body := []byte(`{"reasoning":{"summary":"detailed"}}`)
	require.False(t, shouldPromoteOpenCodeGoReasoningSummary(account, body))
}

func newOpenCodeGoReasoningSummaryTestAccount() *Account {
	account := newOpenAIRejectedFieldTestAccount()
	account.Name = "deepseek"
	account.Credentials["base_url"] = "https://opencode.ai/zen/go/v1/responses"
	return account
}
