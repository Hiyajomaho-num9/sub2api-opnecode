//go:build unit

package service

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestReadCCUpstreamJSONResponse_TooLargeUsesCallerErrorFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		writeError compatErrorWriter
		assertBody func(*testing.T, string)
	}{
		{
			name:       "responses",
			writeError: writeOpenAIResponsesFallbackError,
			assertBody: func(t *testing.T, body string) {
				require.False(t, gjson.Get(body, "type").Exists())
				require.Equal(t, "upstream_error", gjson.Get(body, "error.type").String())
			},
		},
		{
			name:       "anthropic messages",
			writeError: writeAnthropicError,
			assertBody: func(t *testing.T, body string) {
				require.Equal(t, "error", gjson.Get(body, "type").String())
				require.Equal(t, "upstream_error", gjson.Get(body, "error.type").String())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &config.Config{}
			cfg.Gateway.UpstreamResponseReadMaxBytes = 4
			svc := &OpenAIGatewayService{cfg: cfg}
			recorder := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(recorder)
			resp := &http.Response{Body: io.NopCloser(strings.NewReader("too large"))}

			_, _, err := svc.readCCUpstreamJSONResponse(c, resp, tt.writeError)

			require.ErrorIs(t, err, ErrUpstreamResponseBodyTooLarge)
			require.Equal(t, http.StatusBadGateway, recorder.Code)
			tt.assertBody(t, recorder.Body.String())
		})
	}
}
