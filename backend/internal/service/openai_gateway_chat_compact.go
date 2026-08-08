package service

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const chatFallbackCompactionPrefix = "sub2api.compaction.v1."

type chatFallbackCompactionEnvelope struct {
	Version  int    `json:"version"`
	APIKeyID int64  `json:"api_key_id"`
	Summary  string `json:"summary"`
}

// chatFallbackCompactResource is the standalone /responses/compact wire
// shape. It intentionally differs from a normal Responses response: the
// CompactResource schema requires created_at and usage and carries the
// canonical retained user-message window before the compaction item.
type chatFallbackCompactResource struct {
	ID        string                    `json:"id"`
	Object    string                    `json:"object"`
	CreatedAt int64                     `json:"created_at"`
	Output    []json.RawMessage         `json:"output"`
	Usage     *apicompat.ResponsesUsage `json:"usage"`
}

// prepareChatFallbackResponsesBody makes OpenAI compaction items usable by a
// Chat Completions-only upstream. Adapter-owned opaque payloads are restored as
// conversation summaries on ordinary follow-up turns. A compact request also
// drops client tools and appends the same summary instruction used by the Grok
// compatibility path.
func (s *OpenAIGatewayService) prepareChatFallbackResponsesBody(c *gin.Context, body []byte, compactRequest bool) ([]byte, error) {
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode responses request: %w", err)
	}

	input, err := normalizeChatFallbackInput(payload["input"])
	if err != nil {
		return nil, err
	}
	rewritten := make([]any, 0, len(input)+1)
	changed := false
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok {
			rewritten = append(rewritten, raw)
			continue
		}
		switch strings.TrimSpace(stringValue(item["type"])) {
		case "compaction", "compaction_summary":
			changed = true
			summary, err := s.chatFallbackCompactionSummary(c, item)
			if err != nil {
				return nil, err
			}
			rewritten = append(rewritten, chatFallbackConversationSummaryMessage(summary))
		case "compaction_trigger":
			// The trigger is a Responses control item, not conversation content.
			changed = true
		case "additional_tools":
			if compactRequest {
				changed = true
				continue
			}
			rewritten = append(rewritten, raw)
		default:
			rewritten = append(rewritten, raw)
		}
	}

	if compactRequest {
		rewritten = append(rewritten, chatFallbackSummaryMessage(grokCompactSummaryPrompt))
		delete(payload, "tools")
		delete(payload, "tool_choice")
		delete(payload, "parallel_tool_calls")
		delete(payload, "text")
		changed = true
	}
	if !changed {
		return body, nil
	}
	payload["input"] = rewritten
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("encode responses request: %w", err)
	}
	return encoded, nil
}

func normalizeChatFallbackInput(value any) ([]any, error) {
	switch input := value.(type) {
	case nil:
		return []any{}, nil
	case []any:
		return input, nil
	case string:
		return []any{chatFallbackSummaryMessage(input)}, nil
	case map[string]any:
		return []any{input}, nil
	default:
		return nil, fmt.Errorf("responses input must be a string, object, or array")
	}
}

func chatFallbackSummaryMessage(summary string) map[string]any {
	return map[string]any{
		"type": "message",
		"role": "user",
		"content": []any{map[string]any{
			"type": "input_text",
			"text": summary,
		}},
	}
}

func chatFallbackConversationSummaryMessage(summary string) map[string]any {
	return chatFallbackSummaryMessage("<conversation_summary>\n" + strings.TrimSpace(summary) + "\n</conversation_summary>")
}

func (s *OpenAIGatewayService) chatFallbackCompactionSummary(c *gin.Context, item map[string]any) (string, error) {
	encoded := strings.TrimSpace(stringValue(item["encrypted_content"]))
	if strings.HasPrefix(encoded, chatFallbackCompactionPrefix) {
		return s.decryptChatFallbackCompaction(c, encoded)
	}
	// Grok's local compact adapter includes a visible summary alongside its
	// provider-owned encrypted state. It is the only portable part when a later
	// request fails over to a Chat-only account.
	if summary := compactSummaryText(item["summary"]); summary != "" {
		return summary, nil
	}
	return "", fmt.Errorf("compaction item is not readable by the Chat Completions fallback")
}

func (s *OpenAIGatewayService) buildChatFallbackCompactionResponse(
	c *gin.Context,
	compactSourceBody []byte,
	explicitCompactRequest bool,
	originalModel string,
	object string,
	response *apicompat.ChatCompletionsResponse,
) ([]byte, error) {
	if response == nil {
		return nil, fmt.Errorf("compact completion is nil")
	}
	summary := ""
	for _, choice := range response.Choices {
		if text := chatFallbackMessageText(choice.Message.Content); text != "" {
			if choice.FinishReason != "" && choice.FinishReason != "stop" {
				return nil, fmt.Errorf("compact completion ended with finish_reason %q", choice.FinishReason)
			}
			if len(choice.Message.ToolCalls) > 0 {
				return nil, fmt.Errorf("compact completion returned tool calls")
			}
			summary = text
			break
		}
	}
	summary = trimChatFallbackSummaryEnvelope(summary)
	if summary == "" {
		return nil, fmt.Errorf("compact completion contained no visible summary")
	}
	if object == "" {
		object = "response.compaction"
	}
	encryptedContent, err := s.encryptChatFallbackCompaction(c, summary)
	if err != nil {
		return nil, err
	}

	compactItem := apicompat.ResponsesOutput{
		Type:             "compaction",
		ID:               "cmp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		Status:           "completed",
		EncryptedContent: encryptedContent,
	}
	usage := apicompat.ChatUsageToResponsesUsage(response.Usage)
	if explicitCompactRequest {
		retained, err := chatFallbackRetainedUserMessages(compactSourceBody)
		if err != nil {
			return nil, err
		}
		compactJSON, err := json.Marshal(compactItem)
		if err != nil {
			return nil, fmt.Errorf("encode compact output item: %w", err)
		}
		output := append(retained, compactJSON)
		if usage == nil {
			// CompactResource requires usage even when an OpenAI-compatible Chat
			// upstream omitted accounting fields. A zero-valued canonical usage
			// object is safer than emitting null or dropping the required field.
			usage = &apicompat.ResponsesUsage{}
		}
		result := chatFallbackCompactResource{
			ID:        "resp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
			Object:    object,
			CreatedAt: time.Now().Unix(),
			Output:    output,
			Usage:     usage,
		}
		encoded, err := json.Marshal(result)
		if err != nil {
			return nil, fmt.Errorf("encode standalone compact response: %w", err)
		}
		return encoded, nil
	}

	result := apicompat.ResponsesResponse{
		ID:     "resp_" + strings.ReplaceAll(uuid.NewString(), "-", ""),
		Object: object,
		Model:  originalModel,
		Status: "completed",
		Output: []apicompat.ResponsesOutput{compactItem},
		Usage:  usage,
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("encode compact response: %w", err)
	}
	return encoded, nil
}

// chatFallbackRetainedUserMessages builds the canonical retained portion of a
// standalone compact response. OpenAI compact output keeps all user messages
// and appends one opaque compaction item; assistant/tool/reasoning items are
// represented by the encrypted summary and must not be duplicated here.
func chatFallbackRetainedUserMessages(body []byte) ([]json.RawMessage, error) {
	decoder := json.NewDecoder(strings.NewReader(string(body)))
	decoder.UseNumber()
	var payload map[string]any
	if err := decoder.Decode(&payload); err != nil {
		return nil, fmt.Errorf("decode standalone compact request: %w", err)
	}
	input, err := normalizeChatFallbackInput(payload["input"])
	if err != nil {
		return nil, err
	}

	retained := make([]json.RawMessage, 0, len(input))
	for _, raw := range input {
		item, ok := raw.(map[string]any)
		if !ok || !strings.EqualFold(strings.TrimSpace(stringValue(item["role"])), "user") {
			continue
		}
		itemType := strings.TrimSpace(stringValue(item["type"]))
		if itemType != "" && itemType != "message" {
			continue
		}

		message := make(map[string]any, len(item)+4)
		for key, value := range item {
			message[key] = value
		}
		message["type"] = "message"
		message["role"] = "user"
		message["status"] = "completed"
		if strings.TrimSpace(stringValue(message["id"])) == "" {
			message["id"] = "msg_" + strings.ReplaceAll(uuid.NewString(), "-", "")
		}
		switch content := message["content"].(type) {
		case nil:
			message["content"] = []any{}
		case string:
			message["content"] = []any{map[string]any{"type": "input_text", "text": content}}
		case []any:
			// Already in canonical multi-part input form; retain images/files and
			// other provider-compatible user content without lossy re-encoding.
		default:
			return nil, fmt.Errorf("standalone compact user message content must be a string or array")
		}
		encoded, err := json.Marshal(message)
		if err != nil {
			return nil, fmt.Errorf("encode retained compact user message: %w", err)
		}
		retained = append(retained, encoded)
	}
	return retained, nil
}

func (s *OpenAIGatewayService) encryptChatFallbackCompaction(c *gin.Context, summary string) (string, error) {
	gcm, err := s.chatFallbackCompactionGCM()
	if err != nil {
		return "", err
	}
	payload, err := json.Marshal(chatFallbackCompactionEnvelope{
		Version:  1,
		APIKeyID: getAPIKeyIDFromContext(c),
		Summary:  summary,
	})
	if err != nil {
		return "", fmt.Errorf("encode compact envelope: %w", err)
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return "", fmt.Errorf("generate compact nonce: %w", err)
	}
	sealed := gcm.Seal(nil, nonce, payload, []byte(chatFallbackCompactionPrefix))
	wire := append(nonce, sealed...)
	return chatFallbackCompactionPrefix + base64.RawURLEncoding.EncodeToString(wire), nil
}

func (s *OpenAIGatewayService) decryptChatFallbackCompaction(c *gin.Context, encoded string) (string, error) {
	gcm, err := s.chatFallbackCompactionGCM()
	if err != nil {
		return "", err
	}
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(encoded, chatFallbackCompactionPrefix))
	if err != nil {
		return "", fmt.Errorf("decode compact envelope: %w", err)
	}
	if len(raw) < gcm.NonceSize()+gcm.Overhead() {
		return "", fmt.Errorf("compact envelope is truncated")
	}
	nonce, ciphertext := raw[:gcm.NonceSize()], raw[gcm.NonceSize():]
	plaintext, err := gcm.Open(nil, nonce, ciphertext, []byte(chatFallbackCompactionPrefix))
	if err != nil {
		return "", fmt.Errorf("authenticate compact envelope: %w", err)
	}
	var envelope chatFallbackCompactionEnvelope
	if err := json.Unmarshal(plaintext, &envelope); err != nil {
		return "", fmt.Errorf("decode compact envelope payload: %w", err)
	}
	if envelope.Version != 1 {
		return "", fmt.Errorf("unsupported compact envelope version %d", envelope.Version)
	}
	if envelope.APIKeyID != getAPIKeyIDFromContext(c) {
		return "", fmt.Errorf("compact envelope belongs to a different API key")
	}
	summary := strings.TrimSpace(envelope.Summary)
	if summary == "" {
		return "", fmt.Errorf("compact envelope contains an empty summary")
	}
	return summary, nil
}

func (s *OpenAIGatewayService) chatFallbackCompactionGCM() (cipher.AEAD, error) {
	if s == nil || s.cfg == nil || strings.TrimSpace(s.cfg.JWT.Secret) == "" {
		return nil, fmt.Errorf("JWT secret is required for local compact encryption")
	}
	key := sha256.Sum256([]byte("sub2api/openai-compact/v1\x00" + s.cfg.JWT.Secret))
	block, err := aes.NewCipher(key[:])
	if err != nil {
		return nil, fmt.Errorf("create compact cipher: %w", err)
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, fmt.Errorf("create compact AEAD: %w", err)
	}
	return gcm, nil
}

func chatFallbackMessageText(raw json.RawMessage) string {
	raw = json.RawMessage(strings.TrimSpace(string(raw)))
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return strings.TrimSpace(text)
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if text := strings.TrimSpace(part.Text); text != "" {
			texts = append(texts, text)
		}
	}
	return strings.Join(texts, "\n")
}

func trimChatFallbackSummaryEnvelope(summary string) string {
	summary = strings.TrimSpace(summary)
	if strings.HasPrefix(summary, "<summary>") && strings.HasSuffix(summary, "</summary>") {
		summary = strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(summary, "<summary>"), "</summary>"))
	}
	return summary
}
