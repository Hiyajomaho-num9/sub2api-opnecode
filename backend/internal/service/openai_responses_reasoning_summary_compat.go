package service

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const openCodeGoReasoningSummaryCompatContextKey = "opencode_go_reasoning_summary_compat"
const openCodeGoReasoningContentPrefix = "sub2api/opencode-go-reasoning/v1:"

func configureOpenCodeGoReasoningSummaryCompat(c *gin.Context, account *Account, body []byte) {
	if c == nil {
		return
	}
	c.Set(openCodeGoReasoningSummaryCompatContextKey, shouldPromoteOpenCodeGoReasoningSummary(account, body))
}

func openCodeGoReasoningSummaryCompatEnabled(c *gin.Context) bool {
	if c == nil {
		return false
	}
	value, exists := c.Get(openCodeGoReasoningSummaryCompatContextKey)
	enabled, ok := value.(bool)
	return exists && ok && enabled
}

func shouldPromoteOpenCodeGoReasoningSummary(account *Account, body []byte) bool {
	if !isOpenCodeGoResponsesAccount(account) {
		return false
	}
	summary := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "reasoning.summary").String()))
	return summary != "" && summary != "none"
}

func isOpenCodeGoResponsesAccount(account *Account) bool {
	if account == nil || account.Platform != PlatformOpenAI || account.Type != AccountTypeAPIKey {
		return false
	}
	baseURL := strings.TrimSpace(account.GetCredential("base_url"))
	parsed, err := url.Parse(baseURL)
	if err != nil || !strings.EqualFold(parsed.Hostname(), "opencode.ai") {
		return false
	}
	path := strings.ToLower(strings.TrimRight(parsed.Path, "/"))
	return strings.HasPrefix(path, "/zen/go/") || path == "/zen/go"
}

func restoreOpenCodeGoReasoningContentRequest(body []byte) ([]byte, bool, error) {
	if len(body) == 0 || !json.Valid(body) {
		return body, false, nil
	}

	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var root map[string]any
	if err := decoder.Decode(&root); err != nil {
		return body, false, err
	}
	if !restoreOpenCodeGoReasoningInput(root["input"]) {
		return body, false, nil
	}

	restored, err := json.Marshal(root)
	if err != nil {
		return body, false, err
	}
	return restored, true, nil
}

func restoreOpenCodeGoReasoningInput(raw any) bool {
	switch input := raw.(type) {
	case []any:
		changed := false
		for _, rawItem := range input {
			if item, ok := rawItem.(map[string]any); ok && restoreOpenCodeGoReasoningItem(item) {
				changed = true
			}
		}
		return changed
	case map[string]any:
		return restoreOpenCodeGoReasoningItem(input)
	default:
		return false
	}
}

func restoreOpenCodeGoReasoningItem(item map[string]any) bool {
	if strings.TrimSpace(fmt.Sprint(item["type"])) != "reasoning" {
		return false
	}
	encrypted, _ := item["encrypted_content"].(string)
	encoded := strings.TrimPrefix(encrypted, openCodeGoReasoningContentPrefix)
	if encoded == encrypted || encoded == "" {
		return false
	}
	rawContent, err := base64.RawStdEncoding.DecodeString(encoded)
	if err != nil {
		return false
	}

	decoder := json.NewDecoder(bytes.NewReader(rawContent))
	decoder.UseNumber()
	var content []any
	if err := decoder.Decode(&content); err != nil || len(content) == 0 {
		return false
	}
	item["content"] = content
	item["summary"] = []any{}
	delete(item, "encrypted_content")
	return true
}

func promoteOpenCodeGoReasoningSummaryPayload(payload []byte) ([]byte, bool, error) {
	if len(payload) == 0 || !json.Valid(payload) {
		return payload, false, nil
	}

	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	var root map[string]any
	if err := decoder.Decode(&root); err != nil {
		return payload, false, err
	}

	changed := false
	eventType, _ := root["type"].(string)
	switch eventType {
	case "response.content_part.added":
		if promoteOpenCodeGoReasoningSummaryPartEvent(root, "response.reasoning_summary_part.added") {
			changed = true
		}
	case "response.content_part.done":
		if promoteOpenCodeGoReasoningSummaryPartEvent(root, "response.reasoning_summary_part.done") {
			changed = true
		}
	case "response.reasoning_text.delta":
		root["type"] = "response.reasoning_summary_text.delta"
		promoteOpenCodeGoReasoningSummaryIndex(root)
		changed = true
	case "response.reasoning_text.done":
		root["type"] = "response.reasoning_summary_text.done"
		promoteOpenCodeGoReasoningSummaryIndex(root)
		changed = true
	case "response.output_item.done":
		if item, ok := root["item"].(map[string]any); ok && promoteOpenCodeGoReasoningItem(item) {
			changed = true
		}
	case "response.completed":
		if response, ok := root["response"].(map[string]any); ok && promoteOpenCodeGoReasoningOutput(response["output"]) {
			changed = true
		}
	}

	// Non-streaming Responses payloads do not carry an event type.
	if promoteOpenCodeGoReasoningOutput(root["output"]) {
		changed = true
	}
	if !changed {
		return payload, false, nil
	}

	updated, err := json.Marshal(root)
	if err != nil {
		return payload, false, err
	}
	return updated, true, nil
}

func promoteOpenCodeGoReasoningSummaryPartEvent(event map[string]any, promotedType string) bool {
	part, ok := event["part"].(map[string]any)
	if !ok || strings.TrimSpace(fmt.Sprint(part["type"])) != "reasoning_text" {
		return false
	}
	event["type"] = promotedType
	part["type"] = "summary_text"
	promoteOpenCodeGoReasoningSummaryIndex(event)
	return true
}

func promoteOpenCodeGoReasoningSummaryIndex(event map[string]any) {
	if index, exists := event["content_index"]; exists {
		event["summary_index"] = index
		delete(event, "content_index")
		return
	}
	if _, exists := event["summary_index"]; !exists {
		event["summary_index"] = 0
	}
}

func promoteOpenCodeGoReasoningOutput(raw any) bool {
	items, ok := raw.([]any)
	if !ok {
		return false
	}
	changed := false
	for _, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if ok && promoteOpenCodeGoReasoningItem(item) {
			changed = true
		}
	}
	return changed
}

func promoteOpenCodeGoReasoningItem(item map[string]any) bool {
	if strings.TrimSpace(fmt.Sprint(item["type"])) != "reasoning" {
		return false
	}
	content, ok := item["content"].([]any)
	if !ok {
		return false
	}

	summary := make([]any, 0, len(content))
	for _, rawPart := range content {
		part, ok := rawPart.(map[string]any)
		if !ok || strings.TrimSpace(fmt.Sprint(part["type"])) != "reasoning_text" {
			continue
		}
		text, ok := part["text"].(string)
		if !ok || text == "" {
			continue
		}
		summaryPart := make(map[string]any, len(part))
		for key, value := range part {
			summaryPart[key] = value
		}
		summaryPart["type"] = "summary_text"
		summary = append(summary, summaryPart)
	}
	if len(summary) == 0 {
		return false
	}

	changed := false
	if !openCodeGoReasoningSummaryHasText(item["summary"]) {
		item["summary"] = summary
		changed = true
	}
	if encrypted, _ := item["encrypted_content"].(string); strings.TrimSpace(encrypted) == "" {
		if rawContent, err := json.Marshal(content); err == nil {
			item["encrypted_content"] = openCodeGoReasoningContentPrefix + base64.RawStdEncoding.EncodeToString(rawContent)
			changed = true
		}
	}
	if item["content"] != nil {
		item["content"] = nil
		changed = true
	}
	return changed
}

func openCodeGoReasoningSummaryHasText(raw any) bool {
	parts, ok := raw.([]any)
	if !ok {
		return false
	}
	for _, rawPart := range parts {
		part, ok := rawPart.(map[string]any)
		if !ok || strings.TrimSpace(fmt.Sprint(part["type"])) != "summary_text" {
			continue
		}
		if text, ok := part["text"].(string); ok && text != "" {
			return true
		}
	}
	return false
}

type openCodeGoReasoningSummaryStreamBody struct {
	*io.PipeReader
	source io.Closer
}

func (b *openCodeGoReasoningSummaryStreamBody) Close() error {
	readerErr := b.PipeReader.Close()
	sourceErr := b.source.Close()
	if readerErr != nil {
		return readerErr
	}
	return sourceErr
}

func newOpenCodeGoReasoningSummaryStreamBody(source io.ReadCloser, maxLineSize int) io.ReadCloser {
	reader, writer := io.Pipe()
	body := &openCodeGoReasoningSummaryStreamBody{PipeReader: reader, source: source}
	go transformOpenCodeGoReasoningSummaryStream(source, writer, maxLineSize)
	return body
}

func transformOpenCodeGoReasoningSummaryStream(source io.ReadCloser, destination *io.PipeWriter, maxLineSize int) {
	defer func() { _ = source.Close() }()
	if maxLineSize <= 0 {
		maxLineSize = defaultMaxLineSize
	}

	scanner := bufio.NewScanner(source)
	scanBuf := getSSEScannerBuf64K()
	defer putSSEScannerBuf64K(scanBuf)
	scanner.Buffer(scanBuf[:0], maxLineSize)
	documents := newOpenAISSEJSONDocumentScanner(scanner)
	buffered := bufio.NewWriterSize(destination, 4*1024)
	pendingFields := make([]string, 0, 2)
	frameEmitted := false

	writeLine := func(line string) error {
		if _, err := buffered.WriteString(line); err != nil {
			return err
		}
		return buffered.WriteByte('\n')
	}
	writePendingFields := func(payload []byte) error {
		eventType := strings.TrimSpace(gjson.GetBytes(payload, "type").String())
		for _, field := range pendingFields {
			if _, isEvent := extractOpenAISSEEventLine(field); isEvent {
				if eventType != "" {
					if err := writeLine("event: " + eventType); err != nil {
						return err
					}
				} else if err := writeLine(field); err != nil {
					return err
				}
				continue
			}
			if err := writeLine(field); err != nil {
				return err
			}
		}
		return nil
	}

	for documents.Scan() {
		line := documents.Text()
		data, isData := extractOpenAISSEDataLine(line)
		if isData {
			payload := []byte(data)
			if updated, changed, err := promoteOpenCodeGoReasoningSummaryPayload(payload); err != nil {
				_ = buffered.Flush()
				_ = destination.CloseWithError(fmt.Errorf("promote OpenCode Go reasoning summary event: %w", err))
				return
			} else if changed {
				payload = updated
			}
			if err := writePendingFields(payload); err != nil {
				_ = destination.CloseWithError(err)
				return
			}
			if err := writeLine("data: " + string(payload)); err != nil {
				_ = destination.CloseWithError(err)
				return
			}
			if err := writeLine(""); err != nil {
				_ = destination.CloseWithError(err)
				return
			}
			if err := buffered.Flush(); err != nil {
				_ = destination.CloseWithError(err)
				return
			}
			pendingFields = pendingFields[:0]
			frameEmitted = true
			continue
		}

		if line == "" {
			if !frameEmitted {
				for _, field := range pendingFields {
					if err := writeLine(field); err != nil {
						_ = destination.CloseWithError(err)
						return
					}
				}
				if len(pendingFields) > 0 {
					if err := writeLine(""); err != nil {
						_ = destination.CloseWithError(err)
						return
					}
					if err := buffered.Flush(); err != nil {
						_ = destination.CloseWithError(err)
						return
					}
				}
			}
			pendingFields = pendingFields[:0]
			frameEmitted = false
			continue
		}

		pendingFields = append(pendingFields, line)
	}

	if err := documents.Err(); err != nil {
		_ = buffered.Flush()
		_ = destination.CloseWithError(err)
		return
	}
	for _, field := range pendingFields {
		if err := writeLine(field); err != nil {
			_ = destination.CloseWithError(err)
			return
		}
	}
	if err := buffered.Flush(); err != nil {
		_ = destination.CloseWithError(err)
		return
	}
	_ = destination.Close()
}
