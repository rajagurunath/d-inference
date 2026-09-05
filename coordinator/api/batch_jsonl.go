package api

// Batch input parsing and validation for the Tidal batch lane
// (docs/design/tidal-batch-lane.md §3.6).
//
// Two consumer shapes reach the same parsed representation: the OpenAI JSONL
// upload (one {custom_id, method, url, body} object per line) and the
// OpenRouter inline array ({model, requests:[{custom_id, body}]}). Everything
// downstream — blob sealing, dispatch, output assembly — sees only []parsedItem.
//
// Privacy: no error produced here quotes a custom_id, a metadata key or value,
// a filename, a requested model, an endpoint, a content part type, or any body
// bytes. Failures identify a line number and a field name, which is enough for
// a consumer to fix the input and carries nothing a log line could not already
// hold.

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"
)

// Input caps, verbatim from the plan's Global Constraints. maxFileBytes
// matches the sealed-transport body cap so a sealed upload and a multipart
// upload reject at the same size.
const (
	maxFileBytes      = 16 << 20 // 16 MiB
	maxFileLines      = 50000
	maxInlineRequests = 10000

	// batchCompletionWindow is the only accepted completion_window in v1.
	batchCompletionWindow = "24h"

	maxMetadataKeys     = 16
	maxMetadataKeyLen   = 64
	maxMetadataValueLen = 512
)

// customIDPattern is the full set of characters a custom_id may use. It is
// deliberately narrow: a custom_id is echoed into output files and must never
// be usable as a path segment, a log injection, or a blob ref.
var customIDPattern = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

// batchEndpoints are the endpoints a batch may target. Anything else — most
// notably /v1/embeddings — is rejected at creation rather than at dispatch.
var batchEndpoints = map[string]bool{
	"/v1/chat/completions": true,
	"/v1/completions":      true,
}

// textContentPartTypes are the only content part types a batch item may carry.
// Media parts would require the media resolver and a vision-capable provider
// hours after the consumer went away, so they are refused up front.
var textContentPartTypes = map[string]bool{
	"text":       true,
	"input_text": true,
}

// batchError is a validation failure that maps 1:1 onto an HTTP error
// response. Message quotes line numbers and field names only.
type batchError struct {
	Status  int
	Type    string
	Code    string
	Param   string
	Message string
}

func (e *batchError) Error() string { return e.Message }

// batchErr builds a 400 with the standard invalid_request_error envelope type
// and a batch-specific code.
func batchErr(code, param, format string, args ...any) *batchError {
	return &batchError{
		Status:  http.StatusBadRequest,
		Type:    "invalid_request_error",
		Code:    code,
		Param:   param,
		Message: fmt.Sprintf(format, args...),
	}
}

// writeBatchError renders err through the coordinator's standard error
// envelope. A non-batchError is a coordinator bug, not consumer input, so it
// becomes a 500 with no detail.
func (s *Server) writeBatchError(w http.ResponseWriter, err error) {
	var be *batchError
	if !errors.As(err, &be) {
		s.logger.Error("batch: unexpected handler error", "error", err)
		writeJSON(w, http.StatusInternalServerError,
			errorResponse("internal_error", "internal server error"))
		return
	}
	opts := []errorDetailOpt{withCode(be.Code)}
	if be.Param != "" {
		opts = append(opts, withParam(be.Param))
	}
	writeJSON(w, be.Status, errorResponse(be.Type, be.Message, opts...))
}

// batchLine is one line of an uploaded JSONL input file.
type batchLine struct {
	CustomID string          `json:"custom_id"`
	Method   string          `json:"method"`
	URL      string          `json:"url"`
	Body     json.RawMessage `json:"body"`
}

// inlineRequest is one element of the OpenRouter inline requests array. It
// carries no method or url: both come from the batch's own endpoint.
type inlineRequest struct {
	CustomID string          `json:"custom_id"`
	Body     json.RawMessage `json:"body"`
}

// parsedItem is one validated request ready to be sealed and stored. Raw is
// the body with model rewritten to the concrete build id, so the dispatcher
// never repeats alias resolution hours later when the alias may have moved.
//
// BOTH names are kept, because they answer different questions:
//
//   - RequestedModel is what the CONSUMER typed (an alias, or a raw build id).
//     It is the name the per-key allow-list is checked against, exactly as an
//     online request's is (api/inference_preprocess.go), so a key restricted to
//     an alias means the same thing on both lanes.
//   - Model is the resolved build id. It is what dispatch uses.
type parsedItem struct {
	Line           batchLine
	LineNo         int
	Model          string
	RequestedModel string
	Raw            []byte
}

// modelResolver maps a consumer-requested model to the concrete build id, or
// reports that it cannot be served. Injected so the parser stays pure and
// testable without a registry.
type modelResolver func(requested string) (string, bool)

// parseBatchJSONL reads an uploaded JSONL input file and validates every line
// before any of it is stored. It returns the first failure, so a consumer
// fixes one problem per round trip and a partially valid file never becomes a
// half-populated batch.
//
// An empty endpoint means "take it from the first line": a file is uploaded
// before the consumer says which endpoint the batch targets, so upload-time
// validation infers it and then holds every later line to it. The caller reads
// the settled endpoint back off items[0].Line.URL.
func parseBatchJSONL(r io.Reader, endpoint string, maxLines int, resolveModel modelResolver) ([]parsedItem, error) {
	inferEndpoint := endpoint == ""
	if !inferEndpoint && !batchEndpoints[endpoint] {
		return nil, batchErr("invalid_endpoint", "endpoint",
			"endpoint is not available for batch — use /v1/chat/completions or /v1/completions")
	}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), maxFileBytes)

	items := make([]parsedItem, 0, 64)
	seen := make(map[string]struct{}, 64)
	lineNo := 0

	for scanner.Scan() {
		lineNo++
		raw := strings.TrimSpace(scanner.Text())
		if raw == "" {
			continue
		}
		if len(items) >= maxLines {
			return nil, batchErr("batch_too_large", "input_file_id",
				"input file has more than %d requests", maxLines)
		}

		var line batchLine
		if err := json.Unmarshal([]byte(raw), &line); err != nil {
			return nil, batchErr("invalid_line", "input_file_id",
				"line %d is not a JSON object", lineNo)
		}
		if line.Method != "" && !strings.EqualFold(line.Method, http.MethodPost) {
			return nil, batchErr("invalid_line", "method",
				"line %d: method must be POST", lineNo)
		}
		if inferEndpoint && endpoint == "" {
			if !batchEndpoints[line.URL] {
				return nil, batchErr("invalid_endpoint", "url",
					"line %d: url must be /v1/chat/completions or /v1/completions", lineNo)
			}
			endpoint = line.URL
		}
		if line.URL != endpoint {
			return nil, batchErr("invalid_line", "url",
				"line %d: url must equal the batch endpoint", lineNo)
		}
		if err := validateCustomID(line.CustomID, lineNo); err != nil {
			return nil, err
		}
		if _, dup := seen[line.CustomID]; dup {
			return nil, batchErr("duplicate_custom_id", "custom_id",
				"line %d: custom_id already used earlier in this file", lineNo)
		}
		seen[line.CustomID] = struct{}{}

		requested, model, body, err := validateBatchBody(line.Body, "", lineNo, resolveModel)
		if err != nil {
			return nil, err
		}
		items = append(items, parsedItem{
			Line:           batchLine{CustomID: line.CustomID, Method: http.MethodPost, URL: endpoint},
			LineNo:         len(items) + 1,
			Model:          model,
			RequestedModel: requested,
			Raw:            body,
		})
	}
	if err := scanner.Err(); err != nil {
		if errors.Is(err, bufio.ErrTooLong) {
			return nil, batchErr("invalid_line", "input_file_id",
				"a line in the input file is longer than the %d-byte limit", maxFileBytes)
		}
		return nil, batchErr("invalid_line", "input_file_id", "input file could not be read")
	}
	if len(items) == 0 {
		return nil, batchErr("empty_input", "input_file_id", "input file contains no requests")
	}
	return items, nil
}

// parseInlineRequests validates the OpenRouter inline form. The model is
// supplied once at the top level; a body may still carry its own, in which
// case the body wins for that item.
func parseInlineRequests(reqs []inlineRequest, endpoint, model string, max int, resolveModel modelResolver) ([]parsedItem, error) {
	if !batchEndpoints[endpoint] {
		return nil, batchErr("invalid_endpoint", "endpoint",
			"endpoint is not available for batch — use /v1/chat/completions or /v1/completions")
	}
	if strings.TrimSpace(model) == "" {
		return nil, batchErr("invalid_request", "model",
			"model is required when requests are supplied inline")
	}
	if len(reqs) == 0 {
		return nil, batchErr("empty_input", "requests", "requests is empty")
	}
	if len(reqs) > max {
		return nil, batchErr("batch_too_large", "requests",
			"requests holds more than %d entries", max)
	}

	items := make([]parsedItem, 0, len(reqs))
	seen := make(map[string]struct{}, len(reqs))
	for i, req := range reqs {
		lineNo := i + 1
		if err := validateCustomID(req.CustomID, lineNo); err != nil {
			return nil, err
		}
		if _, dup := seen[req.CustomID]; dup {
			return nil, batchErr("duplicate_custom_id", "custom_id",
				"request %d: custom_id already used earlier in this batch", lineNo)
		}
		seen[req.CustomID] = struct{}{}

		requested, resolved, body, err := validateBatchBody(req.Body, model, lineNo, resolveModel)
		if err != nil {
			return nil, err
		}
		items = append(items, parsedItem{
			Line:           batchLine{CustomID: req.CustomID, Method: http.MethodPost, URL: endpoint},
			LineNo:         lineNo,
			Model:          resolved,
			RequestedModel: requested,
			Raw:            body,
		})
	}
	return items, nil
}

// validateCustomID enforces the 64-character [A-Za-z0-9_-] shape. The offending
// value is never echoed: it lands in output files and must not be reflected
// into an error a proxy might log.
func validateCustomID(customID string, lineNo int) error {
	if customID == "" {
		return batchErr("invalid_custom_id", "custom_id",
			"line %d: custom_id is required", lineNo)
	}
	if !customIDPattern.MatchString(customID) {
		return batchErr("invalid_custom_id", "custom_id",
			"line %d: custom_id must match ^[A-Za-z0-9_-]{1,64}$", lineNo)
	}
	return nil
}

// validateBatchBody checks one request body and returns the name the consumer
// asked for, the concrete build model it resolves to, and the body with "model"
// rewritten to that build id. defaultModel is the inline form's top-level model
// and is empty for the file form, where every line carries its own.
//
// The requested name is returned alongside the resolved one because the
// per-key model allow-list is checked against what the consumer typed — the
// build id is a coordinator-internal name a consumer never puts on a key.
func validateBatchBody(body json.RawMessage, defaultModel string, lineNo int, resolveModel modelResolver) (string, string, []byte, error) {
	if len(body) == 0 {
		return "", "", nil, batchErr("invalid_line", "body", "line %d: body is required", lineNo)
	}
	// Decode with UseNumber so an integer like "seed" is kept as its exact
	// json.Number text rather than round-tripped through float64, which loses
	// precision above 2^53 and would silently mangle the value the dispatcher
	// later sends on.
	var parsed map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&parsed); err != nil {
		return "", "", nil, batchErr("invalid_line", "body", "line %d: body is not a JSON object", lineNo)
	}

	if stream, ok := parsed["stream"].(bool); ok && stream {
		return "", "", nil, batchErr("invalid_line", "stream",
			"line %d: stream must be false — batch results are delivered as files", lineNo)
	}
	if n, ok := parsed["n"]; ok && n != nil {
		num, ok := n.(json.Number)
		if !ok {
			return "", "", nil, batchErr("invalid_line", "n", "line %d: n must be a number", lineNo)
		}
		count, err := num.Float64()
		if err != nil {
			return "", "", nil, batchErr("invalid_line", "n", "line %d: n must be a number", lineNo)
		}
		if count > 1 {
			return "", "", nil, batchErr("invalid_line", "n", "line %d: n must be 1 or absent", lineNo)
		}
	}
	if err := validateTextOnlyContent(parsed, lineNo); err != nil {
		return "", "", nil, err
	}

	requested := defaultModel
	if m, ok := parsed["model"].(string); ok && strings.TrimSpace(m) != "" {
		requested = strings.TrimSpace(m)
	}
	if requested == "" {
		return "", "", nil, batchErr("invalid_line", "model", "line %d: model is required", lineNo)
	}
	resolved, ok := resolveModel(requested)
	if !ok {
		return "", "", nil, batchErr("model_not_found", "model",
			"line %d: model is not available for batch", lineNo)
	}

	if current, _ := parsed["model"].(string); current == resolved {
		return requested, resolved, append([]byte(nil), body...), nil
	}
	parsed["model"] = resolved
	rewritten, err := json.Marshal(parsed)
	if err != nil {
		return "", "", nil, batchErr("invalid_line", "body", "line %d: body could not be re-encoded", lineNo)
	}
	return requested, resolved, rewritten, nil
}

// validateTextOnlyContent rejects image, audio, video, and file content parts.
// A batch item may wait hours before it is dispatched; a media reference that
// was live at submission time is not guaranteed to be live at dispatch, and the
// batch lane deliberately does not run the media resolver. The offending part
// type is a consumer string and is never echoed into the error.
func validateTextOnlyContent(parsed map[string]any, lineNo int) error {
	messages, ok := parsed["messages"].([]any)
	if !ok {
		return nil
	}
	for _, m := range messages {
		msg, ok := m.(map[string]any)
		if !ok {
			continue
		}
		parts, ok := msg["content"].([]any)
		if !ok {
			continue
		}
		for _, p := range parts {
			part, ok := p.(map[string]any)
			if !ok {
				continue
			}
			kind, _ := part["type"].(string)
			if !textContentPartTypes[kind] {
				return batchErr("unsupported_content", "messages",
					"line %d: only text content parts are supported in batch", lineNo)
			}
		}
	}
	return nil
}

// validateBatchMetadata enforces the OpenAI metadata limits. Keys and values
// are consumer strings and are never echoed into the error.
func validateBatchMetadata(metadata map[string]string) error {
	if len(metadata) > maxMetadataKeys {
		return batchErr("invalid_metadata", "metadata",
			"metadata holds more than %d keys", maxMetadataKeys)
	}
	for k, v := range metadata {
		if k == "" || len(k) > maxMetadataKeyLen {
			return batchErr("invalid_metadata", "metadata",
				"every metadata key must be 1-%d characters", maxMetadataKeyLen)
		}
		if len(v) > maxMetadataValueLen {
			return batchErr("invalid_metadata", "metadata",
				"every metadata value must be at most %d characters", maxMetadataValueLen)
		}
	}
	return nil
}
