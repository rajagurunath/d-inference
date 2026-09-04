package api

import (
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"testing"
)

// passthroughResolver accepts every model unchanged — the parser's model check
// is injected, so a test that is not about model resolution uses this.
func passthroughResolver(requested string) (string, bool) { return requested, true }

// rejectAllResolver refuses every model, standing in for a catalog miss.
func rejectAllResolver(string) (string, bool) { return "", false }

func batchErrorFrom(t *testing.T, err error) *batchError {
	t.Helper()
	if err == nil {
		t.Fatal("want a validation error, got nil")
	}
	var be *batchError
	if !errors.As(err, &be) {
		t.Fatalf("want *batchError, got %T: %v", err, err)
	}
	return be
}

func TestParseBatchJSONLAcceptsTwoValidLines(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":"hi"}]}}
{"custom_id":"b-2","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}}
`
	items, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2", len(items))
	}
	if items[0].Line.CustomID != "a" || items[1].Line.CustomID != "b-2" {
		t.Fatalf("custom ids not preserved: %+v", items)
	}
	if items[0].LineNo != 1 || items[1].LineNo != 2 {
		t.Fatalf("line numbers not 1-based dense: %d %d", items[0].LineNo, items[1].LineNo)
	}
	if items[0].Model != "m" {
		t.Fatalf("model = %q, want m", items[0].Model)
	}
	if !strings.Contains(string(items[0].Raw), `"messages"`) {
		t.Fatalf("raw body lost its messages: %s", items[0].Raw)
	}
}

func TestParseBatchJSONLRewritesModelToBuildID(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"alias","messages":[]}}`
	items, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines,
		func(string) (string, bool) { return "concrete-build", true })
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if items[0].Model != "concrete-build" {
		t.Fatalf("model = %q, want concrete-build", items[0].Model)
	}
	if !strings.Contains(string(items[0].Raw), `"concrete-build"`) {
		t.Fatalf("body was not rewritten to the build id: %s", items[0].Raw)
	}
}

// TestParseBatchJSONLPreservesLargeIntegersOnRewrite pins the fix for decoding
// a batch line's body with json.Decoder.UseNumber(): rewriting "model" to the
// resolved build id re-marshals the body, and without UseNumber the default
// float64 decode of a large integer like "seed" loses precision above 2^53 and
// silently mangles it on the way back out.
func TestParseBatchJSONLPreservesLargeIntegersOnRewrite(t *testing.T) {
	const bigSeed = "9007199254740993" // 2^53 + 1 — not exactly representable as float64
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"alias","seed":` + bigSeed + `,"messages":[]}}`
	items, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines,
		func(string) (string, bool) { return "concrete-build", true })
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if !strings.Contains(string(items[0].Raw), `"concrete-build"`) {
		t.Fatalf("body was not rewritten to the build id: %s", items[0].Raw)
	}
	if !strings.Contains(string(items[0].Raw), `"seed":`+bigSeed) {
		t.Fatalf("seed was not preserved byte-exact across the rewrite: %s", items[0].Raw)
	}
}

func TestParseBatchJSONLRejectsDuplicateCustomID(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m"}}
{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m"}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "duplicate_custom_id" {
		t.Fatalf("code = %q, want duplicate_custom_id", be.Code)
	}
	if !strings.Contains(be.Message, "line 2") {
		t.Fatalf("message must name the offending line, got %q", be.Message)
	}
	if strings.Contains(be.Message, `"a"`) {
		t.Fatalf("message must not echo the custom_id, got %q", be.Message)
	}
}

func TestParseBatchJSONLRejectsStreaming(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","stream":true}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Param != "stream" {
		t.Fatalf("param = %q, want stream", be.Param)
	}
}

func TestParseBatchJSONLRejectsMultipleCompletions(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","n":2}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Param != "n" {
		t.Fatalf("param = %q, want n", be.Param)
	}

	ok := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","n":1}}`
	if _, err := parseBatchJSONL(strings.NewReader(ok), "/v1/chat/completions", maxFileLines, passthroughResolver); err != nil {
		t.Fatalf("n=1 must be accepted, got %v", err)
	}
}

func TestParseBatchJSONLRejectsWrongURL(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/embeddings","body":{"model":"m"}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Param != "url" {
		t.Fatalf("param = %q, want url", be.Param)
	}
}

func TestParseBatchJSONLRejectsNonPOSTMethod(t *testing.T) {
	input := `{"custom_id":"a","method":"GET","url":"/v1/chat/completions","body":{"model":"m"}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Param != "method" {
		t.Fatalf("param = %q, want method", be.Param)
	}
}

func TestParseBatchJSONLRejectsUnknownModel(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"nope"}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, rejectAllResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "model_not_found" {
		t.Fatalf("code = %q, want model_not_found", be.Code)
	}
}

func TestParseBatchJSONLRejectsNonTextContentParts(t *testing.T) {
	input := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}]}}`
	_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "unsupported_content" {
		t.Fatalf("code = %q, want unsupported_content", be.Code)
	}
}

// TestBatchValidationErrorsNeverEchoConsumerStrings pins the privacy contract
// stated at the top of batch_jsonl.go: a rejection identifies a line number and
// a field name and nothing else. An error body travels back through whatever
// proxies and log pipelines sit in front of the coordinator, so a rejected
// model name, endpoint, custom_id, or content part type must not ride along.
// Every case below feeds a marker string no rejection may quote back.
func TestBatchValidationErrorsNeverEchoConsumerStrings(t *testing.T) {
	const marker = "leaky-consumer-string"

	cases := []struct {
		name     string
		wantCode string
		parse    func() error
	}{
		{
			name:     "unknown model on a file line",
			wantCode: "model_not_found",
			parse: func() error {
				line := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"` + marker + `"}}`
				_, err := parseBatchJSONL(strings.NewReader(line), "/v1/chat/completions", maxFileLines, rejectAllResolver)
				return err
			},
		},
		{
			name:     "unsupported batch endpoint",
			wantCode: "invalid_endpoint",
			parse: func() error {
				_, err := parseBatchJSONL(strings.NewReader("{}"), "/v1/"+marker, maxFileLines, passthroughResolver)
				return err
			},
		},
		{
			name:     "line url that does not match the batch endpoint",
			wantCode: "invalid_line",
			parse: func() error {
				line := `{"custom_id":"a","method":"POST","url":"/v1/completions","body":{"model":"m"}}`
				_, err := parseBatchJSONL(strings.NewReader(line), "/v1/chat/completions", maxFileLines, passthroughResolver)
				return err
			},
		},
		{
			name:     "url that is not a batch endpoint at all",
			wantCode: "invalid_endpoint",
			parse: func() error {
				line := `{"custom_id":"a","method":"POST","url":"/v1/` + marker + `","body":{"model":"m"}}`
				_, err := parseBatchJSONL(strings.NewReader(line), "", maxFileLines, passthroughResolver)
				return err
			},
		},
		{
			name:     "malformed custom_id",
			wantCode: "invalid_custom_id",
			parse: func() error {
				line := `{"custom_id":"` + marker + `!!","method":"POST","url":"/v1/chat/completions","body":{"model":"m"}}`
				_, err := parseBatchJSONL(strings.NewReader(line), "/v1/chat/completions", maxFileLines, passthroughResolver)
				return err
			},
		},
		{
			name:     "non-text content part",
			wantCode: "unsupported_content",
			parse: func() error {
				line := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"m","messages":[{"role":"user","content":[{"type":"` + marker + `"}]}]}}`
				_, err := parseBatchJSONL(strings.NewReader(line), "/v1/chat/completions", maxFileLines, passthroughResolver)
				return err
			},
		},
		{
			name:     "unsupported endpoint on the inline path",
			wantCode: "invalid_endpoint",
			parse: func() error {
				_, err := parseInlineRequests(nil, "/v1/"+marker, "m", maxInlineRequests, passthroughResolver)
				return err
			},
		},
		{
			name:     "unknown model on the inline path",
			wantCode: "model_not_found",
			parse: func() error {
				reqs := []inlineRequest{{CustomID: "a", Body: json.RawMessage(`{"model":"` + marker + `"}`)}}
				_, err := parseInlineRequests(reqs, "/v1/chat/completions", marker, maxInlineRequests, rejectAllResolver)
				return err
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			be := batchErrorFrom(t, tc.parse())
			if be.Code != tc.wantCode {
				t.Fatalf("code = %q, want %q", be.Code, tc.wantCode)
			}
			if strings.Contains(be.Message, marker) {
				t.Fatalf("error message echoes a consumer string: %q", be.Message)
			}
			if strings.Contains(be.Param, marker) {
				t.Fatalf("error param echoes a consumer string: %q", be.Param)
			}
		})
	}
}

func TestParseBatchJSONLRejectsUnknownEndpoint(t *testing.T) {
	_, err := parseBatchJSONL(strings.NewReader("{}"), "/v1/embeddings", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "invalid_endpoint" {
		t.Fatalf("code = %q, want invalid_endpoint", be.Code)
	}
}

func TestParseBatchJSONLRejectsTooManyLines(t *testing.T) {
	var b strings.Builder
	for i := 0; i < maxFileLines+1; i++ {
		b.WriteString(`{"custom_id":"c`)
		b.WriteString(strings.Repeat("0", 1))
		b.WriteString(strconv.Itoa(i))
		b.WriteString(`","method":"POST","url":"/v1/chat/completions","body":{"model":"m"}}` + "\n")
	}
	_, err := parseBatchJSONL(strings.NewReader(b.String()), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "batch_too_large" {
		t.Fatalf("code = %q, want batch_too_large", be.Code)
	}
}

func TestParseBatchJSONLRejectsEmptyFile(t *testing.T) {
	_, err := parseBatchJSONL(strings.NewReader("\n \n"), "/v1/chat/completions", maxFileLines, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "empty_input" {
		t.Fatalf("code = %q, want empty_input", be.Code)
	}
}

func TestParseBatchJSONLRejectsBadCustomID(t *testing.T) {
	for _, id := range []string{"a b", strings.Repeat("x", 65), "a/b", ""} {
		input := `{"custom_id":"` + id + `","method":"POST","url":"/v1/chat/completions","body":{"model":"m"}}`
		_, err := parseBatchJSONL(strings.NewReader(input), "/v1/chat/completions", maxFileLines, passthroughResolver)
		be := batchErrorFrom(t, err)
		if be.Code != "invalid_custom_id" {
			t.Fatalf("custom_id %q: code = %q, want invalid_custom_id", id, be.Code)
		}
	}
}

func TestParseInlineRequestsAcceptsTopLevelModel(t *testing.T) {
	items, err := parseInlineRequests([]inlineRequest{
		{CustomID: "a", Body: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
		{CustomID: "b", Body: []byte(`{"messages":[{"role":"user","content":"yo"}]}`)},
	}, "/v1/chat/completions", "alias", maxInlineRequests,
		func(string) (string, bool) { return "build", true })
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2", len(items))
	}
	for _, it := range items {
		if it.Model != "build" {
			t.Fatalf("model = %q, want build", it.Model)
		}
		if !strings.Contains(string(it.Raw), `"build"`) {
			t.Fatalf("inline body was not stamped with the build id: %s", it.Raw)
		}
		if it.Line.URL != "/v1/chat/completions" {
			t.Fatalf("url = %q", it.Line.URL)
		}
	}
}

func TestParseInlineRequestsRequiresModel(t *testing.T) {
	_, err := parseInlineRequests([]inlineRequest{{CustomID: "a", Body: []byte(`{}`)}},
		"/v1/chat/completions", "", maxInlineRequests, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Param != "model" {
		t.Fatalf("param = %q, want model", be.Param)
	}
}

func TestParseInlineRequestsRejectsTooMany(t *testing.T) {
	reqs := make([]inlineRequest, 3)
	_, err := parseInlineRequests(reqs, "/v1/chat/completions", "m", 2, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "batch_too_large" {
		t.Fatalf("code = %q, want batch_too_large", be.Code)
	}
}

func TestParseInlineRequestsRejectsDuplicateCustomID(t *testing.T) {
	_, err := parseInlineRequests([]inlineRequest{
		{CustomID: "a", Body: []byte(`{}`)},
		{CustomID: "a", Body: []byte(`{}`)},
	}, "/v1/chat/completions", "m", maxInlineRequests, passthroughResolver)
	be := batchErrorFrom(t, err)
	if be.Code != "duplicate_custom_id" {
		t.Fatalf("code = %q, want duplicate_custom_id", be.Code)
	}
}

func TestValidateBatchMetadataLimits(t *testing.T) {
	if err := validateBatchMetadata(map[string]string{"a": "b"}); err != nil {
		t.Fatalf("small metadata must pass: %v", err)
	}
	big := make(map[string]string, maxMetadataKeys+1)
	for i := 0; i <= maxMetadataKeys; i++ {
		big["k"+strconv.Itoa(i)] = "v"
	}
	if be := batchErrorFrom(t, validateBatchMetadata(big)); be.Code != "invalid_metadata" {
		t.Fatalf("code = %q, want invalid_metadata", be.Code)
	}
	longKey := map[string]string{strings.Repeat("k", maxMetadataKeyLen+1): "v"}
	if be := batchErrorFrom(t, validateBatchMetadata(longKey)); be.Code != "invalid_metadata" {
		t.Fatalf("long key must be rejected, got %q", be.Code)
	}
	longVal := map[string]string{"k": strings.Repeat("v", maxMetadataValueLen+1)}
	if be := batchErrorFrom(t, validateBatchMetadata(longVal)); be.Code != "invalid_metadata" {
		t.Fatalf("long value must be rejected, got %q", be.Code)
	}
}
