package testbed

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestBatchInputJSONLRendersOneObjectPerLine(t *testing.T) {
	content, err := BatchInputJSONL([]BatchInputLine{
		{CustomID: "a-0", Model: "m/one", Prompt: "hello", MaxTokens: 8},
		{CustomID: "a-1", Model: "m/one", Prompt: "hello", MaxTokens: 8},
	})
	if err != nil {
		t.Fatalf("BatchInputJSONL: %v", err)
	}

	lines := strings.Split(strings.TrimSuffix(string(content), "\n"), "\n")
	if len(lines) != 2 {
		t.Fatalf("got %d lines, want 2: %q", len(lines), content)
	}

	var first struct {
		CustomID string `json:"custom_id"`
		Method   string `json:"method"`
		URL      string `json:"url"`
		Body     struct {
			Model     string `json:"model"`
			MaxTokens int    `json:"max_tokens"`
			Messages  []struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"messages"`
		} `json:"body"`
	}
	if err := json.Unmarshal([]byte(lines[0]), &first); err != nil {
		t.Fatalf("line 0 is not JSON: %v", err)
	}
	if first.CustomID != "a-0" || first.Method != "POST" || first.URL != BatchChatCompletionsEndpoint {
		t.Fatalf("unexpected envelope: %+v", first)
	}
	if first.Body.Model != "m/one" || first.Body.MaxTokens != 8 {
		t.Fatalf("unexpected body: %+v", first.Body)
	}
	if len(first.Body.Messages) != 1 || first.Body.Messages[0].Role != "user" {
		t.Fatalf("unexpected messages: %+v", first.Body.Messages)
	}
}

func TestBatchInputLineHonoursExplicitURL(t *testing.T) {
	raw, err := BatchInputLine{CustomID: "c", Model: "m", URL: "/v1/completions"}.MarshalJSONL()
	if err != nil {
		t.Fatalf("MarshalJSONL: %v", err)
	}
	if !strings.Contains(string(raw), `"/v1/completions"`) {
		t.Fatalf("explicit url dropped: %s", raw)
	}
}

func TestBatchIsTerminal(t *testing.T) {
	for _, status := range []string{"completed", "failed", "cancelled", "expired"} {
		if !BatchIsTerminal(status) {
			t.Errorf("%q must be terminal", status)
		}
	}
	for _, status := range []string{"validating", "in_progress", "finalizing", "cancelling", ""} {
		if BatchIsTerminal(status) {
			t.Errorf("%q must not be terminal", status)
		}
	}
}

func TestBatchRequestCountsSettled(t *testing.T) {
	if got := (BatchRequestCounts{Completed: 4, Failed: 2, Total: 10}).Settled(); got != 6 {
		t.Fatalf("Settled() = %d, want 6", got)
	}
}
