package api

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
)

func TestResolveRequestLane(t *testing.T) {
	for _, tc := range []struct {
		name   string
		parsed map[string]any
		want   registry.Lane
	}{
		{"absent", map[string]any{}, registry.LaneOnline},
		{"batch", map[string]any{"service_tier": "batch"}, registry.LaneBatch},
		{"auto is ignored", map[string]any{"service_tier": "auto"}, registry.LaneOnline},
		{"default is ignored", map[string]any{"service_tier": "default"}, registry.LaneOnline},
		{"flex is ignored", map[string]any{"service_tier": "flex"}, registry.LaneOnline},
		{"case sensitive", map[string]any{"service_tier": "Batch"}, registry.LaneOnline},
		{"non-string is ignored", map[string]any{"service_tier": 3}, registry.LaneOnline},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveRequestLane(context.Background(), tc.parsed); got != tc.want {
				t.Fatalf("lane=%q, want %q", got, tc.want)
			}
		})
	}

	// A coordinator-stamped lane outranks the body: DispatchBatchItem forwards
	// the consumer's own body, which may carry any service_tier at all.
	ctx := withRequestLane(context.Background(), registry.LaneBatch)
	if got := resolveRequestLane(ctx, map[string]any{"service_tier": "auto"}); got != registry.LaneBatch {
		t.Fatalf("stamped lane=%q, want batch", got)
	}
}

// batchServiceTierServer stands up the fake-provider harness plus an httptest
// server in front of the real router, so service_tier requests travel the whole
// authenticated HTTP path.
func batchServiceTierServer(t *testing.T, model string) (*httptest.Server, *registry.Registry, context.Context) {
	t.Helper()
	srv, reg, ctx, _ := batchFakeProvider(t, model,
		protocol.UsageInfo{PromptTokens: 9, CompletionTokens: 3}, "Hello tier")
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	return ts, reg, ctx
}

func closeSlotsToBatch(t *testing.T, reg *registry.Registry, model string) {
	t.Helper()
	for _, id := range reg.ProviderIDs() {
		p := reg.GetProvider(id)
		p.Mu().Lock()
		p.BackendCapacity = &protocol.BackendCapacity{
			TotalMemoryGB: 64,
			Slots: []protocol.BackendSlotCapacity{
				{Model: model, State: "running", NumRunning: 0, NumWaiting: 1},
			},
		}
		p.Mu().Unlock()
	}
	if p := findRoutableProvider(reg, model); p == nil {
		t.Fatal("fixture: the batch-closed fleet is not routable online either")
	}
}

func postServiceTierChat(t *testing.T, ctx context.Context, ts *httptest.Server, body string) *http.Response {
	t.Helper()
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		ts.URL+"/v1/chat/completions", strings.NewReader(body))
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer test-key")
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("http request: %v", err)
	}
	return resp
}

// TestServiceTierBatchIs429WithRetryAfterWhenNoHeadroom: a synchronous
// service_tier=batch request lands on the batch lane, so a fleet with no
// headroom answers immediately with a retryable 429 + Retry-After instead of
// queueing for the coordinator's 120s wait.
func TestServiceTierBatchIs429WithRetryAfterWhenNoHeadroom(t *testing.T) {
	const model = "test-model"
	ts, reg, ctx := batchServiceTierServer(t, model)
	closeSlotsToBatch(t, reg, model)

	resp := postServiceTierChat(t, ctx, ts,
		`{"model":"test-model","service_tier":"batch","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("status=%d body=%s, want 429", resp.StatusCode, body)
	}
	if got := resp.Header.Get("Retry-After"); got != "5" {
		t.Fatalf("Retry-After=%q, want %q", got, "5")
	}
	if code := responseErrorCode(body); code != batchNoCapacityCode {
		t.Fatalf("error.code=%q body=%s, want %q", code, body, batchNoCapacityCode)
	}
	if depth := reg.Queue().TotalSize(); depth != 0 {
		t.Fatalf("queue depth=%d, want 0 (service_tier=batch never queues)", depth)
	}
}

// TestServiceTierOtherValuesAreIgnored: every non-"batch" service_tier keeps the
// request on the online lane, so the same batch-closed fleet serves it normally.
func TestServiceTierOtherValuesAreIgnored(t *testing.T) {
	const model = "test-model"
	ts, reg, ctx := batchServiceTierServer(t, model)
	closeSlotsToBatch(t, reg, model)

	resp := postServiceTierChat(t, ctx, ts,
		`{"model":"test-model","service_tier":"flex","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s, want 200 (an unknown service_tier is ignored)", resp.StatusCode, body)
	}
}

// TestServiceTierBatchServesFromHeadroom: with headroom available the batch lane
// completes exactly like an online request, and still leaves reputation alone.
func TestServiceTierBatchServesFromHeadroom(t *testing.T) {
	const model = "test-model"
	ts, reg, ctx := batchServiceTierServer(t, model)

	resp := postServiceTierChat(t, ctx, ts,
		`{"model":"test-model","service_tier":"batch","messages":[{"role":"user","content":"hi"}]}`)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s, want 200", resp.StatusCode, body)
	}
	var assembled map[string]any
	if err := json.Unmarshal(body, &assembled); err != nil {
		t.Fatalf("decode: %v (%s)", err, body)
	}
	choices, ok := assembled["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("no choices: %s", body)
	}
	for _, id := range reg.ProviderIDs() {
		p := reg.GetProvider(id)
		p.Mu().Lock()
		total := p.Reputation.TotalJobs
		p.Mu().Unlock()
		if total != 0 {
			t.Fatalf("service_tier=batch fed reputation on %s: %d jobs", id, total)
		}
	}
}
