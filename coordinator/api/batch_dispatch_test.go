package api

import (
	"context"
	"encoding/json"
	"log/slog"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
	"nhooyr.io/websocket"
)

// batchFakeProvider stands up a coordinator with one fake provider on the
// WebSocket harness (mirrors TestNonStreamingE2E) and answers exactly one
// inference request with content plus the given usage.
func batchFakeProvider(t *testing.T, model string, usage protocol.UsageInfo, content string) (
	*Server, *registry.Registry, context.Context, <-chan struct{},
) {
	t.Helper()
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelError}))
	st := store.NewMemory(store.Config{AdminKey: "test-key"})
	reg := registry.New(logger)
	srv := NewServer(reg, st, ServerConfig{}, logger)

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	t.Cleanup(cancel)

	wsURL := "ws" + strings.TrimPrefix(ts.URL, "http") + "/ws/provider"
	conn, _, err := websocket.Dial(ctx, wsURL, nil)
	if err != nil {
		t.Fatalf("websocket dial: %v", err)
	}
	t.Cleanup(func() { conn.Close(websocket.StatusNormalClosure, "") })

	pubKey := testPublicKeyB64()
	regMsg := protocol.RegisterMessage{
		Type:                    protocol.TypeRegister,
		Hardware:                protocol.Hardware{ChipName: "M3 Max", MemoryGB: 64},
		Models:                  []protocol.ModelInfo{{ID: model, ModelType: "test", Quantization: "4bit"}},
		Backend:                 "mlx-swift",
		PublicKey:               pubKey,
		EncryptedResponseChunks: true,
		PrivacyCapabilities:     testPrivacyCaps(),
	}
	regData, _ := json.Marshal(regMsg)
	if err := conn.Write(ctx, websocket.MessageText, regData); err != nil {
		t.Fatalf("write register: %v", err)
	}
	time.Sleep(100 * time.Millisecond)
	makeProviderRoutable(reg)

	done := make(chan struct{})
	go func() {
		defer close(done)
		var inferReq protocol.InferenceRequestMessage
		for {
			_, data, err := conn.Read(ctx)
			if err != nil {
				return
			}
			var raw map[string]any
			if err := json.Unmarshal(data, &raw); err == nil {
				if raw["type"] == protocol.TypeAttestationChallenge {
					conn.Write(ctx, websocket.MessageText, makeValidChallengeResponse(data, pubKey))
					continue
				}
				if raw["type"] != protocol.TypeInferenceRequest {
					continue
				}
			}
			if err := json.Unmarshal(data, &inferReq); err != nil {
				return
			}
			break
		}
		writeEncryptedTestChunk(t, ctx, conn, inferReq, pubKey,
			`data: {"id":"chatcmpl-1","choices":[{"delta":{"content":"`+content+`"}}]}`+"\n\n")
		complete := protocol.InferenceCompleteMessage{
			Type:      protocol.TypeInferenceComplete,
			RequestID: inferReq.RequestID,
			Usage:     usage,
		}
		completeData, _ := json.Marshal(complete)
		conn.Write(ctx, websocket.MessageText, completeData)
	}()

	return srv, reg, ctx, done
}

// TestDispatchBatchItemCompletesThroughTheFunnel drives one batch item through
// the real dispatch funnel against a fake provider and asserts the outcome the
// batch dispatcher will consume: a request id, the provider-reported token
// counts, and an assembled OpenAI body. It also pins the two invariants the lane
// exists to protect — the attempt is invisible to provider reputation, and it
// never took the wait queue.
func TestDispatchBatchItemCompletesThroughTheFunnel(t *testing.T) {
	const model = "test-model"
	usage := protocol.UsageInfo{PromptTokens: 17, CompletionTokens: 4}
	srv, reg, ctx, providerDone := batchFakeProvider(t, model, usage, "Hello batch")

	body := []byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`)
	outcome, err := srv.DispatchBatchItem(ctx, "admin", "", model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != "" {
		t.Fatalf("ErrCode=%q body=%s, want success", outcome.ErrCode, outcome.ResponseBody)
	}
	if outcome.RequestID == "" {
		t.Fatal("no request id on a committed batch attempt")
	}
	if outcome.PromptTokens != usage.PromptTokens || outcome.CompletionTokens != usage.CompletionTokens {
		t.Fatalf("tokens=%d/%d, want %d/%d", outcome.PromptTokens, outcome.CompletionTokens,
			usage.PromptTokens, usage.CompletionTokens)
	}

	var assembled map[string]any
	if err := json.Unmarshal(outcome.ResponseBody, &assembled); err != nil {
		t.Fatalf("response body is not JSON: %v (%s)", err, outcome.ResponseBody)
	}
	choices, ok := assembled["choices"].([]any)
	if !ok || len(choices) == 0 {
		t.Fatalf("no choices in batch response: %s", outcome.ResponseBody)
	}
	content := choices[0].(map[string]any)["message"].(map[string]any)["content"].(string)
	if content != "Hello batch" {
		t.Fatalf("content=%q, want %q", content, "Hello batch")
	}

	// The batch attempt must not have moved provider reputation.
	for _, id := range reg.ProviderIDs() {
		p := reg.GetProvider(id)
		p.Mu().Lock()
		total := p.Reputation.TotalJobs
		p.Mu().Unlock()
		if total != 0 {
			t.Fatalf("provider %s recorded %d jobs from a batch attempt, want 0", id, total)
		}
	}
	if depth := reg.Queue().TotalSize(); depth != 0 {
		t.Fatalf("coordinator queue depth=%d after a batch dispatch, want 0", depth)
	}

	<-providerDone
}

// TestDispatchBatchItemNoCapacityDoesNotQueue: with every slot for the model
// carrying a waiting row, the batch lane has no headroom. The item must come
// back "no_capacity" immediately instead of parking in the coordinator queue.
func TestDispatchBatchItemNoCapacityDoesNotQueue(t *testing.T) {
	const model = "test-model"
	srv, reg, ctx, _ := batchFakeProvider(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "unused")

	// Close every slot to the batch lane (a waiting row is enough) while leaving
	// the provider perfectly admittable online.
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

	// The fleet is merely batch-closed: an online reservation still succeeds, so
	// the refusal below is the lane filter and not a broken fixture.
	if p := findRoutableProvider(reg, model); p == nil {
		t.Fatal("fixture: no provider is routable online — the refusal would not prove the lane filter")
	}

	start := time.Now()
	outcome, err := srv.DispatchBatchItem(ctx, "admin", "", model,
		[]byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != batchNoCapacityCode {
		t.Fatalf("ErrCode=%q body=%s, want %q",
			outcome.ErrCode, outcome.ResponseBody, batchNoCapacityCode)
	}
	// Never queued: the queue's own wait is 120s, so returning promptly is the
	// observable proof alongside the depth assertion.
	if elapsed := time.Since(start); elapsed > 20*time.Second {
		t.Fatalf("batch dispatch took %v — it waited on something", elapsed)
	}
	if depth := reg.Queue().TotalSize(); depth != 0 {
		t.Fatalf("coordinator queue depth=%d, want 0 (batch never queues)", depth)
	}
}
