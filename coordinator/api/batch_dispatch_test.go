package api

import (
	"context"
	"encoding/json"
	"errors"
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
	srv, reg, _, ctx, done := batchFakeProviderWithStore(t, model, usage, content)
	return srv, reg, ctx, done
}

// batchFakeProviderWithStore is batchFakeProvider with the backing store handed
// back, for the tests that need to mint an API key against it.
func batchFakeProviderWithStore(t *testing.T, model string, usage protocol.UsageInfo, content string) (
	*Server, *registry.Registry, *store.MemoryStore, context.Context, <-chan struct{},
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
	warmSlotForBatch(t, reg, model)

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

	return srv, reg, st, ctx, done
}

// warmSlotForBatch stamps the provider-reported slot the batch lane requires:
// the model RESIDENT ("running") with the row counters idle. The batch gate
// admits only warm slots — batch must never be the traffic that makes a
// provider load weights or evict a co-resident model — and the registration
// handshake alone leaves the slot state "unknown" (cold, model on disk). The
// real dispatcher only ever offers items to warm slots, so this is what the
// fixture has to look like for a batch dispatch to be reachable at all.
func warmSlotForBatch(t *testing.T, reg *registry.Registry, model string) {
	t.Helper()
	for _, id := range reg.ProviderIDs() {
		p := reg.GetProvider(id)
		p.Mu().Lock()
		p.BackendCapacity = &protocol.BackendCapacity{
			TotalMemoryGB: 64,
			Slots: []protocol.BackendSlotCapacity{
				{Model: model, State: "running", NumRunning: 0, NumWaiting: 0},
			},
		}
		p.Mu().Unlock()
	}
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

// TestDispatchBatchItemEnforcesTheRealKeysAllowList is the synthetic-key-stub
// regression. DispatchBatchItem used to inject a &store.APIKey{ID: apiKeyID}
// carrying nothing but the id, which satisfies apiKeyFromContext while reporting
// no AllowedModels and no LimitMicroUSD — so keyModelAllowed and
// checkKeySpendCap waved every batch item through and a key restricted to one
// model became unrestricted the moment its traffic arrived on the batch lane.
func TestDispatchBatchItemEnforcesTheRealKeysAllowList(t *testing.T) {
	const model = "test-model"
	srv, _, st, ctx, _ := batchFakeProviderWithStore(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "unused")

	_, key, err := st.CreateAPIKey("admin", store.APIKeyCreate{
		Name:          "batch-restricted",
		AllowedModels: []string{"some-other-model"},
	})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}

	body := []byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`)
	outcome, err := srv.DispatchBatchItem(ctx, "admin", key.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != batchRequestFailedCode {
		t.Fatalf("ErrCode=%q body=%s, want %q — the key's allow-list was bypassed",
			outcome.ErrCode, outcome.ResponseBody, batchRequestFailedCode)
	}
	if code := responseErrorCode(outcome.ResponseBody); code != "model_not_allowed" {
		t.Fatalf("error.code=%q body=%s, want model_not_allowed", code, outcome.ResponseBody)
	}

	// Control: a key with no allow-list on the same fixture is served, so the
	// refusal above is the allow-list and not a broken key lookup.
	_, open, err := st.CreateAPIKey("admin", store.APIKeyCreate{Name: "batch-open"})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	outcome, err = srv.DispatchBatchItem(ctx, "admin", open.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem (open key): %v", err)
	}
	if outcome.ErrCode != "" {
		t.Fatalf("unrestricted key ErrCode=%q body=%s, want success",
			outcome.ErrCode, outcome.ResponseBody)
	}
}

// TestDispatchBatchItemRejectsUnusableKeys: an api key id that does not resolve
// to a live key owned by the account fails the item with a typed error instead
// of dispatching it as if the key existed.
func TestDispatchBatchItemRejectsUnusableKeys(t *testing.T) {
	const model = "test-model"
	srv, _, st, ctx, _ := batchFakeProviderWithStore(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "unused")
	body := []byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`)

	raw, revoked, err := st.CreateAPIKey("admin", store.APIKeyCreate{Name: "batch-revoked"})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}
	if !st.RevokeKey(raw) {
		t.Fatal("fixture: key was not revoked")
	}

	cases := []struct {
		name      string
		accountID string
		keyID     string
	}{
		{"unknown key id", "admin", "key_does_not_exist"},
		{"revoked key", "admin", revoked.ID},
		{"another account's key", "someone-else", revoked.ID},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			outcome, err := srv.DispatchBatchItem(ctx, tc.accountID, tc.keyID, model, body)
			if !errors.Is(err, errBatchAPIKeyUnusable) {
				t.Fatalf("err=%v, want errBatchAPIKeyUnusable", err)
			}
			if outcome.ErrCode != batchRequestFailedCode {
				t.Fatalf("ErrCode=%q, want %q", outcome.ErrCode, batchRequestFailedCode)
			}
		})
	}
}

// TestDispatchBatchItemReportsCallerCancellation: the caller's own context
// ending (coordinator shutdown, the batch cancelled under us) must NOT be
// mapped to "request_failed". request_failed burns one of the item's three
// attempts, and a cancellation proves nothing about whether the item can be
// served — three restarts would otherwise retire a perfectly good item.
func TestDispatchBatchItemReportsCallerCancellation(t *testing.T) {
	const model = "test-model"
	srv, _, ctx, _ := batchFakeProvider(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "unused")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()

	outcome, err := srv.DispatchBatchItem(cancelled, "admin", "", model,
		[]byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`))
	if outcome.ErrCode != batchCancelledCode {
		t.Fatalf("ErrCode=%q body=%s, want %q", outcome.ErrCode, outcome.ResponseBody, batchCancelledCode)
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err=%v, want a wrapped context.Canceled", err)
	}
}

// TestBatchWithARestrictedKeyFailsItsItems closes the loop PR3c opened: a batch
// row stamped with a key whose AllowedModels excludes the batch's model has its
// items settled as request_failed rather than served. Before the key id was
// carried on the batch, the dispatcher passed "" and the same batch would have
// run to completion with the key's allow-list ignored.
func TestBatchWithARestrictedKeyFailsItsItems(t *testing.T) {
	const model = "test-model"
	srv, _, st, ctx, _ := batchFakeProviderWithStore(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "unused")

	_, key, err := st.CreateAPIKey("admin", store.APIKeyCreate{
		Name:          "batch-restricted",
		AllowedModels: []string{"some-other-model"},
	})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}

	// The batch row as handleBatchCreate now writes it: stamped with the key
	// that submitted it.
	now := time.Now().UTC()
	batch := &store.Batch{
		ID:               "batch_restricted",
		AccountID:        "admin",
		APIKeyID:         key.ID,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: "24h",
		CreatedAt:        now,
		ExpiresAt:        now.Add(24 * time.Hour),
		CountsTotal:      1,
		SealedTo:         "coordinator",
		Source:           "inline",
		Model:            model,
	}
	items := []*store.BatchItem{{
		ID: "bitem_restricted", BatchID: batch.ID, CustomID: "a",
		LineNo: 1, State: store.ItemPending, BlobRef: "bitem_restricted-in",
	}}
	if err := st.CreateBatch(batch, items); err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	if ok, err := st.SetBatchStatus(batch.ID, store.BatchValidating, store.BatchInProgress, now); err != nil || !ok {
		t.Fatalf("SetBatchStatus: ok=%v err=%v", ok, err)
	}

	// What the dispatcher does each tick: read the batch back, claim an item,
	// run it through the funnel with the batch's key, settle the outcome.
	open, ok := st.GetBatch("admin", batch.ID)
	if !ok {
		t.Fatal("GetBatch: batch not found")
	}
	if open.APIKeyID != key.ID {
		t.Fatalf("batch APIKeyID = %q, want %q", open.APIKeyID, key.ID)
	}
	claimed, err := st.ClaimPendingItems(open.ID, 8, now)
	if err != nil || len(claimed) != 1 {
		t.Fatalf("ClaimPendingItems: %d items, err=%v", len(claimed), err)
	}

	body := []byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`)
	outcome, err := srv.DispatchBatchItem(ctx, open.AccountID, open.APIKeyID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != batchRequestFailedCode {
		t.Fatalf("ErrCode=%q body=%s, want %q — the key's allow-list was bypassed",
			outcome.ErrCode, outcome.ResponseBody, batchRequestFailedCode)
	}
	if code := responseErrorCode(outcome.ResponseBody); code != "model_not_allowed" {
		t.Fatalf("error.code=%q body=%s, want model_not_allowed", code, outcome.ResponseBody)
	}

	settled, err := st.FinishItem(store.ItemResult{
		ItemID: claimed[0].ID, Succeeded: false, ErrorCode: outcome.ErrCode,
	}, now)
	if err != nil || !settled {
		t.Fatalf("FinishItem: ok=%v err=%v", settled, err)
	}
	listed, err := st.ListItems(open.ID)
	if err != nil {
		t.Fatalf("ListItems: %v", err)
	}
	if listed[0].State != store.ItemFailed || listed[0].LastErrorCode != batchRequestFailedCode {
		t.Fatalf("item settled as state=%s code=%q, want failed/%s",
			listed[0].State, listed[0].LastErrorCode, batchRequestFailedCode)
	}
	final, _ := st.GetBatch("admin", open.ID)
	if final.CountsFailed != 1 || final.CountsCompleted != 0 {
		t.Fatalf("counts completed=%d failed=%d, want 0/1", final.CountsCompleted, final.CountsFailed)
	}
}
