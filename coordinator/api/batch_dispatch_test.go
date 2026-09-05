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

	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/ratelimit"
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
	srv, reg, st, ctx, done, _ := batchFakeProviderCapturingBody(t, model, usage, content)
	return srv, reg, st, ctx, done
}

// batchFakeProviderCapturingBody is batchFakeProviderWithStore plus the channel
// the fake provider publishes the DECRYPTED inference body on, for the tests
// that assert on what actually reached the provider rather than on what the
// consumer sent.
func batchFakeProviderCapturingBody(t *testing.T, model string, usage protocol.UsageInfo, content string) (
	*Server, *registry.Registry, *store.MemoryStore, context.Context, <-chan struct{}, <-chan []byte,
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
	// Buffered so a test that never reads the body never wedges the provider.
	providerBody := make(chan []byte, 1)
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
		// Publish the plaintext the provider actually received. Decryption uses
		// the provider's own private key, so this is the real sealed frame and
		// not a copy of what the handler intended to send.
		if inferReq.EncryptedBody != nil {
			if value, ok := testProviderKeys.Load(pubKey); ok {
				keypair := value.(testProviderKeyPair)
				plaintext, err := e2e.DecryptWithPrivateKey(&e2e.EncryptedPayload{
					EphemeralPublicKey: inferReq.EncryptedBody.EphemeralPublicKey,
					Ciphertext:         inferReq.EncryptedBody.Ciphertext,
				}, keypair.private)
				if err == nil {
					select {
					case providerBody <- plaintext:
					default:
					}
				}
			}
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

	return srv, reg, st, ctx, done, providerBody
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

// TestDispatchBatchItemAppliesThePerKeyRPMLimit is the S5 regression. The
// per-key RPM override is enforced by the rate-limit MIDDLEWARE, which a batch
// item never passes through — DispatchBatchItem calls handleChatCompletions
// directly. A key throttled to a few requests a minute online was therefore
// unthrottled the moment its traffic arrived on the batch lane.
//
// The throttled item must come back as no_capacity, not request_failed: it
// never reached a provider, so the dispatcher releases the claim without
// charging one of the item's three attempts.
func TestDispatchBatchItemAppliesThePerKeyRPMLimit(t *testing.T) {
	const model = "test-model"
	srv, reg, st, ctx, _ := batchFakeProviderWithStore(t, model,
		protocol.UsageInfo{PromptTokens: 1, CompletionTokens: 1}, "Hello batch")
	srv.SetKeyLimiters(ratelimit.New(ratelimit.Config{RPS: 1, Burst: 1}), nil)

	// One request per minute: applyKeyRPMLimit drives the bucket from the KEY's
	// own rate, so the limiter's own config rate is irrelevant here.
	rpm := int64(1)
	_, key, err := st.CreateAPIKey("admin", store.APIKeyCreate{Name: "batch-rpm", RPMLimit: &rpm})
	if err != nil {
		t.Fatalf("CreateAPIKey: %v", err)
	}

	body := []byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`)
	first, err := srv.DispatchBatchItem(ctx, "admin", key.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem (first): %v", err)
	}
	if first.ErrCode != "" {
		t.Fatalf("first item ErrCode=%q body=%s, want success", first.ErrCode, first.ResponseBody)
	}

	// The fixture's provider answers exactly one request, so close the fleet to
	// the batch lane: every dispatch from here on refuses PROMPTLY, and the two
	// refusals below are told apart by their error code alone.
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

	second, err := srv.DispatchBatchItem(ctx, "admin", key.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem (second): %v", err)
	}
	if second.ErrCode != batchNoCapacityCode {
		t.Fatalf("second item ErrCode=%q body=%s, want %q — a throttled item must be re-offered, not failed",
			second.ErrCode, second.ResponseBody, batchNoCapacityCode)
	}
	if code := responseErrorCode(second.ResponseBody); code != "rate_limit_exceeded" {
		t.Fatalf("error.code=%q body=%s, want rate_limit_exceeded — the key's RPM limit was bypassed",
			code, second.ResponseBody)
	}

	// Control: with the per-key limiter disabled the same key on the same
	// fixture is refused by CAPACITY instead, so the refusal above is the key's
	// own rate limit and not the closed fleet.
	srv.SetKeyLimiters(nil, nil)
	third, err := srv.DispatchBatchItem(ctx, "admin", key.ID, model, body)
	if err != nil {
		t.Fatalf("DispatchBatchItem (third): %v", err)
	}
	if code := responseErrorCode(third.ResponseBody); code != batchNoCapacityCode {
		t.Fatalf("control error.code=%q body=%s, want %q", code, third.ResponseBody, batchNoCapacityCode)
	}
}

// TestRefundBatchItemReversesTheCharge drives a real batch item through the
// funnel against a fake provider — which charges the account for it — and then
// asks RefundBatchItem to give the money back, as the dispatcher does for a
// result it is about to discard. The account must end where it started, and the
// credit must be a LedgerRefund carrying the discarded-item reference.
func TestRefundBatchItemReversesTheCharge(t *testing.T) {
	const model = "test-model"
	const account = "acct-batch-refund"
	usage := protocol.UsageInfo{PromptTokens: 3000, CompletionTokens: 500}
	srv, _, st, ctx, providerDone := batchFakeProviderWithStore(t, model, usage, "Hello batch")

	if err := st.Credit(account, 100_000_000, store.LedgerDeposit, "seed"); err != nil {
		t.Fatalf("seed balance: %v", err)
	}
	opening := st.GetBalance(account)
	providerKey := testPublicKeyB64()
	earningsBefore, err := st.GetProviderEarningsSummary(providerKey)
	if err != nil {
		t.Fatalf("provider earnings summary: %v", err)
	}

	outcome, err := srv.DispatchBatchItem(ctx, account, "", model,
		[]byte(`{"model":"ignored","messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("DispatchBatchItem: %v", err)
	}
	if outcome.ErrCode != "" {
		t.Fatalf("ErrCode=%q body=%s, want success", outcome.ErrCode, outcome.ResponseBody)
	}
	<-providerDone

	charged := opening - st.GetBalance(account)
	if charged <= 0 {
		t.Fatalf("the funnel charged %d micro-USD — a refund would prove nothing", charged)
	}

	if err := srv.RefundBatchItem(account, outcome.RequestID,
		outcome.PromptTokens, outcome.CompletionTokens); err != nil {
		t.Fatalf("RefundBatchItem: %v", err)
	}
	if got := st.GetBalance(account); got != opening {
		t.Fatalf("balance after refund = %d, want the opening %d (charged %d)", got, opening, charged)
	}

	var refund *store.LedgerEntry
	entries := st.LedgerHistory(account)
	for i := range entries {
		if entries[i].Reference == batchDiscardedRefundPrefix+outcome.RequestID {
			refund = &entries[i]
			break
		}
	}
	if refund == nil {
		t.Fatalf("no ledger entry referencing %q", batchDiscardedRefundPrefix+outcome.RequestID)
	}
	if refund.Type != store.LedgerRefund {
		t.Fatalf("refund entry type = %q, want %q", refund.Type, store.LedgerRefund)
	}
	if refund.AmountMicroUSD != charged {
		t.Fatalf("refund = %d micro-USD, want the %d that was charged", refund.AmountMicroUSD, charged)
	}

	// The provider's payout is deliberately NOT clawed back: it served the
	// request. Dropping the answer is the coordinator's doing, so the platform
	// absorbs the difference (see RefundBatchItem).
	earningsAfter, err := st.GetProviderEarningsSummary(providerKey)
	if err != nil {
		t.Fatalf("provider earnings summary: %v", err)
	}
	if earningsAfter != earningsBefore {
		t.Fatalf("provider earnings moved on a consumer refund: %+v -> %+v", earningsBefore, earningsAfter)
	}
}
