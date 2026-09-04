package batchlane

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

const (
	testAccount = "acct_batchlane"
	testModel   = "mlx-community/gemma-4-e2b-it-4bit"
)

var testStart = time.Date(2026, 9, 5, 12, 0, 0, 0, time.UTC)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

func testBlobs(t *testing.T) *sealedblob.Store {
	t.Helper()
	key, err := sealedblob.RandomKey()
	if err != nil {
		t.Fatalf("random key: %v", err)
	}
	bs, err := sealedblob.New(t.TempDir(), key)
	if err != nil {
		t.Fatalf("blob store: %v", err)
	}
	return bs
}

// itemBody is the sealed request body of one line: the plain OpenAI chat body,
// which is what PR2's JSONL parser stores under the item's blob ref.
func itemBody(line int) []byte {
	return []byte(fmt.Sprintf(
		`{"model":%q,"messages":[{"role":"user","content":"line-%d"}]}`, testModel, line))
}

// seedBatch creates an in_progress batch of n items with their bodies sealed.
func seedBatch(t *testing.T, st store.Store, blobs *sealedblob.Store, id string, n int, expiresAt time.Time) *store.Batch {
	t.Helper()
	return seedBatchKeyed(t, st, blobs, id, n, expiresAt, "")
}

// seedBatchKeyed is seedBatch for a batch whose results the consumer wants
// sealed to its own key.
func seedBatchKeyed(t *testing.T, st store.Store, blobs *sealedblob.Store, id string, n int, expiresAt time.Time, resultPublicKey string) *store.Batch {
	t.Helper()
	b := &store.Batch{
		ID:               id,
		AccountID:        testAccount,
		InputFileID:      "file-" + id,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: "24h",
		CreatedAt:        testStart,
		ExpiresAt:        expiresAt,
		CountsTotal:      n,
		SealedTo:         "coordinator",
		Source:           "file",
		ResultPublicKey:  resultPublicKey,
	}
	if resultPublicKey != "" {
		b.SealedTo = "consumer"
	}
	items := make([]*store.BatchItem, 0, n)
	for i := 0; i < n; i++ {
		itemID := fmt.Sprintf("bitem_%s_%02d", id, i)
		if err := blobs.PutPlain(itemID, itemBody(i)); err != nil {
			t.Fatalf("seal item body: %v", err)
		}
		items = append(items, &store.BatchItem{
			ID:       itemID,
			BatchID:  id,
			CustomID: fmt.Sprintf("cid-%02d", i),
			LineNo:   i + 1,
			State:    store.ItemPending,
			BlobRef:  itemID,
		})
	}
	if err := st.CreateBatch(b, items); err != nil {
		t.Fatalf("create batch: %v", err)
	}
	if ok, err := st.SetBatchStatus(id, store.BatchValidating, store.BatchInProgress, testStart); err != nil || !ok {
		t.Fatalf("start batch: ok=%v err=%v", ok, err)
	}
	got, _ := st.GetBatch(testAccount, id)
	return got
}

// idleSlot is one provider·model slot with room for maxPerSlot batch rows.
func idleSlot(v *FakeView, maxPerSlot int) SlotKey {
	key := SlotKey{ProviderID: "P1", Model: testModel}
	v.Set(key, SlotSignal{DecodeTPS: 40, DecodeFloor: 15, KV: 0.20, MaxPerSlot: maxPerSlot})
	return key
}

type harness struct {
	st       store.Store
	blobs    *sealedblob.Store
	view     *FakeView
	dispatch *FakeDispatch
	finalize *FakeFinalize
	d        *Dispatcher
}

func newHarness(t *testing.T, cfg Config) *harness {
	t.Helper()
	h := &harness{
		st:       store.NewMemory(store.Config{}),
		blobs:    testBlobs(t),
		view:     NewFakeView(),
		dispatch: &FakeDispatch{},
		finalize: &FakeFinalize{},
	}
	h.d = New(h.st, h.blobs, h.view, h.dispatch.Fn(), h.finalize.Fn(), cfg, testLogger())
	return h
}

// tick runs one iteration and waits for the dispatch goroutines it started, so
// a test can assert on the state a settle will see on the next tick.
func (h *harness) tick(t *testing.T, ctx context.Context, now time.Time) {
	t.Helper()
	h.d.Tick(ctx, now)
	h.d.awaitDispatch()
}

func itemStates(t *testing.T, st store.Store, batchID string) map[store.ItemState]int {
	t.Helper()
	items, err := st.ListItems(batchID)
	if err != nil {
		t.Fatalf("list items: %v", err)
	}
	out := map[store.ItemState]int{}
	for _, it := range items {
		out[it.State]++
	}
	return out
}

func TestTickClaimsUpToTargetPerSlot(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 3)
	// Block every dispatch so claimed items stay in flight and the budget the
	// next tick sees is the target minus what is already out.
	h.dispatch.Block = make(chan struct{})
	seedBatch(t, h.st, h.blobs, "batch_target", 10, testStart.Add(24*time.Hour))

	// The AIMD target starts at 0 and grows by one per healthy tick, so tick 1
	// dispatches exactly one item.
	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	h.d.Tick(ctx, testStart.Add(time.Second))
	waitForCalls(t, h.dispatch, 2)

	h.d.Tick(ctx, testStart.Add(2*time.Second))
	waitForCalls(t, h.dispatch, 3)

	// Target is now pinned at MaxPerSlot 3 and all three rows are occupied, so
	// a fourth tick claims nothing.
	h.d.Tick(ctx, testStart.Add(3*time.Second))
	if n := h.dispatch.Len(); n != 3 {
		t.Fatalf("dispatch calls = %d, want 3 (MaxPerSlot)", n)
	}
	if states := itemStates(t, h.st, "batch_target"); states[store.ItemInflight] != 3 {
		t.Fatalf("inflight = %d, want 3: %v", states[store.ItemInflight], states)
	}

	close(h.dispatch.Block)
	h.d.awaitDispatch()
}

func TestTickBacksOffWhenOnlineArrives(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	key := idleSlot(h.view, 8)
	h.dispatch.Block = make(chan struct{})
	seedBatch(t, h.st, h.blobs, "batch_backoff", 20, testStart.Add(24*time.Hour))

	// Walk the target up to 4 over four healthy ticks.
	for i := 0; i < 4; i++ {
		h.d.Tick(ctx, testStart.Add(time.Duration(i)*time.Second))
		waitForCalls(t, h.dispatch, i+1)
	}
	if got := h.d.SlotTarget(key); got != 4 {
		t.Fatalf("target = %d, want 4", got)
	}

	// Online traffic queues on the slot: the target halves and, with four rows
	// already in flight, nothing new is claimed.
	h.view.Update(key, func(s *SlotSignal) { s.Waiting = 1 })
	h.d.Tick(ctx, testStart.Add(4*time.Second))
	if got := h.d.SlotTarget(key); got != 2 {
		t.Fatalf("target after a waiting row = %d, want 2", got)
	}
	if n := h.dispatch.Len(); n != 4 {
		t.Fatalf("dispatch calls = %d, want 4 (no new claims while over target)", n)
	}

	close(h.dispatch.Block)
	h.d.awaitDispatch()
}

func TestTickFinishesItemsAndFinalizes(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(n int, model string, body []byte) (Outcome, error) {
		return Outcome{
			RequestID:        fmt.Sprintf("req-%d", n),
			PromptTokens:     11,
			CompletionTokens: 22,
			ResponseBody:     []byte(`{"choices":[{"message":{"content":"hi"}}]}`),
		}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_done", 1, testStart.Add(24*time.Hour))

	h.tick(t, ctx, testStart)                  // claim + dispatch
	h.tick(t, ctx, testStart.Add(time.Second)) // drain + settle

	items, _ := h.st.ListItems("batch_done")
	it := items[0]
	if it.State != store.ItemSucceeded {
		t.Fatalf("state = %s, want succeeded", it.State)
	}
	if it.PromptTokens != 11 || it.CompletionTokens != 22 {
		t.Fatalf("tokens = %d/%d, want 11/22", it.PromptTokens, it.CompletionTokens)
	}
	if it.RequestID != "req-0" {
		t.Fatalf("request id = %q, want req-0", it.RequestID)
	}
	if it.ResultBlobRef == "" {
		t.Fatal("result blob ref is empty")
	}
	got, err := h.blobs.Open(it.ResultBlobRef)
	if err != nil {
		t.Fatalf("open result blob: %v", err)
	}
	if !bytes.Contains(got, []byte(`"content":"hi"`)) {
		t.Fatalf("result blob does not hold the response body")
	}

	b, _ := h.st.GetBatch(testAccount, "batch_done")
	if b.CountsCompleted != 1 || b.CountsFailed != 0 {
		t.Fatalf("counts = %d/%d, want 1/0", b.CountsCompleted, b.CountsFailed)
	}
	if !h.finalize.Called("batch_done") {
		t.Fatal("finalize was not called for a settled batch")
	}
}

func TestSuccessSealsToTheConsumerKeyWhenSet(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (Outcome, error) {
		return Outcome{ResponseBody: []byte(`{"secret":"consumer-only"}`)}, nil
	}

	consumer, err := e2e.GenerateSessionKeys()
	if err != nil {
		t.Fatalf("consumer keys: %v", err)
	}
	seedBatchKeyed(t, h.st, h.blobs, "batch_sealed", 1, testStart.Add(24*time.Hour),
		base64.StdEncoding.EncodeToString(consumer.PublicKey[:]))

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(time.Second))

	items, _ := h.st.ListItems("batch_sealed")
	if items[0].State != store.ItemSucceeded {
		t.Fatalf("state = %s, want succeeded", items[0].State)
	}
	if _, err := h.blobs.Open(items[0].ResultBlobRef); err == nil {
		t.Fatal("the coordinator can still open a result sealed to the consumer key")
	}
	raw, err := h.blobs.Raw(items[0].ResultBlobRef)
	if err != nil {
		t.Fatalf("raw: %v", err)
	}
	if bytes.Contains(raw, []byte("consumer-only")) {
		t.Fatal("the sealed result blob contains plaintext")
	}
}

func TestTickRetriesThenFails(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (Outcome, error) {
		return Outcome{}, errors.New("provider exploded")
	}
	seedBatch(t, h.st, h.blobs, "batch_retry", 1, testStart.Add(24*time.Hour))

	// Each round is one claim + dispatch and one drain + settle.
	for i := 0; i < 3; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(2*i)*time.Second))
		h.tick(t, ctx, testStart.Add(time.Duration(2*i+1)*time.Second))
	}
	if n := h.dispatch.Len(); n != 3 {
		t.Fatalf("dispatch calls = %d, want 3 (MaxAttempts)", n)
	}

	items, _ := h.st.ListItems("batch_retry")
	if items[0].State != store.ItemFailed {
		t.Fatalf("state = %s, want failed", items[0].State)
	}
	if items[0].LastErrorCode != ErrCodeRequestFailed {
		t.Fatalf("error code = %q, want %q", items[0].LastErrorCode, ErrCodeRequestFailed)
	}
	b, _ := h.st.GetBatch(testAccount, "batch_retry")
	if b.CountsFailed != 1 {
		t.Fatalf("counts_failed = %d, want 1", b.CountsFailed)
	}
}

func TestNoCapacityReleasesClaim(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (Outcome, error) {
		return Outcome{ErrCode: ErrCodeNoCapacity}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_nocap", 1, testStart.Add(24*time.Hour))

	for i := 0; i < 6; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(i)*time.Second))
	}
	// Take the slot away so the last tick only drains and cannot re-claim.
	h.view.Remove(key)
	h.tick(t, ctx, testStart.Add(6*time.Second))

	items, _ := h.st.ListItems("batch_nocap")
	if items[0].State != store.ItemPending {
		t.Fatalf("state = %s, want pending", items[0].State)
	}
	if items[0].Attempts != 0 {
		t.Fatalf("attempts = %d, want 0 — a claim that found no capacity is not an attempt", items[0].Attempts)
	}
}

// A dispatch the funnel reports as cancelled (coordinator shutdown, or the
// batch's own context) is accounted exactly like no_capacity: back to pending,
// no attempt charged.
func TestCancelledOutcomeReleasesWithoutAnAttempt(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (Outcome, error) {
		return Outcome{ErrCode: ErrCodeCancelled}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_cancelledrc", 1, testStart.Add(24*time.Hour))

	for i := 0; i < 6; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(i)*time.Second))
	}
	h.view.Remove(key)
	h.tick(t, ctx, testStart.Add(6*time.Second))

	items, _ := h.st.ListItems("batch_cancelledrc")
	if items[0].State != store.ItemPending {
		t.Fatalf("state = %s, want pending", items[0].State)
	}
	if items[0].Attempts != 0 {
		t.Fatalf("attempts = %d, want 0", items[0].Attempts)
	}
	if h.dispatch.Len() < 3 {
		t.Fatalf("dispatch calls = %d, want the item retried without exhausting attempts", h.dispatch.Len())
	}
}

func TestSweepExpiresPastDeadline(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Block = make(chan struct{})
	expires := testStart.Add(30 * time.Second)
	seedBatch(t, h.st, h.blobs, "batch_expire", 3, expires)

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	// Past the deadline: the in-flight item's context is cancelled, every open
	// item is expired and the batch follows.
	h.d.Tick(ctx, expires.Add(time.Second))
	close(h.dispatch.Block)
	h.d.awaitDispatch()

	b, _ := h.st.GetBatch(testAccount, "batch_expire")
	if b.Status != store.BatchExpired {
		t.Fatalf("status = %s, want expired", b.Status)
	}
	states := itemStates(t, h.st, "batch_expire")
	if states[store.ItemExpired] != 3 {
		t.Fatalf("expired items = %d, want 3: %v", states[store.ItemExpired], states)
	}
	if b.CountsCompleted != 0 || b.CountsFailed != 0 {
		t.Fatalf("counts = %d/%d, want 0/0 — expiry moves neither", b.CountsCompleted, b.CountsFailed)
	}
	if !h.finalize.Called("batch_expire") {
		t.Fatal("an expired batch was not finalized")
	}

	// A result that lands after the sweep is ignored.
	h.d.Tick(ctx, expires.Add(2*time.Second))
	if states := itemStates(t, h.st, "batch_expire"); states[store.ItemExpired] != 3 {
		t.Fatalf("late result changed item states: %v", states)
	}
}

func TestRestartRequeuesInflight(t *testing.T) {
	st := store.NewMemory(store.Config{})
	blobs := testBlobs(t)
	seedBatch(t, st, blobs, "batch_restart", 4, testStart.Add(24*time.Hour))
	claimed, err := st.ClaimPendingItems("batch_restart", 2, testStart)
	if err != nil || len(claimed) != 2 {
		t.Fatalf("claim: %d items, err %v", len(claimed), err)
	}

	// A fresh dispatcher is a coordinator that just restarted: the rows it left
	// inflight belong to a process that no longer exists.
	New(st, blobs, NewFakeView(), (&FakeDispatch{}).Fn(), (&FakeFinalize{}).Fn(),
		Config{MaxAttempts: 3}, testLogger())

	states := itemStates(t, st, "batch_restart")
	if states[store.ItemInflight] != 0 || states[store.ItemPending] != 4 {
		t.Fatalf("after restart: %v, want 4 pending and 0 inflight", states)
	}
	items, _ := st.ListItems("batch_restart")
	requeued := 0
	for _, it := range items {
		if it.Attempts == 1 {
			requeued++
		}
	}
	if requeued != 2 {
		t.Fatalf("%d items kept their attempt, want 2 — a requeue is not a released claim", requeued)
	}
}

func TestCancelStopsInflightWork(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	release := make(chan struct{})
	h.dispatch.Block = release
	seedBatch(t, h.st, h.blobs, "batch_cancel", 2, testStart.Add(24*time.Hour))

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	// The consumer cancels. PR2's handler moves the batch to cancelling; the
	// dispatcher drains it.
	if ok, err := h.st.SetBatchStatus("batch_cancel", store.BatchInProgress, store.BatchCancelling, testStart.Add(time.Second)); err != nil || !ok {
		t.Fatalf("cancel: ok=%v err=%v", ok, err)
	}
	h.d.Tick(ctx, testStart.Add(2*time.Second))

	// The in-flight dispatch saw its context cancelled without the test ever
	// closing the block channel.
	h.d.awaitDispatch()
	select {
	case <-release:
		t.Fatal("the dispatch returned because the test released it, not because its context was cancelled")
	default:
	}

	states := itemStates(t, h.st, "batch_cancel")
	if states[store.ItemCancelled] != 2 {
		t.Fatalf("cancelled items = %d, want 2: %v", states[store.ItemCancelled], states)
	}

	// The late outcome is drained on the next tick and changes nothing.
	h.d.Tick(ctx, testStart.Add(3*time.Second))
	if states := itemStates(t, h.st, "batch_cancel"); states[store.ItemCancelled] != 2 {
		t.Fatalf("a late outcome moved a cancelled item: %v", states)
	}
	b, _ := h.st.GetBatch(testAccount, "batch_cancel")
	if b.Status != store.BatchCancelled {
		t.Fatalf("status = %s, want cancelled", b.Status)
	}
	if b.CountsCompleted != 0 || b.CountsFailed != 0 {
		t.Fatalf("counts = %d/%d, want 0/0", b.CountsCompleted, b.CountsFailed)
	}
}

// A batch whose slack has run out is granted a progress floor of one in-flight
// item even while every slot's AIMD target is zero, so a busy online tenant
// cannot starve it to expiry.
func TestFloorGuaranteesProgressWhenUrgent(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	// Online traffic is queueing, so the AIMD target is pinned at zero.
	h.view.Update(key, func(s *SlotSignal) { s.Waiting = 2 })
	h.dispatch.Block = make(chan struct{})
	// Ten minutes of slack against a six-hour horizon: urgency 0.97, past the
	// 0.9 floor, however few items are left.
	seedBatch(t, h.st, h.blobs, "batch_urgent", 5, testStart.Add(10*time.Minute))

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)
	if got := h.d.SlotTarget(key); got != 0 {
		t.Fatalf("slot target = %d, want 0 — the floor must not raise the AIMD target", got)
	}

	// The bucket refills at 0.2/s, so a second item is not granted one second
	// later but is five seconds later.
	h.d.Tick(ctx, testStart.Add(time.Second))
	if n := h.dispatch.Len(); n != 1 {
		t.Fatalf("dispatch calls = %d, want 1 — the floor is rate limited", n)
	}
	h.d.Tick(ctx, testStart.Add(6*time.Second))
	waitForCalls(t, h.dispatch, 2)

	close(h.dispatch.Block)
	h.d.awaitDispatch()
}

// A healthy batch gets no floor: escalation is for batches that are actually
// going to miss their window.
func TestFloorDoesNotApplyToAHealthyBatch(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.view.Update(key, func(s *SlotSignal) { s.Waiting = 2 })
	seedBatch(t, h.st, h.blobs, "batch_healthy", 3, testStart.Add(24*time.Hour))

	for i := 0; i < 5; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(i)*time.Second))
	}
	if n := h.dispatch.Len(); n != 0 {
		t.Fatalf("dispatch calls = %d, want 0 — an on-track batch waits for capacity", n)
	}
}

// Priority orders claims across batches: the urgent one is served first when
// the budget only covers one item.
func TestClaimsAreOrderedByPriority(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 1)
	h.dispatch.Block = make(chan struct{})

	// Created in the "wrong" order: the healthy batch first.
	seedBatch(t, h.st, h.blobs, "batch_relaxed", 5, testStart.Add(24*time.Hour))
	seedBatch(t, h.st, h.blobs, "batch_tight", 5, testStart.Add(20*time.Minute))

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	if states := itemStates(t, h.st, "batch_tight"); states[store.ItemInflight] != 1 {
		t.Fatalf("the urgent batch got %v, want one inflight item", states)
	}
	if states := itemStates(t, h.st, "batch_relaxed"); states[store.ItemInflight] != 0 {
		t.Fatalf("the on-track batch got %v, want nothing", states)
	}

	close(h.dispatch.Block)
	h.d.awaitDispatch()
}

// Nothing is claimed from a batch that is not in_progress, so a cancelling or
// validating batch drains rather than dispatching.
func TestNoClaimsFromABatchThatIsNotInProgress(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	seedBatch(t, h.st, h.blobs, "batch_cancelling", 3, testStart.Add(24*time.Hour))
	if ok, _ := h.st.SetBatchStatus("batch_cancelling", store.BatchInProgress, store.BatchCancelling, testStart); !ok {
		t.Fatal("could not move the batch to cancelling")
	}

	h.tick(t, ctx, testStart)
	if n := h.dispatch.Len(); n != 0 {
		t.Fatalf("dispatch calls = %d, want 0", n)
	}
}

// The dispatcher hands the funnel the item's plaintext and keeps nothing:
// after the tick the only copies of the body are the two sealed blobs on disk.
func TestPlaintextIsNotRetained(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(_ int, model string, body []byte) (Outcome, error) {
		if model != testModel {
			t.Errorf("model = %q, want %q", model, testModel)
		}
		if !bytes.Contains(body, []byte("line-0")) {
			t.Errorf("the funnel did not receive the item body")
		}
		return Outcome{ResponseBody: []byte(`{"choices":[]}`)}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_plain", 1, testStart.Add(24*time.Hour))

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(time.Second))

	if n := h.d.InflightItems(); n != 0 {
		t.Fatalf("dispatcher still tracks %d in-flight items", n)
	}
	// Nothing on disk is plaintext: neither the request nor the result.
	entries, err := os.ReadDir(h.blobs.Dir())
	if err != nil {
		t.Fatalf("read blob dir: %v", err)
	}
	if len(entries) != 2 {
		t.Fatalf("blob dir holds %d files, want 2 (request + result)", len(entries))
	}
	for _, e := range entries {
		raw, err := os.ReadFile(filepath.Join(h.blobs.Dir(), e.Name()))
		if err != nil {
			t.Fatalf("read blob: %v", err)
		}
		if bytes.Contains(raw, []byte("line-0")) || bytes.Contains(raw, []byte(`"choices"`)) {
			t.Fatalf("blob %s holds plaintext", e.Name())
		}
	}
}

// waitForCalls blocks until the fake funnel has seen at least n calls. Dispatch
// runs in goroutines, so the count a tick produces is not visible the moment
// Tick returns.
func waitForCalls(t *testing.T, f *FakeDispatch, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if f.Len() >= n {
			return
		}
		time.Sleep(time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d dispatch calls (have %d)", n, f.Len())
}
