package batchlane_test

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
	"sync"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
	"github.com/eigeninference/d-inference/coordinator/batchlane/batchlanetest"
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
	return seedBatchWith(t, st, blobs, id, n, expiresAt, resultPublicKey, "")
}

// seedBatchWith is the full seeder: resultPublicKey selects consumer sealing,
// apiKeyID is the key the batch was submitted with ("" for a caller that had
// none).
func seedBatchWith(
	t *testing.T,
	st store.Store,
	blobs *sealedblob.Store,
	id string,
	n int,
	expiresAt time.Time,
	resultPublicKey, apiKeyID string,
) *store.Batch {
	t.Helper()
	b := &store.Batch{
		ID:               id,
		AccountID:        testAccount,
		APIKeyID:         apiKeyID,
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
		inputRef := itemID + "-in" // api.BatchItemInputRef
		if err := blobs.PutPlain(inputRef, itemBody(i)); err != nil {
			t.Fatalf("seal item body: %v", err)
		}
		items = append(items, &store.BatchItem{
			ID:       itemID,
			BatchID:  id,
			CustomID: fmt.Sprintf("cid-%02d", i),
			LineNo:   i + 1,
			State:    store.ItemPending,
			BlobRef:  inputRef,
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
func idleSlot(v *batchlanetest.FakeView, maxPerSlot int) batchlane.SlotKey {
	key := batchlane.SlotKey{ProviderID: "P1", Model: testModel}
	v.Set(key, batchlane.SlotSignal{DecodeTPS: 40, DecodeFloor: 15, KV: 0.20, KVKnown: true, MaxPerSlot: maxPerSlot})
	return key
}

type harness struct {
	st       store.Store
	blobs    *sealedblob.Store
	view     *batchlanetest.FakeView
	dispatch *batchlanetest.FakeDispatch
	finalize *batchlanetest.FakeFinalize
	d        *batchlane.Dispatcher
}

func newHarness(t *testing.T, cfg batchlane.Config) *harness {
	t.Helper()
	return newHarnessOn(t, cfg, store.NewMemory(store.Config{}))
}

// newHarnessOn is newHarness over a caller-supplied store, so a test can wrap
// the memory store in a probe before the dispatcher ever reads it.
func newHarnessOn(t *testing.T, cfg batchlane.Config, st store.Store) *harness {
	t.Helper()
	h := &harness{
		st:       st,
		blobs:    testBlobs(t),
		view:     batchlanetest.NewFakeView(),
		dispatch: &batchlanetest.FakeDispatch{},
		finalize: &batchlanetest.FakeFinalize{},
	}
	h.d = batchlane.New(h.st, h.blobs, h.view, h.dispatch.Fn(), h.finalize.Fn(), cfg, testLogger())
	return h
}

// tick runs one iteration and waits for the dispatch goroutines it started, so
// a test can assert on the state a settle will see on the next tick.
func (h *harness) tick(t *testing.T, ctx context.Context, now time.Time) {
	t.Helper()
	h.d.Tick(ctx, now)
	h.d.AwaitDispatch()
}

// assemblerFinalize is the part of api's FinalizeBatchIfDone the dispatcher's
// sweep now depends on: once a batch has no open items left, finalize — not the
// sweep — performs the terminal transition. Expired items make the target
// expired, a cancelling batch becomes cancelled, everything else completes.
func assemblerFinalize(st store.Store) func(string, time.Time) error {
	return func(batchID string, now time.Time) error {
		b, ok := st.GetBatchByID(batchID)
		if !ok {
			return nil
		}
		_, pending, inflight, _, _, err := st.CountItems(batchID)
		if err != nil {
			return err
		}
		if pending+inflight > 0 {
			return nil
		}
		to := store.BatchCompleted
		switch b.Status {
		case store.BatchCancelling:
			to = store.BatchCancelled
		case store.BatchInProgress:
			items, err := st.ListItems(batchID)
			if err != nil {
				return err
			}
			for _, it := range items {
				if it.State == store.ItemExpired {
					to = store.BatchExpired
					break
				}
			}
		default:
			return nil
		}
		_, err = st.SetBatchStatus(batchID, b.Status, to, now)
		return err
	}
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
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
	h.d.AwaitDispatch()
}

func TestTickBacksOffWhenOnlineArrives(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
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
	h.view.Update(key, func(s *batchlane.SlotSignal) { s.Waiting = 1 })
	h.d.Tick(ctx, testStart.Add(4*time.Second))
	if got := h.d.SlotTarget(key); got != 2 {
		t.Fatalf("target after a waiting row = %d, want 2", got)
	}
	if n := h.dispatch.Len(); n != 4 {
		t.Fatalf("dispatch calls = %d, want 4 (no new claims while over target)", n)
	}

	close(h.dispatch.Block)
	h.d.AwaitDispatch()
}

func TestTickFinishesItemsAndFinalizes(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(n int, model string, body []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{
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
	if it.ResultBlobRef != batchlane.ResultBlobRef(it.ID) {
		t.Fatalf("result blob ref = %q, want %q", it.ResultBlobRef, batchlane.ResultBlobRef(it.ID))
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ResponseBody: []byte(`{"secret":"consumer-only"}`)}, nil
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{}, errors.New("provider exploded")
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
	if items[0].LastErrorCode != batchlane.ErrCodeRequestFailed {
		t.Fatalf("error code = %q, want %q", items[0].LastErrorCode, batchlane.ErrCodeRequestFailed)
	}
	b, _ := h.st.GetBatch(testAccount, "batch_retry")
	if b.CountsFailed != 1 {
		t.Fatalf("counts_failed = %d, want 1", b.CountsFailed)
	}
}

func TestNoCapacityReleasesClaim(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ErrCode: batchlane.ErrCodeNoCapacity}, nil
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ErrCode: batchlane.ErrCodeCancelled}, nil
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Block = make(chan struct{})
	// The sweep expires the items and hands the terminal transition to
	// finalize, which is where the assembled files are attached first.
	h.finalize.Then = assemblerFinalize(h.st)
	expires := testStart.Add(30 * time.Second)
	seedBatch(t, h.st, h.blobs, "batch_expire", 3, expires)

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	// Past the deadline: the in-flight item's context is cancelled, every open
	// item is expired and the batch follows.
	h.d.Tick(ctx, expires.Add(time.Second))
	close(h.dispatch.Block)
	h.d.AwaitDispatch()

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
	batchlane.New(st, blobs, batchlanetest.NewFakeView(), (&batchlanetest.FakeDispatch{}).Fn(), (&batchlanetest.FakeFinalize{}).Fn(),
		batchlane.Config{MaxAttempts: 3}, testLogger())

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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	release := make(chan struct{})
	h.dispatch.Block = release
	h.finalize.Then = assemblerFinalize(h.st)
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
	h.d.AwaitDispatch()
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	// Online traffic is queueing, so the AIMD target is pinned at zero.
	h.view.Update(key, func(s *batchlane.SlotSignal) { s.Waiting = 2 })
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
	h.d.AwaitDispatch()
}

// A healthy batch gets no floor: escalation is for batches that are actually
// going to miss their window.
func TestFloorDoesNotApplyToAHealthyBatch(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	h.view.Update(key, func(s *batchlane.SlotSignal) { s.Waiting = 2 })
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
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
	h.d.AwaitDispatch()
}

// Nothing is claimed from a batch that is not in_progress, so a cancelling or
// validating batch drains rather than dispatching.
func TestNoClaimsFromABatchThatIsNotInProgress(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
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
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(_ int, model string, body []byte) (batchlane.Outcome, error) {
		if model != testModel {
			t.Errorf("model = %q, want %q", model, testModel)
		}
		if !bytes.Contains(body, []byte("line-0")) {
			t.Errorf("the funnel did not receive the item body")
		}
		return batchlane.Outcome{ResponseBody: []byte(`{"choices":[]}`)}, nil
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
func waitForCalls(t *testing.T, f *batchlanetest.FakeDispatch, n int) {
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

// TestDispatchCarriesTheBatchesAPIKey: the batch row's submitting key is what
// the funnel is called with, so the key's AllowedModels and spend cap apply to
// every item. Before PR3c the dispatcher passed "" and key-level enforcement
// was silently skipped for all batch traffic.
func TestDispatchCarriesTheBatchesAPIKey(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	const keyID = "key_submitter"
	seedBatchWith(t, h.st, h.blobs, "batch_keyed", 1, testStart.Add(24*time.Hour), "", keyID)

	h.tick(t, ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	calls := h.dispatch.Calls()
	if calls[0].APIKeyID != keyID {
		t.Fatalf("dispatch APIKeyID = %q, want %q — the batch's key was dropped", calls[0].APIKeyID, keyID)
	}
	if calls[0].AccountID != testAccount {
		t.Fatalf("dispatch AccountID = %q, want %q", calls[0].AccountID, testAccount)
	}
}

// A batch created by a caller with no API key (Privy JWT session, admin key)
// dispatches with "" and runs under account-level limits only — the same thing
// that caller's online requests do.
func TestDispatchCarriesAnEmptyAPIKeyForKeylessBatches(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	seedBatch(t, h.st, h.blobs, "batch_keyless", 1, testStart.Add(24*time.Hour))

	h.tick(t, ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	if got := h.dispatch.Calls()[0].APIKeyID; got != "" {
		t.Fatalf("dispatch APIKeyID = %q, want \"\"", got)
	}
}

// A non-nil error carrying request_failed is the funnel reporting a permanent
// condition (an unusable API key, a body that cannot be parsed): the item is
// failed on the first outcome rather than burning its whole attempt budget.
func TestPermanentFailureIsNotRetried(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ErrCode: batchlane.ErrCodeRequestFailed}, errors.New("api key revoked")
	}
	seedBatch(t, h.st, h.blobs, "batch_permanent", 1, testStart.Add(24*time.Hour))

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(time.Second))

	if n := h.dispatch.Len(); n != 1 {
		t.Fatalf("dispatch calls = %d, want 1 — a permanent failure is not retried", n)
	}
	items, _ := h.st.ListItems("batch_permanent")
	if items[0].State != store.ItemFailed {
		t.Fatalf("state = %s, want failed", items[0].State)
	}
	if items[0].LastErrorCode != batchlane.ErrCodeRequestFailed {
		t.Fatalf("error code = %q, want %q", items[0].LastErrorCode, batchlane.ErrCodeRequestFailed)
	}
}

// An item whose sealed body is gone can never succeed, so it fails at once
// rather than being re-offered until its attempts run out.
func TestMissingItemBodyFailsPermanently(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	seedBatch(t, h.st, h.blobs, "batch_noblob", 1, testStart.Add(24*time.Hour))
	items, _ := h.st.ListItems("batch_noblob")
	if err := h.blobs.Delete(items[0].BlobRef); err != nil {
		t.Fatalf("delete input blob: %v", err)
	}

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(time.Second))

	if n := h.dispatch.Len(); n != 0 {
		t.Fatalf("dispatch calls = %d, want 0 — the body never opened", n)
	}
	items, _ = h.st.ListItems("batch_noblob")
	if items[0].State != store.ItemFailed {
		t.Fatalf("state = %s, want failed", items[0].State)
	}
}

// Once a batch has left the open list its result blobs start a retention clock;
// the sweep deletes them when it runs out, and runs the file retention pass on
// its own cadence.
func TestRetentionPurgesResultBlobsAndFiles(t *testing.T) {
	ctx := context.Background()
	purges := 0
	h := newHarness(t, batchlane.Config{
		MaxAttempts:     3,
		OutputRetention: time.Hour,
		PurgeInterval:   10 * time.Minute,
		Purge:           func(time.Time) (int, error) { purges++; return 0, nil },
	})
	idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ResponseBody: []byte(`{"choices":[]}`)}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_retain", 1, testStart.Add(24*time.Hour))

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(time.Second))

	items, _ := h.st.ListItems("batch_retain")
	ref := items[0].ResultBlobRef
	if _, err := h.blobs.Raw(ref); err != nil {
		t.Fatalf("result blob missing right after settle: %v", err)
	}

	// PR2's finalize takes the batch terminal. The fake hook does not, so drive
	// the same transition the dispatcher would observe.
	if ok, err := h.st.SetBatchStatus("batch_retain", store.BatchInProgress, store.BatchCompleted, testStart.Add(2*time.Second)); err != nil || !ok {
		t.Fatalf("complete batch: ok=%v err=%v", ok, err)
	}
	h.tick(t, ctx, testStart.Add(2*time.Second)) // schedules retention
	if _, err := h.blobs.Raw(ref); err != nil {
		t.Fatal("the result blob was purged before its retention expired")
	}

	h.tick(t, ctx, testStart.Add(2*time.Hour))
	if _, err := h.blobs.Raw(ref); !errors.Is(err, sealedblob.ErrNotFound) {
		t.Fatalf("result blob after retention: err = %v, want ErrNotFound", err)
	}
	if purges == 0 {
		t.Fatal("the file retention pass never ran")
	}
}

// The purge hook runs on its interval, not on every tick.
func TestFileRetentionPassIsRateLimited(t *testing.T) {
	ctx := context.Background()
	purges := 0
	h := newHarness(t, batchlane.Config{
		MaxAttempts:   3,
		PurgeInterval: time.Minute,
		Purge:         func(time.Time) (int, error) { purges++; return 0, nil },
	})

	for i := 0; i < 10; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(i)*time.Second))
	}
	if purges != 1 {
		t.Fatalf("purge ran %d times in ten seconds, want 1", purges)
	}
	h.tick(t, ctx, testStart.Add(2*time.Minute))
	if purges != 2 {
		t.Fatalf("purge ran %d times, want 2 after the interval elapsed", purges)
	}
}

// A crash between sealing an item body and committing its rows leaves a blob
// nothing references. The orphan sweep is the only thing that can find one.
func TestOrphanItemBlobsArePurged(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{
		MaxAttempts:     3,
		OutputRetention: time.Hour,
		OrphanInterval:  10 * time.Minute,
	})
	// A live batch, so the sweep has real rows to leave alone.
	seedBatch(t, h.st, h.blobs, "batch_orphan", 2, testStart.Add(24*time.Hour))
	live, _ := h.st.ListItems("batch_orphan")

	// Three blobs nothing references: an input, a result, and a file blob the
	// file retention pass owns.
	orphanInput := "bitem_deadbeefdeadbeefdeadbeef-in"
	orphanResult := "bitem_cafebabecafebabecafebabe"
	fileBlob := "file-000000000000000000000000"
	for _, ref := range []string{orphanInput, orphanResult, fileBlob} {
		if err := h.blobs.PutPlain(ref, []byte("x")); err != nil {
			t.Fatalf("seed %s: %v", ref, err)
		}
	}
	// Age every blob past the retention window; a fresh blob belongs to a batch
	// that may still be committing.
	old := testStart.Add(-2 * time.Hour)
	for _, e := range mustReadDir(t, h.blobs.Dir()) {
		touch(t, h.blobs.Dir(), e, old)
	}

	// The pass is skipped on the first tick a dispatcher ever sees and runs one
	// OrphanInterval later.
	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(10*time.Minute))

	for _, ref := range []string{orphanInput, orphanResult} {
		if _, err := h.blobs.Raw(ref); !errors.Is(err, sealedblob.ErrNotFound) {
			t.Fatalf("orphan %s survived: err = %v", ref, err)
		}
	}
	if _, err := h.blobs.Raw(fileBlob); err != nil {
		t.Fatalf("the sweep deleted a file blob it does not own: %v", err)
	}
	for _, it := range live {
		if _, err := h.blobs.Raw(it.BlobRef); err != nil {
			t.Fatalf("the sweep deleted a referenced item blob %s: %v", it.BlobRef, err)
		}
	}
}

// A blob younger than the retention window belongs to a batch that may still be
// committing its rows, so the sweep must not race it.
func TestOrphanSweepLeavesYoungBlobsAlone(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{
		MaxAttempts:     3,
		OutputRetention: time.Hour,
		OrphanInterval:  10 * time.Minute,
	})
	ref := "bitem_feedfacefeedfacefeedface-in"
	if err := h.blobs.PutPlain(ref, []byte("x")); err != nil {
		t.Fatalf("seed: %v", err)
	}
	// Written a minute ago against a one-hour window: still inside it.
	touch(t, h.blobs.Dir(), ref, testStart.Add(-time.Minute))

	h.tick(t, ctx, testStart)
	h.tick(t, ctx, testStart.Add(10*time.Minute))

	if _, err := h.blobs.Raw(ref); err != nil {
		t.Fatalf("a young orphan candidate was deleted: %v", err)
	}
}

// The orphan pass runs on its interval, not on every tick.
func TestOrphanSweepIsRateLimited(t *testing.T) {
	ctx := context.Background()
	probe := &countingItemStore{Store: store.NewMemory(store.Config{})}
	h := newHarnessOn(t, batchlane.Config{
		MaxAttempts:     3,
		OutputRetention: time.Hour,
		OrphanInterval:  time.Hour,
	}, probe)
	ref := "bitem_0123456789abcdef01234567-in"
	if err := h.blobs.PutPlain(ref, []byte("x")); err != nil {
		t.Fatalf("seed: %v", err)
	}
	touch(t, h.blobs.Dir(), ref, testStart.Add(-2*time.Hour))

	// The pass never runs on the very first tick, and then only once an
	// OrphanInterval has elapsed: five consecutive ticks produce no pass at all.
	for i := 0; i < 5; i++ {
		h.tick(t, ctx, testStart.Add(time.Duration(i)*time.Second))
	}
	if probe.probes() != 0 {
		t.Fatalf("orphan pass ran on the first ticks (%d probes), want none", probe.probes())
	}
	h.tick(t, ctx, testStart.Add(time.Hour))
	if probe.probes() != 1 {
		t.Fatalf("orphan probes after the interval elapsed = %d, want 1", probe.probes())
	}
	for i := 1; i < 5; i++ {
		h.tick(t, ctx, testStart.Add(time.Hour+time.Duration(i)*time.Second))
	}
	if probe.probes() != 1 {
		t.Fatalf("orphan pass ran %d times, want 1 — it is rate limited", probe.probes())
	}
}

// countingItemStore counts the existence probes the orphan sweep makes, and can
// pretend every id it is asked about still has a row. The sweep runs off the
// tick, so the counter is read only after AwaitDispatch and is guarded anyway.
type countingItemStore struct {
	store.Store

	mu          sync.Mutex
	calls       int
	alwaysExist bool
}

func (c *countingItemStore) BatchItemExists(itemID string) (bool, error) {
	c.mu.Lock()
	c.calls++
	always := c.alwaysExist
	c.mu.Unlock()
	if always {
		return true, nil
	}
	return c.Store.BatchItemExists(itemID)
}

func (c *countingItemStore) probes() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.calls
}

// touch backdates a blob so a test can put it on either side of the retention
// window without waiting.
func touch(t *testing.T, dir, ref string, at time.Time) {
	t.Helper()
	if err := os.Chtimes(filepath.Join(dir, ref), at, at); err != nil {
		t.Fatalf("chtimes %s: %v", ref, err)
	}
}

func mustReadDir(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read dir: %v", err)
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		names = append(names, e.Name())
	}
	return names
}

// ---------------------------------------------------------------------------
// Review regressions
// ---------------------------------------------------------------------------

// A batch that leaves the open list while one of its items is still in flight
// used to make the settle's re-read of the batch fail, and the result was then
// written under the COORDINATOR's key instead of the consumer's — a silent
// downgrade of exactly the property the consumer paid for. The batch row is now
// carried from the tick's claim into the settle, so there is nothing to re-read.
func TestConsumerSealedResultIsNeverDowngraded(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	idleSlot(h.view, 4)
	h.dispatch.Block = make(chan struct{})

	consumer, err := e2e.GenerateSessionKeys()
	if err != nil {
		t.Fatalf("consumer keys: %v", err)
	}
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ResponseBody: []byte(`{"secret":"consumer-only"}`)}, nil
	}
	seedBatchKeyed(t, h.st, h.blobs, "batch_nodowngrade", 1, testStart.Add(24*time.Hour),
		base64.StdEncoding.EncodeToString(consumer.PublicKey[:]))

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)

	// The batch goes terminal underneath the in-flight dispatch, so it is gone
	// from ListOpenBatches by the time the result arrives.
	if ok, err := h.st.SetBatchStatus("batch_nodowngrade", store.BatchInProgress, store.BatchCompleted,
		testStart.Add(time.Second)); err != nil || !ok {
		t.Fatalf("complete batch: ok=%v err=%v", ok, err)
	}
	close(h.dispatch.Block)
	h.d.AwaitDispatch()

	h.tick(t, ctx, testStart.Add(2*time.Second)) // drain + settle

	items, _ := h.st.ListItems("batch_nodowngrade")
	if items[0].State != store.ItemSucceeded {
		t.Fatalf("state = %s, want succeeded", items[0].State)
	}
	ref := items[0].ResultBlobRef
	if _, err := h.blobs.Open(ref); err == nil {
		t.Fatal("the coordinator can open a result the consumer asked to be sealed to its own key")
	}
	raw, err := h.blobs.Raw(ref)
	if err != nil {
		t.Fatalf("raw: %v", err)
	}
	if bytes.Contains(raw, []byte("consumer-only")) {
		t.Fatal("the result blob holds plaintext")
	}
}

// A result that arrives after the sweep has cancelled the batch writes no blob
// at all: the item is terminal, FinishItem would refuse it, and a blob written
// and then deleted is one more chance to write it under the wrong key.
func TestConsumerSealedResultAfterCancelWritesNoBlob(t *testing.T) {
	ctx := context.Background()
	st := store.NewMemory(store.Config{})
	blobs := testBlobs(t)
	view := batchlanetest.NewFakeView()
	idleSlot(view, 4)

	dispatched := make(chan struct{})
	release := make(chan struct{})
	// Deliberately NOT the fake funnel: this dispatch reports a SUCCESS even
	// though its context was cancelled, which is the race the settle must
	// survive — the provider answered just as the sweep closed the batch.
	dispatch := func(context.Context, string, string, string, []byte) (batchlane.Outcome, error) {
		close(dispatched)
		<-release
		return batchlane.Outcome{ResponseBody: []byte(`{"secret":"consumer-only"}`)}, nil
	}

	consumer, err := e2e.GenerateSessionKeys()
	if err != nil {
		t.Fatalf("consumer keys: %v", err)
	}
	seedBatchKeyed(t, st, blobs, "batch_latecancel", 1, testStart.Add(24*time.Hour),
		base64.StdEncoding.EncodeToString(consumer.PublicKey[:]))
	items, _ := st.ListItems("batch_latecancel")
	ref := batchlane.ResultBlobRef(items[0].ID)

	d := batchlane.New(st, blobs, view, dispatch, (&batchlanetest.FakeFinalize{}).Fn(),
		batchlane.Config{MaxAttempts: 3}, testLogger())

	d.Tick(ctx, testStart)
	<-dispatched

	// The consumer cancels while the dispatch is out.
	if ok, err := st.SetBatchStatus("batch_latecancel", store.BatchInProgress, store.BatchCancelling,
		testStart.Add(time.Second)); err != nil || !ok {
		t.Fatalf("cancel: ok=%v err=%v", ok, err)
	}
	d.Tick(ctx, testStart.Add(2*time.Second)) // sweep: items cancelled, batch cancelled
	close(release)
	d.AwaitDispatch()

	d.Tick(ctx, testStart.Add(3*time.Second)) // drain the late success
	d.AwaitDispatch()

	if _, err := blobs.Raw(ref); !errors.Is(err, sealedblob.ErrNotFound) {
		t.Fatalf("a result blob exists for a cancelled item: err = %v, want ErrNotFound", err)
	}
	if _, err := blobs.Open(ref); err == nil {
		t.Fatal("the coordinator opened a result blob for a cancelled consumer-sealed item")
	}
	// The input blob is the only thing left on disk.
	if names := mustReadDir(t, blobs.Dir()); len(names) != 1 || names[0] != items[0].BlobRef {
		t.Fatalf("blob dir = %v, want only the input blob %q", names, items[0].BlobRef)
	}
	if states := itemStates(t, st, "batch_latecancel"); states[store.ItemCancelled] != 1 {
		t.Fatalf("item states = %v, want one cancelled", states)
	}
}

// A dispatch goroutine that panics must not take the process down, and must not
// strand its claim inflight until the batch expires: the recovery reports a
// permanent request_failed so the item settles.
func TestPanicInADispatchSettlesTheItem(t *testing.T) {
	ctx := context.Background()
	st := store.NewMemory(store.Config{})
	blobs := testBlobs(t)
	view := batchlanetest.NewFakeView()
	idleSlot(view, 4)

	calls := 0
	dispatch := func(context.Context, string, string, string, []byte) (batchlane.Outcome, error) {
		calls++
		panic("the funnel exploded")
	}
	seedBatch(t, st, blobs, "batch_panic", 1, testStart.Add(24*time.Hour))

	d := batchlane.New(st, blobs, view, dispatch, (&batchlanetest.FakeFinalize{}).Fn(),
		batchlane.Config{MaxAttempts: 3}, testLogger())

	d.Tick(ctx, testStart)
	d.AwaitDispatch()
	d.Tick(ctx, testStart.Add(time.Second)) // drain + settle
	d.AwaitDispatch()

	if calls != 1 {
		t.Fatalf("dispatch calls = %d, want 1 — a panic is permanent, not retried", calls)
	}
	items, _ := st.ListItems("batch_panic")
	if items[0].State != store.ItemFailed {
		t.Fatalf("state = %s, want failed — a panicking dispatch stranded its claim", items[0].State)
	}
	if items[0].LastErrorCode != batchlane.ErrCodeRequestFailed {
		t.Fatalf("error code = %q, want %q", items[0].LastErrorCode, batchlane.ErrCodeRequestFailed)
	}
	if n := d.InflightItems(); n != 0 {
		t.Fatalf("dispatcher still tracks %d in-flight items after a panic", n)
	}
	b, _ := st.GetBatch(testAccount, "batch_panic")
	if b.CountsFailed != 1 {
		t.Fatalf("counts_failed = %d, want 1", b.CountsFailed)
	}
}

// flakyFinishStore fails the first n FinishItem calls, the way a store behind a
// dropped connection would.
type flakyFinishStore struct {
	store.Store

	mu        sync.Mutex
	remaining int
}

func (f *flakyFinishStore) FinishItem(r store.ItemResult, at time.Time) (bool, error) {
	f.mu.Lock()
	if f.remaining > 0 {
		f.remaining--
		f.mu.Unlock()
		return false, errors.New("store: connection reset")
	}
	f.mu.Unlock()
	return f.Store.FinishItem(r, at)
}

// A FinishItem that fails leaves the item inflight in the store and a result
// blob on disk that nothing will ever read. The settle now releases the claim
// and drops the blob, so the next tick re-claims the item instead of losing it
// until the batch expires.
func TestFinishItemErrorReOffersTheItem(t *testing.T) {
	ctx := context.Background()
	flaky := &flakyFinishStore{Store: store.NewMemory(store.Config{}), remaining: 1}
	h := newHarnessOn(t, batchlane.Config{MaxAttempts: 3}, flaky)
	key := idleSlot(h.view, 4)
	h.dispatch.Respond = func(int, string, []byte) (batchlane.Outcome, error) {
		return batchlane.Outcome{ResponseBody: []byte(`{"choices":[]}`)}, nil
	}
	seedBatch(t, h.st, h.blobs, "batch_finishfail", 1, testStart.Add(24*time.Hour))
	items, _ := h.st.ListItems("batch_finishfail")
	ref := batchlane.ResultBlobRef(items[0].ID)

	h.tick(t, ctx, testStart) // claim + dispatch
	// Take the slot away so the tick that settles cannot immediately re-claim,
	// and the released item is observable at rest.
	h.view.Remove(key)
	h.tick(t, ctx, testStart.Add(time.Second)) // drain: FinishItem fails

	items, _ = h.st.ListItems("batch_finishfail")
	if items[0].State != store.ItemPending {
		t.Fatalf("state after a failed FinishItem = %s, want pending", items[0].State)
	}
	if _, err := h.blobs.Raw(ref); !errors.Is(err, sealedblob.ErrNotFound) {
		t.Fatalf("the unusable result blob survived: err = %v, want ErrNotFound", err)
	}

	idleSlot(h.view, 4)
	h.tick(t, ctx, testStart.Add(2*time.Second)) // re-claim + dispatch
	h.tick(t, ctx, testStart.Add(3*time.Second)) // drain: FinishItem succeeds

	items, _ = h.st.ListItems("batch_finishfail")
	if items[0].State != store.ItemSucceeded {
		t.Fatalf("state = %s, want succeeded on the retry", items[0].State)
	}
	if h.dispatch.Len() != 2 {
		t.Fatalf("dispatch calls = %d, want 2", h.dispatch.Len())
	}
	if _, err := h.blobs.Raw(items[0].ResultBlobRef); err != nil {
		t.Fatalf("no result blob after the retry: %v", err)
	}
}

// One orphan pass is bounded on the store probes it makes, not only on the
// blobs it deletes: a directory whose blobs all still have rows produces no
// deletes at all, and without a scan bound it would issue one query per blob,
// every pass, forever.
func TestOrphanSweepBoundsItsProbes(t *testing.T) {
	ctx := context.Background()
	probe := &countingItemStore{Store: store.NewMemory(store.Config{}), alwaysExist: true}
	h := newHarnessOn(t, batchlane.Config{
		MaxAttempts:     3,
		OutputRetention: time.Hour,
		OrphanInterval:  10 * time.Minute,
	}, probe)

	old := testStart.Add(-2 * time.Hour)
	for i := 0; i < 5000; i++ {
		ref := fmt.Sprintf("bitem_%024x", i)
		if err := os.WriteFile(filepath.Join(h.blobs.Dir(), ref), []byte("x"), 0o600); err != nil {
			t.Fatalf("seed blob %d: %v", i, err)
		}
		touch(t, h.blobs.Dir(), ref, old)
	}

	h.tick(t, ctx, testStart)                     // first tick: no pass
	h.tick(t, ctx, testStart.Add(10*time.Minute)) // one full pass

	got := probe.probes()
	if got == 0 {
		t.Fatal("the orphan pass never ran")
	}
	if got > batchlane.MaxOrphanScan {
		t.Fatalf("orphan pass made %d probes over 5000 blobs, want at most %d",
			got, batchlane.MaxOrphanScan)
	}
	if got != batchlane.MaxOrphanScan {
		t.Fatalf("orphan pass made %d probes, want exactly the bound %d "+
			"(every blob is a candidate and none is deleted)", got, batchlane.MaxOrphanScan)
	}
}

// The deadline progress floor is granted once per tick FLEET-WIDE. Granting it
// per urgent batch would put one row per urgent batch onto a fleet whose AIMD
// target is zero — exactly the online interference the floor exists to bound.
func TestProgressFloorGrantsOneItemPerTickAcrossBatches(t *testing.T) {
	ctx := context.Background()
	h := newHarness(t, batchlane.Config{MaxAttempts: 3})
	key := idleSlot(h.view, 4)
	// Online traffic is queueing, so every slot's AIMD target is pinned at zero
	// and nothing but the floor can claim.
	h.view.Update(key, func(s *batchlane.SlotSignal) { s.Waiting = 2 })
	h.dispatch.Block = make(chan struct{})

	// Three batches, all well past FloorUrgency, all with full buckets.
	for _, id := range []string{"batch_u1", "batch_u2", "batch_u3"} {
		seedBatch(t, h.st, h.blobs, id, 5, testStart.Add(10*time.Minute))
	}

	h.d.Tick(ctx, testStart)
	waitForCalls(t, h.dispatch, 1)
	if got := h.d.SlotTarget(key); got != 0 {
		t.Fatalf("slot target = %d, want 0", got)
	}
	if n := h.dispatch.Len(); n != 1 {
		t.Fatalf("dispatch calls = %d, want 1 — the floor is one item per tick, not one per batch", n)
	}

	// A second tick one second later grants nothing: every bucket refills at
	// 0.2/s, so the grant is still rate limited fleet-wide.
	h.d.Tick(ctx, testStart.Add(time.Second))
	if n := h.dispatch.Len(); n != 1 {
		t.Fatalf("dispatch calls = %d, want 1 after a second tick", n)
	}

	close(h.dispatch.Block)
	h.d.AwaitDispatch()
}
