package api

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
	"github.com/eigeninference/d-inference/coordinator/ratelimit"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// TestBatchRetrievalRoutesAreRateLimited is the S4 regression. The four batch
// retrieval routes ran outside rateLimitConsumer on the theory that they are
// cheap reads. None is: a file download streams up to the 16 MiB input cap off
// disk and an inline batch's GET decrypts up to maxInlineResults sealed blobs,
// so an unmetered poll loop on any of them lets one account monopolize the
// coordinator while every metered route stays throttled.
func TestBatchRetrievalRoutesAreRateLimited(t *testing.T) {
	e := newBatchEnv(t)
	// One token, refilled once per ~17 minutes. The already-metered upload
	// spends it, so every retrieval route below must be refused — and would be
	// served if it still bypassed the limiter.
	e.srv.SetRateLimiter(ratelimit.New(ratelimit.Config{RPS: 0.001, Burst: 1}))

	status, file := e.uploadMultipart(jsonlLines(1), "batch", "in.jsonl", e.key)
	if status != http.StatusOK {
		t.Fatalf("fixture: the upload did not spend the bucket (status=%d body=%v)", status, file)
	}

	for _, path := range []string{
		"/v1/batches",
		"/v1/batches/batch_does_not_exist",
		"/v1/files/file-does-not-exist",
		"/v1/files/file-does-not-exist/content",
	} {
		t.Run(path, func(t *testing.T) {
			status, raw := e.request(http.MethodGet, path, "", nil, e.key)
			if status != http.StatusTooManyRequests {
				t.Fatalf("status=%d body=%s, want 429 — this route bypasses the consumer limiter",
					status, raw)
			}
		})
	}

	// Control: with no limiter configured the same route is served, so the 429s
	// above are the limiter and not a broken fixture.
	e.srv.SetRateLimiter(nil)
	if status, _ := e.getJSON("/v1/batches", e.key); status != http.StatusOK {
		t.Fatalf("status=%d with no limiter configured, want 200", status)
	}
}

// TestInlineResultsAreCapped is the other half of S4: a terminal inline batch
// with more items than maxInlineResults answers with the first maxInlineResults
// results, a results_truncated flag, and the output file that holds the
// complete set — instead of decrypting every blob it owns on one GET.
func TestInlineResultsAreCapped(t *testing.T) {
	e := newBatchEnv(t)
	const total = maxInlineResults + 5

	now := time.Now().UTC()
	batch := &store.Batch{
		ID: "batch_capped", AccountID: e.account,
		Endpoint: "/v1/chat/completions", CompletionWindow: "24h",
		CreatedAt: now, ExpiresAt: now.Add(24 * time.Hour),
		CountsTotal: total, SealedTo: sealedToCoordinator,
		Source: batchSourceInline, Model: batchTestModel, RequestedModel: batchTestModel,
	}
	items := make([]*store.BatchItem, 0, total)
	for i := 0; i < total; i++ {
		items = append(items, &store.BatchItem{
			ID: fmt.Sprintf("bitem_capped_%04d", i), BatchID: batch.ID,
			CustomID: fmt.Sprintf("req-%d", i), LineNo: i + 1,
			State: store.ItemPending, BlobRef: fmt.Sprintf("bitem_capped_%04d-in", i),
		})
	}
	if err := e.st.CreateBatch(batch, items); err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	if ok, err := e.st.SetBatchStatus(batch.ID, store.BatchValidating, store.BatchInProgress, now); err != nil || !ok {
		t.Fatalf("SetBatchStatus: ok=%v err=%v", ok, err)
	}
	// Settle every item as a success with a sealed result, exactly as the
	// dispatcher does, so the GET has real blobs to open.
	blobs := e.srv.BatchBlobs()
	for _, it := range items {
		if err := blobs.PutPlain(batchlane.ResultBlobRef(it.ID),
			[]byte(`{"id":"`+it.ID+`","choices":[{"message":{"content":"ok"}}]}`)); err != nil {
			t.Fatalf("seal result: %v", err)
		}
		if _, err := e.st.ClaimPendingItems(batch.ID, 1, now); err != nil {
			t.Fatalf("ClaimPendingItems: %v", err)
		}
		if ok, err := e.st.FinishItem(store.ItemResult{
			ItemID: it.ID, Succeeded: true, ResultBlobRef: batchlane.ResultBlobRef(it.ID),
		}, now); err != nil || !ok {
			t.Fatalf("FinishItem(%s): ok=%v err=%v", it.ID, ok, err)
		}
	}
	if _, err := e.srv.FinalizeBatchIfDone(batch.ID, now); err != nil {
		t.Fatalf("FinalizeBatchIfDone: %v", err)
	}

	status, obj := e.getJSON("/v1/batches/"+batch.ID, e.key)
	if status != http.StatusOK {
		t.Fatalf("status=%d body=%v", status, obj)
	}
	results, ok := obj["results"].([]any)
	if !ok {
		t.Fatalf("no results array: %v", obj)
	}
	if len(results) != maxInlineResults {
		t.Fatalf("results=%d, want the %d cap", len(results), maxInlineResults)
	}
	if truncated, _ := obj["results_truncated"].(bool); !truncated {
		t.Fatalf("results_truncated=%v, want true for a %d-item batch", obj["results_truncated"], total)
	}
	if id, _ := obj["output_file_id"].(string); id == "" {
		t.Fatal("a truncated response must still name the output file holding the complete set")
	}

	// Control: a batch inside the cap reports every result and no flag.
	small := e.smallTerminalInlineBatch(t, 3)
	status, obj = e.getJSON("/v1/batches/"+small, e.key)
	if status != http.StatusOK {
		t.Fatalf("control status=%d body=%v", status, obj)
	}
	if results, _ := obj["results"].([]any); len(results) != 3 {
		t.Fatalf("control results=%d, want 3", len(results))
	}
	if _, present := obj["results_truncated"]; present {
		t.Fatalf("control carried results_truncated: %v", obj)
	}
}

// smallTerminalInlineBatch creates and settles an inline batch of n succeeded
// items, returning its id.
func (e *batchEnv) smallTerminalInlineBatch(t *testing.T, n int) string {
	t.Helper()
	requests := make([]any, 0, n)
	for i := 0; i < n; i++ {
		requests = append(requests, map[string]any{
			"custom_id": fmt.Sprintf("small-%d", i),
			"body":      map[string]any{"messages": []any{map[string]any{"role": "user", "content": batchTestPrompt}}},
		})
	}
	status, obj := e.postJSON("/v1/batches", map[string]any{
		"model": batchTestModel, "endpoint": "/v1/chat/completions", "requests": requests,
	}, e.key)
	if status != http.StatusAccepted {
		t.Fatalf("create small batch: status=%d body=%v", status, obj)
	}
	batchID, _ := obj["id"].(string)

	now := time.Now().UTC()
	items, err := e.st.ListItems(batchID)
	if err != nil {
		t.Fatalf("ListItems: %v", err)
	}
	blobs := e.srv.BatchBlobs()
	for _, it := range items {
		if err := blobs.PutPlain(batchlane.ResultBlobRef(it.ID),
			[]byte(`{"id":"`+it.ID+`","choices":[{"message":{"content":"ok"}}]}`)); err != nil {
			t.Fatalf("seal result: %v", err)
		}
		if _, err := e.st.ClaimPendingItems(batchID, 1, now); err != nil {
			t.Fatalf("ClaimPendingItems: %v", err)
		}
		if ok, err := e.st.FinishItem(store.ItemResult{
			ItemID: it.ID, Succeeded: true, ResultBlobRef: batchlane.ResultBlobRef(it.ID),
		}, now); err != nil || !ok {
			t.Fatalf("FinishItem: ok=%v err=%v", ok, err)
		}
	}
	if _, err := e.srv.FinalizeBatchIfDone(batchID, now); err != nil {
		t.Fatalf("FinalizeBatchIfDone: %v", err)
	}
	return batchID
}
