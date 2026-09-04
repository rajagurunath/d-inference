package store

import (
	"context"
	"errors"
	"regexp"
	"sync"
	"testing"
	"time"
)

// Batch-lane store tests. Like the other store tests in this package they run
// the same body against every backend returned by storeBackends (memory always;
// postgres when DATABASE_URL is set).

// seedBatch creates the input file, the batch, and n pending items, and returns
// the batch with its items. The batch is left in validating.
func seedBatch(t *testing.T, s Store, accountID string, n int) (*Batch, []*BatchItem) {
	t.Helper()

	file := &BatchFile{
		ID:        uniqueID("file"),
		AccountID: accountID,
		Purpose:   "batch",
		Filename:  "requests.jsonl",
		SizeBytes: int64(n * 128),
		CreatedAt: time.Now().UTC(),
		SealedBy:  "coordinator",
	}
	file.BlobRef = file.ID
	if err := s.CreateBatchFile(file); err != nil {
		t.Fatalf("CreateBatchFile: %v", err)
	}

	now := time.Now().UTC()
	b := &Batch{
		ID:               uniqueID("batch"),
		AccountID:        accountID,
		InputFileID:      file.ID,
		Endpoint:         "/v1/chat/completions",
		CompletionWindow: "24h",
		CreatedAt:        now,
		ExpiresAt:        now.Add(24 * time.Hour),
		CountsTotal:      n,
		SealedTo:         "coordinator",
		Source:           "file",
		Metadata:         map[string]string{"job": "nightly"},
	}
	items := make([]*BatchItem, 0, n)
	for i := 0; i < n; i++ {
		id := uniqueID("bitem")
		items = append(items, &BatchItem{
			ID:       id,
			BatchID:  b.ID,
			CustomID: uniqueID("req"),
			LineNo:   i + 1,
			State:    ItemPending,
			BlobRef:  id,
		})
	}
	if err := s.CreateBatch(b, items); err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	return b, items
}

// startBatch seeds a batch and moves it to in_progress so items can be claimed.
func startBatch(t *testing.T, s Store, accountID string, n int) (*Batch, []*BatchItem) {
	t.Helper()
	b, items := seedBatch(t, s, accountID, n)
	ok, err := s.SetBatchStatus(b.ID, BatchValidating, BatchInProgress, time.Now().UTC())
	if err != nil || !ok {
		t.Fatalf("SetBatchStatus(validating->in_progress) = %v, %v", ok, err)
	}
	return b, items
}

func TestCreateBatchStartsValidating(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, items := seedBatch(t, s, uniqueID("acct"), 3)

			got, ok := s.GetBatch(b.AccountID, b.ID)
			if !ok {
				t.Fatal("GetBatch: batch not found")
			}
			if got.Status != BatchValidating {
				t.Errorf("status = %q, want %q", got.Status, BatchValidating)
			}
			if got.CountsTotal != 3 {
				t.Errorf("counts_total = %d, want 3", got.CountsTotal)
			}
			if got.Metadata["job"] != "nightly" {
				t.Errorf("metadata = %v, want job=nightly", got.Metadata)
			}
			if got.Endpoint != b.Endpoint || got.InputFileID != b.InputFileID {
				t.Errorf("round-trip mismatch: %+v", got)
			}

			listed, err := s.ListItems(b.ID)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(listed) != len(items) {
				t.Fatalf("ListItems returned %d items, want %d", len(listed), len(items))
			}
			for i, it := range listed {
				if it.LineNo != i+1 {
					t.Errorf("item %d line_no = %d, want %d", i, it.LineNo, i+1)
				}
				if it.State != ItemPending {
					t.Errorf("item %d state = %q, want %q", i, it.State, ItemPending)
				}
			}
		})
	}
}

// TestCreateBatchScalesToLargeItemCount guards against CreateBatch inserting
// items one at a time under a fixed timeout: a batch this size must load via
// a bulk path (CopyFrom on Postgres) well inside a few seconds.
func TestCreateBatchScalesToLargeItemCount(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			const n = 5000
			started := time.Now()
			b, items := seedBatch(t, s, uniqueID("acct"), n)
			if elapsed := time.Since(started); elapsed > 5*time.Second {
				t.Fatalf("CreateBatch(%d items) took %v, want under 5s", n, elapsed)
			}

			listed, err := s.ListItems(b.ID)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(listed) != len(items) {
				t.Fatalf("ListItems returned %d items, want %d", len(listed), len(items))
			}
		})
	}
}

func TestCreateBatchRejectsDuplicateCustomID(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			file := &BatchFile{
				ID:        uniqueID("file"),
				AccountID: acct,
				Purpose:   "batch",
				Filename:  "requests.jsonl",
				SizeBytes: 256,
				CreatedAt: time.Now().UTC(),
				SealedBy:  "coordinator",
			}
			file.BlobRef = file.ID
			if err := s.CreateBatchFile(file); err != nil {
				t.Fatalf("CreateBatchFile: %v", err)
			}

			now := time.Now().UTC()
			b := &Batch{
				ID:               uniqueID("batch"),
				AccountID:        acct,
				InputFileID:      file.ID,
				Endpoint:         "/v1/chat/completions",
				CompletionWindow: "24h",
				CreatedAt:        now,
				ExpiresAt:        now.Add(24 * time.Hour),
				CountsTotal:      2,
				SealedTo:         "coordinator",
				Source:           "file",
			}
			dup := uniqueID("req")
			id1, id2 := uniqueID("bitem"), uniqueID("bitem")
			items := []*BatchItem{
				{ID: id1, BatchID: b.ID, CustomID: dup, LineNo: 1, State: ItemPending, BlobRef: id1},
				{ID: id2, BatchID: b.ID, CustomID: dup, LineNo: 2, State: ItemPending, BlobRef: id2},
			}

			err := s.CreateBatch(b, items)
			if !errors.Is(err, ErrDuplicateCustomID) {
				t.Fatalf("CreateBatch = %v, want ErrDuplicateCustomID", err)
			}
			if _, ok := s.GetBatch(acct, b.ID); ok {
				t.Fatal("batch was created despite a duplicate custom_id")
			}
		})
	}
}

// TestCreateBatchInlineSourceHasNoInputFile covers the inline batch form,
// which has no uploaded file: input_file_id must accept an empty string
// rather than requiring a row in batch_files.
func TestCreateBatchInlineSourceHasNoInputFile(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			now := time.Now().UTC()
			b := &Batch{
				ID:               uniqueID("batch"),
				AccountID:        acct,
				InputFileID:      "",
				Endpoint:         "/v1/chat/completions",
				CompletionWindow: "24h",
				CreatedAt:        now,
				ExpiresAt:        now.Add(24 * time.Hour),
				CountsTotal:      1,
				SealedTo:         "coordinator",
				Source:           "inline",
				Model:            "some-model",
			}
			id := uniqueID("bitem")
			items := []*BatchItem{{ID: id, BatchID: b.ID, CustomID: uniqueID("req"), LineNo: 1, State: ItemPending, BlobRef: id}}

			if err := s.CreateBatch(b, items); err != nil {
				t.Fatalf("CreateBatch(inline): %v", err)
			}

			got, ok := s.GetBatch(acct, b.ID)
			if !ok {
				t.Fatal("GetBatch: batch not found")
			}
			if got.InputFileID != "" {
				t.Fatalf("input_file_id = %q, want empty", got.InputFileID)
			}
			if got.Source != "inline" {
				t.Fatalf("source = %q, want inline", got.Source)
			}
		})
	}
}

func TestGetBatchIsScopedToAccount(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := seedBatch(t, s, uniqueID("acct"), 1)
			if _, ok := s.GetBatch("someone-else", b.ID); ok {
				t.Fatal("GetBatch returned a batch for the wrong account")
			}
		})
	}
}

func TestCreateBatchAndClaimIsAtomic(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 10)

			var (
				got int
				mu  sync.Mutex
				wg  sync.WaitGroup
			)
			for i := 0; i < 4; i++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					claimed, err := s.ClaimPendingItems(b.ID, 5, time.Now().UTC())
					if err != nil {
						return
					}
					mu.Lock()
					got += len(claimed)
					mu.Unlock()
				}()
			}
			wg.Wait()

			if got != 10 {
				t.Fatalf("claimed %d, want exactly 10", got)
			}
			_, _, inflight, _, _, err := s.CountItems(b.ID)
			if err != nil {
				t.Fatalf("CountItems: %v", err)
			}
			if inflight != 10 {
				t.Fatalf("inflight = %d, want 10", inflight)
			}
		})
	}
}

func TestClaimOrdersByLineNoAndCountsAnAttempt(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 5)

			at := time.Now().UTC()
			claimed, err := s.ClaimPendingItems(b.ID, 3, at)
			if err != nil {
				t.Fatalf("ClaimPendingItems: %v", err)
			}
			if len(claimed) != 3 {
				t.Fatalf("claimed %d, want 3", len(claimed))
			}
			for i, it := range claimed {
				if it.LineNo != i+1 {
					t.Errorf("claim %d line_no = %d, want %d", i, it.LineNo, i+1)
				}
				if it.State != ItemInflight {
					t.Errorf("claim %d state = %q, want %q", i, it.State, ItemInflight)
				}
				if it.Attempts != 1 {
					t.Errorf("claim %d attempts = %d, want 1", i, it.Attempts)
				}
				if it.SubmittedAt == nil {
					t.Errorf("claim %d submitted_at not set", i)
				}
				if it.BlobRef == "" {
					t.Errorf("claim %d blob_ref empty", i)
				}
			}
		})
	}
}

func TestClaimRefusesUnlessInProgress(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := seedBatch(t, s, uniqueID("acct"), 4)

			claimed, err := s.ClaimPendingItems(b.ID, 4, time.Now().UTC())
			if err != nil {
				t.Fatalf("ClaimPendingItems (validating): %v", err)
			}
			if len(claimed) != 0 {
				t.Fatalf("claimed %d while validating, want 0", len(claimed))
			}

			if ok, err := s.SetBatchStatus(b.ID, BatchValidating, BatchInProgress, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("SetBatchStatus(in_progress) = %v, %v", ok, err)
			}
			if ok, err := s.SetBatchStatus(b.ID, BatchInProgress, BatchCancelling, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("SetBatchStatus(cancelling) = %v, %v", ok, err)
			}
			claimed, err = s.ClaimPendingItems(b.ID, 4, time.Now().UTC())
			if err != nil {
				t.Fatalf("ClaimPendingItems (cancelling): %v", err)
			}
			if len(claimed) != 0 {
				t.Fatalf("claimed %d while cancelling, want 0", len(claimed))
			}
		})
	}
}

func TestReleaseItemDoesNotCountAttempt(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 2)

			claimed, err := s.ClaimPendingItems(b.ID, 1, time.Now().UTC())
			if err != nil || len(claimed) != 1 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			released, err := s.ReleaseItem(claimed[0].ID)
			if err != nil {
				t.Fatalf("ReleaseItem: %v", err)
			}
			if !released {
				t.Fatal("ReleaseItem = false, want true")
			}

			items, err := s.ListItems(b.ID, ItemPending)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(items) != 2 {
				t.Fatalf("pending = %d, want 2", len(items))
			}
			if items[0].Attempts != 0 {
				t.Errorf("attempts = %d after release, want 0", items[0].Attempts)
			}
			if items[0].SubmittedAt != nil {
				t.Errorf("submitted_at = %v after release, want nil", items[0].SubmittedAt)
			}

			// A second release is a no-op: the item is pending, not inflight.
			if released, err := s.ReleaseItem(claimed[0].ID); err != nil || released {
				t.Fatalf("second ReleaseItem = %v, %v; want false, nil", released, err)
			}
		})
	}
}

func TestFinishItemIsIdempotent(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 3)

			claimed, err := s.ClaimPendingItems(b.ID, 2, time.Now().UTC())
			if err != nil || len(claimed) != 2 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}

			result := ItemResult{
				ItemID:           claimed[0].ID,
				Succeeded:        true,
				PromptTokens:     11,
				CompletionTokens: 22,
				RequestID:        "req-abc",
				ResultBlobRef:    claimed[0].ID + "-out",
			}
			ok, err := s.FinishItem(result, time.Now().UTC())
			if err != nil || !ok {
				t.Fatalf("FinishItem = %v, %v; want true, nil", ok, err)
			}
			ok, err = s.FinishItem(result, time.Now().UTC())
			if err != nil {
				t.Fatalf("second FinishItem: %v", err)
			}
			if ok {
				t.Fatal("second FinishItem = true, want false (already terminal)")
			}

			got, found := s.GetBatch(b.AccountID, b.ID)
			if !found {
				t.Fatal("GetBatch: not found")
			}
			if got.CountsCompleted != 1 || got.CountsFailed != 0 {
				t.Fatalf("counts = %d completed / %d failed, want 1/0", got.CountsCompleted, got.CountsFailed)
			}

			done, err := s.ListItems(b.ID, ItemSucceeded)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(done) != 1 {
				t.Fatalf("succeeded = %d, want 1", len(done))
			}
			if done[0].PromptTokens != 11 || done[0].CompletionTokens != 22 {
				t.Errorf("tokens = %d/%d, want 11/22", done[0].PromptTokens, done[0].CompletionTokens)
			}
			if done[0].RequestID != "req-abc" || done[0].ResultBlobRef != result.ResultBlobRef {
				t.Errorf("request_id / result_blob_ref = %q / %q", done[0].RequestID, done[0].ResultBlobRef)
			}
			if done[0].FinishedAt == nil {
				t.Error("finished_at not set")
			}

			// A cancelled item never reaches a terminal count, even if a late
			// result arrives for it.
			if _, err := s.CancelOpenItems(b.ID, time.Now().UTC()); err != nil {
				t.Fatalf("CancelOpenItems: %v", err)
			}
			ok, err = s.FinishItem(ItemResult{ItemID: claimed[1].ID, Succeeded: true}, time.Now().UTC())
			if err != nil {
				t.Fatalf("FinishItem (cancelled): %v", err)
			}
			if ok {
				t.Fatal("FinishItem on a cancelled item = true, want false")
			}
			got, _ = s.GetBatch(b.AccountID, b.ID)
			if got.CountsCompleted != 1 || got.CountsFailed != 0 {
				t.Fatalf("counts moved after a late result: %d/%d, want 1/0", got.CountsCompleted, got.CountsFailed)
			}
		})
	}
}

func TestFinishItemFailureMovesFailedCount(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 1)

			claimed, err := s.ClaimPendingItems(b.ID, 1, time.Now().UTC())
			if err != nil || len(claimed) != 1 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			ok, err := s.FinishItem(ItemResult{
				ItemID:    claimed[0].ID,
				Succeeded: false,
				ErrorCode: "request_failed",
			}, time.Now().UTC())
			if err != nil || !ok {
				t.Fatalf("FinishItem = %v, %v; want true, nil", ok, err)
			}

			got, _ := s.GetBatch(b.AccountID, b.ID)
			if got.CountsCompleted != 0 || got.CountsFailed != 1 {
				t.Fatalf("counts = %d/%d, want 0 completed / 1 failed", got.CountsCompleted, got.CountsFailed)
			}
			failed, err := s.ListItems(b.ID, ItemFailed)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(failed) != 1 || failed[0].LastErrorCode != "request_failed" {
				t.Fatalf("failed items = %+v", failed)
			}
		})
	}
}

func TestCountsExcludeExpiredAndCancelled(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 6)

			claimed, err := s.ClaimPendingItems(b.ID, 4, time.Now().UTC())
			if err != nil || len(claimed) != 4 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			for i := 0; i < 3; i++ {
				if ok, err := s.FinishItem(ItemResult{ItemID: claimed[i].ID, Succeeded: true}, time.Now().UTC()); err != nil || !ok {
					t.Fatalf("FinishItem(%d) = %v, %v", i, ok, err)
				}
			}
			if ok, err := s.FinishItem(ItemResult{ItemID: claimed[3].ID, ErrorCode: "request_failed"}, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("FinishItem(failed) = %v, %v", ok, err)
			}

			expired, err := s.ExpireOpenItems(b.ID, time.Now().UTC())
			if err != nil {
				t.Fatalf("ExpireOpenItems: %v", err)
			}
			if expired != 2 {
				t.Fatalf("expired %d items, want 2", expired)
			}

			got, _ := s.GetBatch(b.AccountID, b.ID)
			if got.CountsCompleted != 3 || got.CountsFailed != 1 || got.CountsTotal != 6 {
				t.Fatalf("counts = %d completed / %d failed / %d total, want 3/1/6",
					got.CountsCompleted, got.CountsFailed, got.CountsTotal)
			}

			total, pending, inflight, succeeded, failed, err := s.CountItems(b.ID)
			if err != nil {
				t.Fatalf("CountItems: %v", err)
			}
			if total != 6 || pending != 0 || inflight != 0 || succeeded != 3 || failed != 1 {
				t.Fatalf("CountItems = %d/%d/%d/%d/%d, want 6/0/0/3/1",
					total, pending, inflight, succeeded, failed)
			}
		})
	}
}

func TestRequeueInflightItems(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 3)

			if _, err := s.ClaimPendingItems(b.ID, 3, time.Now().UTC()); err != nil {
				t.Fatalf("ClaimPendingItems: %v", err)
			}
			n, err := s.RequeueInflightItems(b.ID)
			if err != nil {
				t.Fatalf("RequeueInflightItems: %v", err)
			}
			if n != 3 {
				t.Fatalf("requeued %d, want 3", n)
			}

			items, err := s.ListItems(b.ID, ItemPending)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(items) != 3 {
				t.Fatalf("pending = %d, want 3", len(items))
			}
			// The attempt already happened, so requeue keeps the count (unlike
			// ReleaseItem, which un-counts a claim that never dispatched).
			if items[0].Attempts != 1 {
				t.Errorf("attempts = %d after requeue, want 1", items[0].Attempts)
			}
		})
	}
}

func TestCancelOpenItemsLeavesTerminalItemsAlone(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 4)

			claimed, err := s.ClaimPendingItems(b.ID, 2, time.Now().UTC())
			if err != nil || len(claimed) != 2 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			if ok, err := s.FinishItem(ItemResult{ItemID: claimed[0].ID, Succeeded: true}, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("FinishItem = %v, %v", ok, err)
			}

			n, err := s.CancelOpenItems(b.ID, time.Now().UTC())
			if err != nil {
				t.Fatalf("CancelOpenItems: %v", err)
			}
			if n != 3 {
				t.Fatalf("cancelled %d, want 3 (1 inflight + 2 pending)", n)
			}
			cancelled, err := s.ListItems(b.ID, ItemCancelled)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(cancelled) != 3 {
				t.Fatalf("cancelled items = %d, want 3", len(cancelled))
			}
			succeeded, err := s.ListItems(b.ID, ItemSucceeded)
			if err != nil {
				t.Fatalf("ListItems: %v", err)
			}
			if len(succeeded) != 1 {
				t.Fatalf("succeeded items = %d, want 1", len(succeeded))
			}
		})
	}
}

func TestSetBatchStatusCAS(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := seedBatch(t, s, uniqueID("acct"), 1)

			at := time.Now().UTC()
			ok, err := s.SetBatchStatus(b.ID, BatchInProgress, BatchCompleted, at)
			if err != nil {
				t.Fatalf("SetBatchStatus: %v", err)
			}
			if ok {
				t.Fatal("CAS from a mismatched status returned true")
			}
			got, _ := s.GetBatch(b.AccountID, b.ID)
			if got.Status != BatchValidating {
				t.Fatalf("status = %q after a failed CAS, want %q", got.Status, BatchValidating)
			}

			if ok, err := s.SetBatchStatus(b.ID, BatchValidating, BatchInProgress, at); err != nil || !ok {
				t.Fatalf("SetBatchStatus(in_progress) = %v, %v", ok, err)
			}
			got, _ = s.GetBatch(b.AccountID, b.ID)
			if got.Status != BatchInProgress {
				t.Fatalf("status = %q, want %q", got.Status, BatchInProgress)
			}
			if got.InProgressAt == nil {
				t.Fatal("in_progress_at not stamped")
			}

			if ok, err := s.SetBatchStatus(b.ID, BatchInProgress, BatchCompleted, at); err != nil || !ok {
				t.Fatalf("SetBatchStatus(completed) = %v, %v", ok, err)
			}
			got, _ = s.GetBatch(b.AccountID, b.ID)
			if got.CompletedAt == nil {
				t.Fatal("completed_at not stamped")
			}

			if _, err := s.SetBatchStatus(uniqueID("missing"), BatchValidating, BatchFailed, at); err != nil {
				t.Fatalf("SetBatchStatus on a missing batch: %v", err)
			}
		})
	}
}

func TestSetBatchStatusStampsCancelledAt(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 1)

			at := time.Now().UTC()
			if ok, err := s.SetBatchStatus(b.ID, BatchInProgress, BatchCancelling, at); err != nil || !ok {
				t.Fatalf("SetBatchStatus(cancelling) = %v, %v", ok, err)
			}
			if ok, err := s.SetBatchStatus(b.ID, BatchCancelling, BatchCancelled, at); err != nil || !ok {
				t.Fatalf("SetBatchStatus(cancelled) = %v, %v", ok, err)
			}
			got, _ := s.GetBatch(b.AccountID, b.ID)
			if got.CancelledAt == nil {
				t.Fatal("cancelled_at not stamped")
			}
		})
	}
}

func TestAttachOutputFilesFirstWriterWins(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, _ := startBatch(t, s, uniqueID("acct"), 1)

			first := uniqueID("file-out")
			second := uniqueID("file-out")
			errFile := uniqueID("file-err")

			ok, err := s.AttachOutputFiles(b.ID, &first, &errFile)
			if err != nil || !ok {
				t.Fatalf("AttachOutputFiles = %v, %v; want true, nil", ok, err)
			}
			ok, err = s.AttachOutputFiles(b.ID, &second, nil)
			if err != nil {
				t.Fatalf("second AttachOutputFiles: %v", err)
			}
			if ok {
				t.Fatal("second AttachOutputFiles = true, want false (first writer wins)")
			}

			got, _ := s.GetBatch(b.AccountID, b.ID)
			if got.OutputFileID == nil || *got.OutputFileID != first {
				t.Fatalf("output_file_id = %v, want %q", got.OutputFileID, first)
			}
			if got.ErrorFileID == nil || *got.ErrorFileID != errFile {
				t.Fatalf("error_file_id = %v, want %q", got.ErrorFileID, errFile)
			}
		})
	}
}

func TestListOpenBatches(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			running, _ := startBatch(t, s, acct, 1)
			cancelling, _ := startBatch(t, s, acct, 1)
			if ok, err := s.SetBatchStatus(cancelling.ID, BatchInProgress, BatchCancelling, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("SetBatchStatus(cancelling) = %v, %v", ok, err)
			}
			done, _ := startBatch(t, s, acct, 1)
			if ok, err := s.SetBatchStatus(done.ID, BatchInProgress, BatchCompleted, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("SetBatchStatus(completed) = %v, %v", ok, err)
			}
			seedBatch(t, s, acct, 1) // still validating

			open, err := s.ListOpenBatches()
			if err != nil {
				t.Fatalf("ListOpenBatches: %v", err)
			}
			ids := map[string]bool{}
			for _, b := range open {
				ids[b.ID] = true
			}
			if !ids[running.ID] || !ids[cancelling.ID] {
				t.Fatalf("open batches %v missing %q or %q", ids, running.ID, cancelling.ID)
			}
			if ids[done.ID] {
				t.Fatalf("completed batch %q reported as open", done.ID)
			}
			if len(open) != 2 {
				t.Fatalf("open batches = %d, want 2", len(open))
			}
		})
	}
}

func TestListBatchesIsScopedPagedAndNewestFirst(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			other := uniqueID("acct")

			var created []*Batch
			for range 3 {
				b, _ := seedBatch(t, s, acct, 1)
				created = append(created, b)
				time.Sleep(2 * time.Millisecond)
			}
			seedBatch(t, s, other, 1)

			page, err := s.ListBatches(acct, 2, "")
			if err != nil {
				t.Fatalf("ListBatches: %v", err)
			}
			if len(page) != 2 {
				t.Fatalf("page = %d batches, want 2", len(page))
			}
			if page[0].ID != created[2].ID || page[1].ID != created[1].ID {
				t.Fatalf("page order = %q, %q; want %q, %q",
					page[0].ID, page[1].ID, created[2].ID, created[1].ID)
			}

			next, err := s.ListBatches(acct, 2, page[1].ID)
			if err != nil {
				t.Fatalf("ListBatches(after): %v", err)
			}
			if len(next) != 1 || next[0].ID != created[0].ID {
				t.Fatalf("next page = %+v, want [%q]", next, created[0].ID)
			}

			for _, b := range page {
				if b.AccountID != acct {
					t.Fatalf("ListBatches leaked account %q", b.AccountID)
				}
			}
		})
	}
}

func TestListBatchesDefaultsLimitTo20(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			for range 25 {
				seedBatch(t, s, acct, 0)
			}

			page, err := s.ListBatches(acct, 0, "")
			if err != nil {
				t.Fatalf("ListBatches: %v", err)
			}
			if len(page) != 20 {
				t.Fatalf("page = %d batches, want 20 (default limit)", len(page))
			}
		})
	}
}

func TestListBatchesUnknownCursorReturnsFirstPage(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			b, _ := seedBatch(t, s, acct, 0)

			page, err := s.ListBatches(acct, 10, uniqueID("missing-batch"))
			if err != nil {
				t.Fatalf("ListBatches(unknown cursor): %v", err)
			}
			if len(page) != 1 || page[0].ID != b.ID {
				t.Fatalf("page = %+v, want [%q] (first page)", page, b.ID)
			}
		})
	}
}

// TestListBatchesCursorIsAccountScoped covers a cursor that names a real
// batch belonging to a different account: it must not resolve, since a
// resolved cursor would let one account probe another's page boundaries.
func TestListBatchesCursorIsAccountScoped(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			other := uniqueID("acct")

			mine, _ := seedBatch(t, s, acct, 0)
			theirs, _ := seedBatch(t, s, other, 0)

			page, err := s.ListBatches(acct, 10, theirs.ID)
			if err != nil {
				t.Fatalf("ListBatches: %v", err)
			}
			if len(page) != 1 || page[0].ID != mine.ID {
				t.Fatalf("page = %+v, want [%q] (cross-account cursor ignored)", page, mine.ID)
			}
		})
	}
}

func TestCompletionRateWindow(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			now := time.Now().UTC()

			if _, known := s.CompletionRate(time.Minute, now); known {
				t.Fatal("CompletionRate reported known with no finished items")
			}

			b, _ := startBatch(t, s, uniqueID("acct"), 13)
			claimed, err := s.ClaimPendingItems(b.ID, 13, now)
			if err != nil || len(claimed) != 13 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			for i := 0; i < 12; i++ {
				if ok, err := s.FinishItem(ItemResult{ItemID: claimed[i].ID, Succeeded: true}, now.Add(-time.Duration(i)*time.Second)); err != nil || !ok {
					t.Fatalf("FinishItem(%d) = %v, %v", i, ok, err)
				}
			}
			// Outside the window: must not count.
			if ok, err := s.FinishItem(ItemResult{ItemID: claimed[12].ID, Succeeded: true}, now.Add(-10*time.Minute)); err != nil || !ok {
				t.Fatalf("FinishItem(old) = %v, %v", ok, err)
			}

			rate, known := s.CompletionRate(time.Minute, now)
			if !known {
				t.Fatal("CompletionRate reported unknown with 12 finished items")
			}
			if rate < 0.199 || rate > 0.201 {
				t.Fatalf("rate = %v, want 0.2/s", rate)
			}
		})
	}
}

// TestCompletionRateExcludesExpiredAndCancelled guards against counting an
// expired or cancelled item as finished: closeOpenItems stamps finished_at on
// both, but neither is a completion.
func TestCompletionRateExcludesExpiredAndCancelled(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			now := time.Now().UTC()

			b, _ := startBatch(t, s, uniqueID("acct"), 2)
			claimed, err := s.ClaimPendingItems(b.ID, 2, now)
			if err != nil || len(claimed) != 2 {
				t.Fatalf("ClaimPendingItems = %d items, %v", len(claimed), err)
			}
			if ok, err := s.FinishItem(ItemResult{ItemID: claimed[0].ID, Succeeded: true}, now); err != nil || !ok {
				t.Fatalf("FinishItem = %v, %v", ok, err)
			}
			// The other item expires rather than finishing; it must not count.
			if _, err := s.ExpireOpenItems(b.ID, now); err != nil {
				t.Fatalf("ExpireOpenItems: %v", err)
			}

			rate, known := s.CompletionRate(time.Minute, now)
			if !known {
				t.Fatal("CompletionRate reported unknown with one succeeded item")
			}
			if want := 1.0 / 60.0; rate < want-0.0001 || rate > want+0.0001 {
				t.Fatalf("rate = %v, want %v (expired item must not count)", rate, want)
			}
		})
	}
}

func TestBatchFileLifecycle(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			acct := uniqueID("acct")
			now := time.Now().UTC()

			f := &BatchFile{
				ID:        uniqueID("file"),
				AccountID: acct,
				Purpose:   "batch_output",
				Filename:  "output.jsonl",
				SizeBytes: 4096,
				CreatedAt: now.Add(-8 * 24 * time.Hour),
				SealedBy:  "coordinator",
			}
			f.BlobRef = f.ID
			if err := s.CreateBatchFile(f); err != nil {
				t.Fatalf("CreateBatchFile: %v", err)
			}

			got, ok := s.GetBatchFile(acct, f.ID)
			if !ok {
				t.Fatal("GetBatchFile: not found")
			}
			if got.Purpose != "batch_output" || got.SizeBytes != 4096 || got.BlobRef != f.ID {
				t.Fatalf("round-trip mismatch: %+v", got)
			}
			if got.PurgedAt != nil {
				t.Fatalf("purged_at = %v on a fresh file, want nil", got.PurgedAt)
			}
			if _, ok := s.GetBatchFile("someone-else", f.ID); ok {
				t.Fatal("GetBatchFile returned a file for the wrong account")
			}

			fresh := &BatchFile{
				ID:        uniqueID("file"),
				AccountID: acct,
				Purpose:   "batch",
				Filename:  "requests.jsonl",
				SizeBytes: 10,
				CreatedAt: now,
				SealedBy:  "coordinator",
			}
			fresh.BlobRef = fresh.ID
			if err := s.CreateBatchFile(fresh); err != nil {
				t.Fatalf("CreateBatchFile(fresh): %v", err)
			}

			purgeable, err := s.ListPurgeableFiles(now.Add(-7 * 24 * time.Hour))
			if err != nil {
				t.Fatalf("ListPurgeableFiles: %v", err)
			}
			if len(purgeable) != 1 || purgeable[0].ID != f.ID {
				t.Fatalf("purgeable = %+v, want [%q]", purgeable, f.ID)
			}

			if err := s.MarkBatchFilePurged(f.ID, now); err != nil {
				t.Fatalf("MarkBatchFilePurged: %v", err)
			}
			purgeable, err = s.ListPurgeableFiles(now.Add(-7 * 24 * time.Hour))
			if err != nil {
				t.Fatalf("ListPurgeableFiles (after purge): %v", err)
			}
			if len(purgeable) != 0 {
				t.Fatalf("purgeable after purge = %+v, want none", purgeable)
			}
			got, _ = s.GetBatchFile(acct, f.ID)
			if got.PurgedAt == nil {
				t.Fatal("purged_at not stamped")
			}
		})
	}
}

// contentColumnPattern names anything that could carry prompt or response
// content, a hash of it, a filename, a custom_id value, or metadata values.
// Batch tables hold ids, counts, timestamps, and state only; content lives in
// sealed blobs keyed by id (docs/design/tidal-batch-lane.md §3.3).
var contentColumnPattern = regexp.MustCompile(`(?i)body|prompt_text|content|text|payload|hash|digest|message`)

func TestNoContentColumnsInBatchTables(t *testing.T) {
	s := testPostgresStore(t)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	rows, err := s.pool.Query(ctx, `
		SELECT table_name, column_name
		  FROM information_schema.columns
		 WHERE table_schema = current_schema()
		   AND table_name IN ('batch_files', 'batches', 'batch_items')
		 ORDER BY table_name, ordinal_position`)
	if err != nil {
		t.Fatalf("query information_schema.columns: %v", err)
	}
	defer rows.Close()

	seen := map[string]int{}
	for rows.Next() {
		var table, column string
		if err := rows.Scan(&table, &column); err != nil {
			t.Fatalf("scan column row: %v", err)
		}
		seen[table]++
		if contentColumnPattern.MatchString(column) {
			t.Errorf("%s.%s may carry content — batch tables are metadata only", table, column)
		}
	}
	if err := rows.Err(); err != nil {
		t.Fatalf("iterate columns: %v", err)
	}
	for _, table := range []string{"batch_files", "batches", "batch_items"} {
		if seen[table] == 0 {
			t.Errorf("table %s was not created by migrate()", table)
		}
	}
}

// BatchItemExists is the dispatcher's orphan-sweep probe: it must answer for an
// item in any state, and answer false for an id no row carries, without
// depending on the batch or the account.
func TestBatchItemExists(t *testing.T) {
	for name, s := range storeBackends(t) {
		t.Run(name, func(t *testing.T) {
			b, items := startBatch(t, s, uniqueID("acct"), 3)

			for _, it := range items {
				ok, err := s.BatchItemExists(it.ID)
				if err != nil || !ok {
					t.Fatalf("BatchItemExists(%s pending) = %v, %v", it.ID, ok, err)
				}
			}

			// Still true once the item has moved on from pending.
			claimed, err := s.ClaimPendingItems(b.ID, 1, time.Now().UTC())
			if err != nil || len(claimed) != 1 {
				t.Fatalf("ClaimPendingItems = %d, %v", len(claimed), err)
			}
			if ok, err := s.FinishItem(ItemResult{ItemID: claimed[0].ID, Succeeded: true}, time.Now().UTC()); err != nil || !ok {
				t.Fatalf("FinishItem = %v, %v", ok, err)
			}
			if ok, err := s.BatchItemExists(claimed[0].ID); err != nil || !ok {
				t.Fatalf("BatchItemExists(succeeded) = %v, %v", ok, err)
			}

			if ok, err := s.BatchItemExists("bitem_000000000000000000000000"); err != nil || ok {
				t.Fatalf("BatchItemExists(unknown) = %v, %v, want false", ok, err)
			}
		})
	}
}
