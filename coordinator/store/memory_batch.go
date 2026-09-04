package store

// In-memory implementation of the batch-lane store methods (design §3.2).
// These mirror postgres_batch.go exactly: same status CAS, same claim order
// (line_no, only while the batch is in_progress), same idempotent settle that
// moves the batch counters with the item transition, all under the single
// MemoryStore mutex — which is what the Postgres backend gets from
// FOR UPDATE SKIP LOCKED and a transaction.

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

// cloneBatchFile returns a detached copy so callers cannot mutate stored state.
func cloneBatchFile(f *BatchFile) *BatchFile {
	cp := *f
	if f.PurgedAt != nil {
		at := *f.PurgedAt
		cp.PurgedAt = &at
	}
	return &cp
}

func cloneBatch(b *Batch) *Batch {
	cp := *b
	if b.InProgressAt != nil {
		at := *b.InProgressAt
		cp.InProgressAt = &at
	}
	if b.CompletedAt != nil {
		at := *b.CompletedAt
		cp.CompletedAt = &at
	}
	if b.CancelledAt != nil {
		at := *b.CancelledAt
		cp.CancelledAt = &at
	}
	if b.OutputFileID != nil {
		id := *b.OutputFileID
		cp.OutputFileID = &id
	}
	if b.ErrorFileID != nil {
		id := *b.ErrorFileID
		cp.ErrorFileID = &id
	}
	if b.Metadata != nil {
		cp.Metadata = make(map[string]string, len(b.Metadata))
		for k, v := range b.Metadata {
			cp.Metadata[k] = v
		}
	}
	return &cp
}

func cloneBatchItem(it *BatchItem) *BatchItem {
	cp := *it
	if it.SubmittedAt != nil {
		at := *it.SubmittedAt
		cp.SubmittedAt = &at
	}
	if it.FinishedAt != nil {
		at := *it.FinishedAt
		cp.FinishedAt = &at
	}
	return &cp
}

// --- BatchFileStore ---

// CreateBatchFile records a file whose sealed bytes are already on disk.
func (s *MemoryStore) CreateBatchFile(f *BatchFile) error {
	if f == nil {
		return errors.New("store: batch file is required")
	}
	if f.ID == "" {
		return errors.New("store: batch file id is required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.batchFiles[f.ID]; exists {
		return fmt.Errorf("store: batch file %q already exists", f.ID)
	}
	cp := cloneBatchFile(f)
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}
	s.batchFiles[cp.ID] = cp
	return nil
}

// GetBatchFile returns the file if it belongs to accountID.
func (s *MemoryStore) GetBatchFile(accountID, id string) (*BatchFile, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	f, ok := s.batchFiles[id]
	if !ok || f.AccountID != accountID {
		return nil, false
	}
	return cloneBatchFile(f), true
}

// MarkBatchFilePurged stamps purged_at. Marking an already-purged or unknown
// file is a no-op so a retention sweep can run twice.
func (s *MemoryStore) MarkBatchFilePurged(id string, at time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	f, ok := s.batchFiles[id]
	if !ok || f.PurgedAt != nil {
		return nil
	}
	stamp := at
	f.PurgedAt = &stamp
	return nil
}

// ListPurgeableFiles returns unpurged files created before the cutoff, oldest
// first.
func (s *MemoryStore) ListPurgeableFiles(before time.Time) ([]*BatchFile, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := []*BatchFile{}
	for _, f := range s.batchFiles {
		if f.PurgedAt != nil || !f.CreatedAt.Before(before) {
			continue
		}
		out = append(out, cloneBatchFile(f))
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID < out[j].ID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	return out, nil
}

// --- BatchStore ---

// CreateBatch writes the batch and all of its items under one lock, so a
// concurrent claim never sees a half-written batch.
func (s *MemoryStore) CreateBatch(b *Batch, items []*BatchItem) error {
	if b == nil {
		return errors.New("store: batch is required")
	}
	if b.ID == "" {
		return errors.New("store: batch id is required")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if _, exists := s.batches[b.ID]; exists {
		return fmt.Errorf("store: batch %q already exists", b.ID)
	}
	if err := checkDuplicateCustomIDs(items); err != nil {
		return err
	}

	cp := cloneBatch(b)
	cp.Status = BatchValidating
	if cp.CreatedAt.IsZero() {
		cp.CreatedAt = time.Now().UTC()
	}

	stored := make([]*BatchItem, 0, len(items))
	for _, it := range items {
		if it == nil || it.ID == "" {
			return errors.New("store: batch item id is required")
		}
		if _, exists := s.batchItems[it.ID]; exists {
			return fmt.Errorf("store: batch item %q already exists", it.ID)
		}
		itemCopy := cloneBatchItem(it)
		itemCopy.BatchID = cp.ID
		itemCopy.State = ItemPending
		stored = append(stored, itemCopy)
	}
	sortBatchItemsByLineNo(stored)

	s.batches[cp.ID] = cp
	s.batchItemsByBatch[cp.ID] = stored
	for _, it := range stored {
		s.batchItems[it.ID] = it
	}
	return nil
}

// GetBatch returns the batch if it belongs to accountID.
func (s *MemoryStore) GetBatch(accountID, id string) (*Batch, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	b, ok := s.batches[id]
	if !ok || b.AccountID != accountID {
		return nil, false
	}
	return cloneBatch(b), true
}

// ListBatches returns an account's batches newest first, paged by the batch id
// in after.
func (s *MemoryStore) ListBatches(accountID string, limit int, after string) ([]*Batch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if limit <= 0 {
		limit = 20
	}

	scoped := make([]*Batch, 0, len(s.batches))
	for _, b := range s.batches {
		if b.AccountID == accountID {
			scoped = append(scoped, b)
		}
	}
	sort.Slice(scoped, func(i, j int) bool {
		if scoped[i].CreatedAt.Equal(scoped[j].CreatedAt) {
			return scoped[i].ID > scoped[j].ID
		}
		return scoped[i].CreatedAt.After(scoped[j].CreatedAt)
	})

	start := 0
	if after != "" {
		for i, b := range scoped {
			if b.ID == after {
				start = i + 1
				break
			}
		}
	}
	out := []*Batch{}
	for _, b := range scoped[start:] {
		if len(out) >= limit {
			break
		}
		out = append(out, cloneBatch(b))
	}
	return out, nil
}

// SetBatchStatus compare-and-sets the status and stamps the matching timestamp.
func (s *MemoryStore) SetBatchStatus(id string, from, to BatchStatus, at time.Time) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	b, ok := s.batches[id]
	if !ok || b.Status != from {
		return false, nil
	}
	b.Status = to
	stamp := at
	switch to {
	case BatchInProgress:
		b.InProgressAt = &stamp
	case BatchCompleted, BatchFailed, BatchExpired:
		b.CompletedAt = &stamp
	case BatchCancelled:
		b.CancelledAt = &stamp
	}
	return true, nil
}

// AttachOutputFiles records the assembled result files, first writer wins.
func (s *MemoryStore) AttachOutputFiles(id string, outputFileID, errorFileID *string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	b, ok := s.batches[id]
	if !ok || b.OutputFileID != nil || b.ErrorFileID != nil {
		return false, nil
	}
	if outputFileID != nil {
		out := *outputFileID
		b.OutputFileID = &out
	}
	if errorFileID != nil {
		errFile := *errorFileID
		b.ErrorFileID = &errFile
	}
	return true, nil
}

// ListOpenBatches returns every in_progress or cancelling batch, oldest first.
func (s *MemoryStore) ListOpenBatches() ([]*Batch, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	out := []*Batch{}
	for _, b := range s.batches {
		if b.Status == BatchInProgress || b.Status == BatchCancelling {
			out = append(out, cloneBatch(b))
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].CreatedAt.Equal(out[j].CreatedAt) {
			return out[i].ID < out[j].ID
		}
		return out[i].CreatedAt.Before(out[j].CreatedAt)
	})
	return out, nil
}

// CompletionRate counts items that finished in [now-window, now] across every
// batch and divides by the window.
func (s *MemoryStore) CompletionRate(window time.Duration, now time.Time) (float64, bool) {
	if window <= 0 {
		return 0, false
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	since := now.Add(-window)
	finished := 0
	for _, it := range s.batchItems {
		if it.State != ItemSucceeded && it.State != ItemFailed {
			continue
		}
		if it.FinishedAt == nil {
			continue
		}
		if it.FinishedAt.Before(since) || it.FinishedAt.After(now) {
			continue
		}
		finished++
	}
	if finished == 0 {
		return 0, false
	}
	return float64(finished) / window.Seconds(), true
}

// --- BatchItemStore ---

// ClaimPendingItems moves up to limit pending items to inflight in line_no
// order, but only while the batch is in_progress.
func (s *MemoryStore) ClaimPendingItems(batchID string, limit int, at time.Time) ([]*BatchItem, error) {
	if limit <= 0 {
		return []*BatchItem{}, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	b, ok := s.batches[batchID]
	if !ok || b.Status != BatchInProgress {
		return []*BatchItem{}, nil
	}

	claimed := []*BatchItem{}
	for _, it := range s.batchItemsByBatch[batchID] {
		if len(claimed) >= limit {
			break
		}
		if it.State != ItemPending {
			continue
		}
		it.State = ItemInflight
		it.Attempts++
		stamp := at
		it.SubmittedAt = &stamp
		claimed = append(claimed, cloneBatchItem(it))
	}
	return claimed, nil
}

// ReleaseItem returns one inflight item to pending and un-counts its claim.
func (s *MemoryStore) ReleaseItem(itemID string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	it, ok := s.batchItems[itemID]
	if !ok || it.State != ItemInflight {
		return false, nil
	}
	it.State = ItemPending
	if it.Attempts > 0 {
		it.Attempts--
	}
	it.SubmittedAt = nil
	return true, nil
}

// RequeueInflightItems returns every inflight item of a batch to pending,
// keeping the attempt count.
func (s *MemoryStore) RequeueInflightItems(batchID string) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := 0
	for _, it := range s.batchItemsByBatch[batchID] {
		if it.State != ItemInflight {
			continue
		}
		it.State = ItemPending
		it.SubmittedAt = nil
		n++
	}
	return n, nil
}

// FinishItem settles one inflight item and moves the batch counter with it.
func (s *MemoryStore) FinishItem(r ItemResult, at time.Time) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	it, ok := s.batchItems[r.ItemID]
	if !ok || it.State != ItemInflight {
		return false, nil
	}

	if r.Succeeded {
		it.State = ItemSucceeded
	} else {
		it.State = ItemFailed
	}
	it.LastErrorCode = r.ErrorCode
	it.PromptTokens = r.PromptTokens
	it.CompletionTokens = r.CompletionTokens
	it.RequestID = r.RequestID
	it.ResultBlobRef = r.ResultBlobRef
	stamp := at
	it.FinishedAt = &stamp

	if b, ok := s.batches[it.BatchID]; ok {
		if r.Succeeded {
			b.CountsCompleted++
		} else {
			b.CountsFailed++
		}
	}
	return true, nil
}

// ExpireOpenItems moves pending and inflight items to expired.
func (s *MemoryStore) ExpireOpenItems(batchID string, at time.Time) (int, error) {
	return s.closeOpenItems(batchID, ItemExpired, at)
}

// CancelOpenItems moves pending and inflight items to cancelled.
func (s *MemoryStore) CancelOpenItems(batchID string, at time.Time) (int, error) {
	return s.closeOpenItems(batchID, ItemCancelled, at)
}

// closeOpenItems is the shared body of ExpireOpenItems and CancelOpenItems:
// both drain the open states into a terminal one that counts in neither
// counts_completed nor counts_failed.
func (s *MemoryStore) closeOpenItems(batchID string, to ItemState, at time.Time) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := 0
	for _, it := range s.batchItemsByBatch[batchID] {
		if it.State != ItemPending && it.State != ItemInflight {
			continue
		}
		it.State = to
		stamp := at
		it.FinishedAt = &stamp
		n++
	}
	return n, nil
}

// ListItems returns a batch's items in line_no order, filtered to states when
// any are given.
func (s *MemoryStore) ListItems(batchID string, states ...ItemState) ([]*BatchItem, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	want := make(map[ItemState]struct{}, len(states))
	for _, st := range states {
		want[st] = struct{}{}
	}

	out := []*BatchItem{}
	for _, it := range s.batchItemsByBatch[batchID] {
		if len(want) > 0 {
			if _, ok := want[it.State]; !ok {
				continue
			}
		}
		out = append(out, cloneBatchItem(it))
	}
	return out, nil
}

// CountItems returns the item tallies for one batch.
func (s *MemoryStore) CountItems(batchID string) (total, pending, inflight, succeeded, failed int, err error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	for _, it := range s.batchItemsByBatch[batchID] {
		total++
		switch it.State {
		case ItemPending:
			pending++
		case ItemInflight:
			inflight++
		case ItemSucceeded:
			succeeded++
		case ItemFailed:
			failed++
		}
	}
	return total, pending, inflight, succeeded, failed, nil
}
