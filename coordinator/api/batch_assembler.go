package api

// Output assembly and retention for the batch lane
// (docs/design/tidal-batch-lane.md §3.3, §3.6).
//
// FinalizeBatchIfDone is the one place a batch reaches a terminal state. The
// dispatcher calls it after every settled item and from its sweep; it is a
// no-op unless the batch is open and nothing is left pending or in flight, so
// calling it too often is free and calling it twice is safe.
//
// Result blobs are written by the dispatcher keyed by item id before the item
// is settled, so a crash between the two leaves a rewritable blob rather than a
// lost result. Item input blobs are dropped as soon as the batch finalizes;
// result blobs and the assembled files live until the retention sweep, never
// until retrieval, so a failed download stays retryable.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// Terminal error codes an item can carry, and the fixed message each renders
// as. Provider text never reaches an error file.
// batchRequestFailedCode (batch_dispatch.go) is the third member of this set;
// it is the code the dispatch funnel already settles a failed attempt with.
const (
	batchItemErrorExpired   = "batch_expired"
	batchItemErrorCancelled = "batch_cancelled"
)

var batchItemErrorMessages = map[string]string{
	batchRequestFailedCode:  "The request could not be completed by any provider.",
	batchItemErrorExpired:   "The batch expired before this request was processed.",
	batchItemErrorCancelled: "The batch was cancelled before this request was processed.",
}

// BatchOutputRetention is how long an assembled output or error file, and the
// result blobs behind it, survive after the batch completes.
const BatchOutputRetention = 7 * 24 * time.Hour

// FinalizeResult reports what a finalize pass did. It is nil when the batch was
// not ready (or not open), which is the common case on a busy tick.
type FinalizeResult struct {
	Status       store.BatchStatus
	OutputFileID *string
	ErrorFileID  *string
}

// FinalizeBatchIfDone completes a batch whose items have all settled. It
// returns (nil, nil) when the batch is not open or still has pending or
// in-flight items.
//
// This is the entry point the batch dispatcher calls after every FinishItem and
// from its periodic sweep.
func (s *Server) FinalizeBatchIfDone(batchID string, now time.Time) (*FinalizeResult, error) {
	blobs := s.batchBlobs
	if blobs == nil {
		return nil, errors.New("batch: the batch lane is not configured")
	}

	// ListOpenBatches is the only unscoped read of a batch the store offers,
	// and its in_progress/cancelling filter is exactly finalize's precondition.
	open, err := s.store.ListOpenBatches()
	if err != nil {
		return nil, fmt.Errorf("batch: list open batches: %w", err)
	}
	var batch *store.Batch
	for _, b := range open {
		if b.ID == batchID {
			batch = b
			break
		}
	}
	if batch == nil {
		return nil, nil
	}

	_, pending, inflight, _, _, err := s.store.CountItems(batchID)
	if err != nil {
		return nil, fmt.Errorf("batch: count items: %w", err)
	}
	if pending+inflight > 0 {
		return nil, nil
	}

	items, err := s.store.ListItems(batchID)
	if err != nil {
		return nil, fmt.Errorf("batch: list items: %w", err)
	}
	outputJSONL, errorJSONL, err := s.assembleBatchFiles(blobs, batch, items)
	if err != nil {
		return nil, err
	}

	outputFileID, err := s.storeAssembledFile(blobs, batch, batchFilePurposeOutput, outputJSONL, now)
	if err != nil {
		return nil, err
	}
	errorFileID, err := s.storeAssembledFile(blobs, batch, batchFilePurposeError, errorJSONL, now)
	if err != nil {
		s.discardAssembledFile(blobs, outputFileID, now)
		return nil, err
	}

	if outputFileID != nil || errorFileID != nil {
		attached, err := s.store.AttachOutputFiles(batchID, outputFileID, errorFileID)
		if err != nil {
			return nil, fmt.Errorf("batch: attach output files: %w", err)
		}
		if !attached {
			// A concurrent finalize won. Drop the files this pass built so the
			// batch never points at one pair while another sits on disk.
			s.discardAssembledFile(blobs, outputFileID, now)
			s.discardAssembledFile(blobs, errorFileID, now)
			if current, ok := s.store.GetBatch(batch.AccountID, batchID); ok {
				return &FinalizeResult{Status: current.Status, OutputFileID: current.OutputFileID, ErrorFileID: current.ErrorFileID}, nil
			}
			return nil, nil
		}
	}

	target := store.BatchCompleted
	if batch.Status == store.BatchCancelling {
		target = store.BatchCancelled
	}
	moved, err := s.store.SetBatchStatus(batchID, batch.Status, target, now)
	if err != nil {
		return nil, fmt.Errorf("batch: finalize status: %w", err)
	}
	if !moved {
		return nil, nil
	}

	// The prompts are no longer needed: the results are assembled and the batch
	// can never dispatch again.
	for _, it := range items {
		if it.BlobRef == "" {
			continue
		}
		if err := blobs.Delete(it.BlobRef); err != nil {
			s.logger.Error("batch: deleting an item input blob failed", "item_id", it.ID, "error", err)
		}
	}

	s.logger.Info("batch: finalized",
		"batch_id", batchID, "status", string(target),
		"completed", batch.CountsCompleted, "failed", batch.CountsFailed, "total", batch.CountsTotal)

	return &FinalizeResult{Status: target, OutputFileID: outputFileID, ErrorFileID: errorFileID}, nil
}

// assembleBatchFiles builds the output and error JSONL bodies. Succeeded items
// go to the output file in line_no order; failed and expired items go to the
// error file. Cancelled items appear in neither — they were never attempted.
func (s *Server) assembleBatchFiles(blobs *sealedblob.Store, batch *store.Batch, items []*store.BatchItem) ([]byte, []byte, error) {
	var output, errorFile bytes.Buffer
	for _, it := range items {
		switch it.State {
		case store.ItemSucceeded:
			body, err := s.readItemResult(blobs, batch, it)
			if err != nil {
				return nil, nil, err
			}
			line := map[string]any{
				"id":        it.ID,
				"custom_id": it.CustomID,
				"response": itemResponse{
					StatusCode: http.StatusOK,
					RequestID:  it.RequestID,
					Body:       body,
				},
				"error": nil,
			}
			if err := writeJSONLine(&output, line); err != nil {
				return nil, nil, err
			}
		case store.ItemFailed, store.ItemExpired:
			line := map[string]any{
				"id":        it.ID,
				"custom_id": it.CustomID,
				"response":  nil,
				"error":     batchItemError(it),
			}
			if err := writeJSONLine(&errorFile, line); err != nil {
				return nil, nil, err
			}
		}
	}
	return output.Bytes(), errorFile.Bytes(), nil
}

// readItemResult returns the bytes that become one output line's response.body.
// A coordinator-sealed batch yields the plain response JSON; a consumer-sealed
// batch yields the e2e.EncryptedPayload object verbatim, which the coordinator
// cannot open by construction.
func (s *Server) readItemResult(blobs *sealedblob.Store, batch *store.Batch, it *store.BatchItem) (json.RawMessage, error) {
	ref := it.ResultBlobRef
	if ref == "" {
		ref = BatchItemResultRef(it.ID)
	}
	var (
		raw []byte
		err error
	)
	if batch.SealedTo == sealedToConsumer {
		raw, err = blobs.Raw(ref)
	} else {
		raw, err = blobs.Open(ref)
	}
	if err != nil {
		if errors.Is(err, sealedblob.ErrNotFound) {
			// The item settled as succeeded but its blob is gone (a purge that
			// raced a late finalize). Emit JSON null rather than losing the line.
			s.logger.Warn("batch: a succeeded item has no result blob", "batch_id", batch.ID, "item_id", it.ID)
			return json.RawMessage("null"), nil
		}
		return nil, fmt.Errorf("batch: read item result: %w", err)
	}
	if !json.Valid(raw) {
		s.logger.Error("batch: an item result is not valid JSON", "batch_id", batch.ID, "item_id", it.ID)
		return json.RawMessage("null"), nil
	}
	return json.RawMessage(raw), nil
}

// batchItemError renders the fixed error body for a terminal item.
func batchItemError(it *store.BatchItem) itemErrorBody {
	code := it.LastErrorCode
	if it.State == store.ItemExpired {
		code = batchItemErrorExpired
	}
	if _, known := batchItemErrorMessages[code]; !known {
		code = batchRequestFailedCode
	}
	return itemErrorBody{Code: code, Message: batchItemErrorMessages[code]}
}

func writeJSONLine(buf *bytes.Buffer, line map[string]any) error {
	encoded, err := json.Marshal(line)
	if err != nil {
		return fmt.Errorf("batch: encode result line: %w", err)
	}
	buf.Write(encoded)
	buf.WriteByte('\n')
	return nil
}

// storeAssembledFile seals one assembled JSONL body and records its row. Empty
// content produces no file at all, which is what a batch with no failures (or
// no successes) should report.
func (s *Server) storeAssembledFile(blobs *sealedblob.Store, batch *store.Batch, purpose string, content []byte, now time.Time) (*string, error) {
	if len(content) == 0 {
		return nil, nil
	}
	fileID, err := newBatchID("file-")
	if err != nil {
		return nil, err
	}
	if err := blobs.PutPlain(fileID, content); err != nil {
		return nil, fmt.Errorf("batch: seal %s file: %w", purpose, err)
	}
	rec := &store.BatchFile{
		ID:        fileID,
		AccountID: batch.AccountID,
		Purpose:   purpose,
		// The batch id is the coordinator's own identifier; no consumer string
		// ever reaches a filename.
		Filename:  batch.ID + "_" + purpose + ".jsonl",
		SizeBytes: int64(len(content)),
		CreatedAt: now,
		BlobRef:   fileID,
		SealedBy:  "coordinator",
	}
	if err := s.store.CreateBatchFile(rec); err != nil {
		if delErr := blobs.Delete(fileID); delErr != nil {
			s.logger.Error("batch: removing an orphaned result blob failed", "file_id", fileID, "error", delErr)
		}
		return nil, fmt.Errorf("batch: record %s file: %w", purpose, err)
	}
	return &fileID, nil
}

// discardAssembledFile removes a file this finalize pass built but lost the
// race to attach.
func (s *Server) discardAssembledFile(blobs *sealedblob.Store, fileID *string, now time.Time) {
	if fileID == nil {
		return
	}
	if err := blobs.Delete(*fileID); err != nil {
		s.logger.Error("batch: discarding a losing result blob failed", "file_id", *fileID, "error", err)
	}
	if err := s.store.MarkBatchFilePurged(*fileID, now); err != nil {
		s.logger.Error("batch: marking a losing result file purged failed", "file_id", *fileID, "error", err)
	}
}

// inlineBatchResults renders the OpenRouter inline results array for a
// terminal inline batch. It reads the same result blobs the output file is
// assembled from, so both surfaces agree.
func (s *Server) inlineBatchResults(batch *store.Batch) ([]inlineResult, error) {
	blobs := s.batchBlobs
	if blobs == nil {
		return nil, errors.New("batch: the batch lane is not configured")
	}
	items, err := s.store.ListItems(batch.ID)
	if err != nil {
		return nil, fmt.Errorf("batch: list items: %w", err)
	}
	results := make([]inlineResult, 0, len(items))
	for _, it := range items {
		switch it.State {
		case store.ItemSucceeded:
			body, err := s.readItemResult(blobs, batch, it)
			if err != nil {
				return nil, err
			}
			results = append(results, inlineResult{
				ID:       it.ID,
				CustomID: it.CustomID,
				Response: &itemResponse{StatusCode: http.StatusOK, RequestID: it.RequestID, Body: body},
			})
		case store.ItemFailed, store.ItemExpired, store.ItemCancelled:
			code := batchItemErrorCancelled
			if it.State != store.ItemCancelled {
				code = batchItemError(it).Code
			}
			results = append(results, inlineResult{
				ID:       it.ID,
				CustomID: it.CustomID,
				Error:    &itemErrorBody{Code: code, Message: batchItemErrorMessages[code]},
			})
		}
	}
	return results, nil
}

// PurgeExpiredBatchFiles deletes the sealed bytes of every batch file past its
// retention horizon and marks the row purged, leaving the file's metadata
// answerable. It is idempotent, so the dispatcher's sweep can call it on every
// tick, and it returns the number of files whose content it dropped.
//
// Retention is measured from the file's creation, which for an assembled output
// or error file is the moment the batch completed.
func (s *Server) PurgeExpiredBatchFiles(now time.Time) (int, error) {
	blobs := s.batchBlobs
	if blobs == nil {
		return 0, nil
	}
	files, err := s.store.ListPurgeableFiles(now.Add(-BatchOutputRetention))
	if err != nil {
		return 0, fmt.Errorf("batch: list purgeable files: %w", err)
	}
	purged := 0
	for _, f := range files {
		if err := blobs.Delete(f.BlobRef); err != nil {
			s.logger.Error("batch: purging a file blob failed", "file_id", f.ID, "error", err)
			continue
		}
		if err := s.store.MarkBatchFilePurged(f.ID, now); err != nil {
			s.logger.Error("batch: marking a purged file failed", "file_id", f.ID, "error", err)
			continue
		}
		purged++
	}
	if purged > 0 {
		s.logger.Info("batch: retention sweep purged file content", "files", purged)
	}
	return purged, nil
}
