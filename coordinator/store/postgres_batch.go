package store

// Postgres implementation of the batch-lane store methods (design §3.2).
// The DDL lives in the migrate() slice in postgres.go. Every table here is
// metadata only — bodies and results are sealed blobs on disk keyed by id — so
// no statement in this file reads or writes content, and no error string
// carries a custom_id, a filename, or a metadata value.
//
// Concurrency: the claim is one UPDATE over a FOR UPDATE SKIP LOCKED
// sub-select, so two coordinators ticking at the same time split the pending
// items instead of double-dispatching them. FinishItem runs the item
// transition and the batch counter bump in one transaction, gated on the item
// still being inflight, so a duplicate or late result moves nothing.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/jackc/pgx/v5"
)

// batchItemColumns is the column list every item read returns, in the order
// scanBatchItem expects.
const batchItemColumns = `id, batch_id, custom_id, line_no, state, attempts, last_error_code,
	prompt_tokens, completion_tokens, submitted_at, finished_at, request_id, blob_ref, result_blob_ref`

// batchColumns is the column list every batch read returns, in the order
// scanBatch expects.
const batchColumns = `id, account_id, input_file_id, endpoint, status, completion_window,
	created_at, expires_at, in_progress_at, completed_at, cancelled_at,
	counts_total, counts_completed, counts_failed, output_file_id, error_file_id,
	result_public_key, sealed_to, source, model, metadata_json`

// marshalBatchMetadata encodes a batch's metadata map for the JSONB column. A
// nil map becomes {} rather than JSON null so a round trip always returns a
// usable map.
func marshalBatchMetadata(metadata map[string]string) ([]byte, error) {
	if metadata == nil {
		metadata = map[string]string{}
	}
	data, err := json.Marshal(metadata)
	if err != nil {
		return nil, fmt.Errorf("store: marshal batch metadata: %w", err)
	}
	return data, nil
}

func scanBatchItem(row pgx.Row) (*BatchItem, error) {
	var it BatchItem
	err := row.Scan(&it.ID, &it.BatchID, &it.CustomID, &it.LineNo, &it.State, &it.Attempts,
		&it.LastErrorCode, &it.PromptTokens, &it.CompletionTokens, &it.SubmittedAt,
		&it.FinishedAt, &it.RequestID, &it.BlobRef, &it.ResultBlobRef)
	if err != nil {
		return nil, err
	}
	return &it, nil
}

func scanBatch(row pgx.Row) (*Batch, error) {
	var (
		b        Batch
		metadata []byte
	)
	err := row.Scan(&b.ID, &b.AccountID, &b.InputFileID, &b.Endpoint, &b.Status, &b.CompletionWindow,
		&b.CreatedAt, &b.ExpiresAt, &b.InProgressAt, &b.CompletedAt, &b.CancelledAt,
		&b.CountsTotal, &b.CountsCompleted, &b.CountsFailed, &b.OutputFileID, &b.ErrorFileID,
		&b.ResultPublicKey, &b.SealedTo, &b.Source, &b.Model, &metadata)
	if err != nil {
		return nil, err
	}
	if len(metadata) > 0 {
		if err := json.Unmarshal(metadata, &b.Metadata); err != nil {
			return nil, fmt.Errorf("decode batch metadata: %w", err)
		}
	}
	return &b, nil
}

// --- BatchFileStore ---

// CreateBatchFile records a file whose sealed bytes are already on disk.
func (s *PostgresStore) CreateBatchFile(f *BatchFile) error {
	if f == nil {
		return errors.New("store: batch file is required")
	}
	if f.ID == "" {
		return errors.New("store: batch file id is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	createdAt := f.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}
	_, err := s.pool.Exec(ctx, `
		INSERT INTO batch_files (id, account_id, purpose, filename, size_bytes, created_at, blob_ref, sealed_by, purged_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)`,
		f.ID, f.AccountID, f.Purpose, f.Filename, f.SizeBytes, createdAt, f.BlobRef, f.SealedBy, f.PurgedAt)
	if err != nil {
		return fmt.Errorf("store: create batch file %q: %w", f.ID, err)
	}
	return nil
}

// GetBatchFile returns the file if it belongs to accountID.
func (s *PostgresStore) GetBatchFile(accountID, id string) (*BatchFile, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var f BatchFile
	err := s.pool.QueryRow(ctx, `
		SELECT id, account_id, purpose, filename, size_bytes, created_at, blob_ref, sealed_by, purged_at
		  FROM batch_files WHERE id = $1 AND account_id = $2`, id, accountID,
	).Scan(&f.ID, &f.AccountID, &f.Purpose, &f.Filename, &f.SizeBytes, &f.CreatedAt,
		&f.BlobRef, &f.SealedBy, &f.PurgedAt)
	if err != nil {
		return nil, false
	}
	return &f, true
}

// MarkBatchFilePurged stamps purged_at. The WHERE clause makes a repeat sweep a
// no-op rather than an overwrite.
func (s *PostgresStore) MarkBatchFilePurged(id string, at time.Time) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_, err := s.pool.Exec(ctx,
		`UPDATE batch_files SET purged_at = $2 WHERE id = $1 AND purged_at IS NULL`, id, at)
	if err != nil {
		return fmt.Errorf("store: mark batch file %q purged: %w", id, err)
	}
	return nil
}

// ListPurgeableFiles returns unpurged files created before the cutoff, oldest
// first.
func (s *PostgresStore) ListPurgeableFiles(before time.Time) ([]*BatchFile, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	rows, err := s.pool.Query(ctx, `
		SELECT id, account_id, purpose, filename, size_bytes, created_at, blob_ref, sealed_by, purged_at
		  FROM batch_files
		 WHERE purged_at IS NULL AND created_at < $1
		 ORDER BY created_at, id`, before)
	if err != nil {
		return nil, fmt.Errorf("store: list purgeable batch files: %w", err)
	}
	defer rows.Close()

	out := []*BatchFile{}
	for rows.Next() {
		var f BatchFile
		if err := rows.Scan(&f.ID, &f.AccountID, &f.Purpose, &f.Filename, &f.SizeBytes,
			&f.CreatedAt, &f.BlobRef, &f.SealedBy, &f.PurgedAt); err != nil {
			return nil, fmt.Errorf("store: scan purgeable batch file: %w", err)
		}
		out = append(out, &f)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate purgeable batch files: %w", err)
	}
	return out, nil
}

// --- BatchStore ---

// CreateBatch writes the batch and all of its items in one transaction, so a
// concurrent claim never sees a half-written batch.
func (s *PostgresStore) CreateBatch(b *Batch, items []*BatchItem) error {
	if b == nil {
		return errors.New("store: batch is required")
	}
	if b.ID == "" {
		return errors.New("store: batch id is required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	metadata, err := marshalBatchMetadata(b.Metadata)
	if err != nil {
		return err
	}
	createdAt := b.CreatedAt
	if createdAt.IsZero() {
		createdAt = time.Now().UTC()
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("store: begin create batch tx: %w", err)
	}
	defer tx.Rollback(ctx)

	if _, err := tx.Exec(ctx, `
		INSERT INTO batches (id, account_id, input_file_id, endpoint, status, completion_window,
			created_at, expires_at, counts_total, result_public_key, sealed_to, source, model, metadata_json)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`,
		b.ID, b.AccountID, b.InputFileID, b.Endpoint, BatchValidating, b.CompletionWindow,
		createdAt, b.ExpiresAt, b.CountsTotal, b.ResultPublicKey, b.SealedTo, b.Source, b.Model, metadata,
	); err != nil {
		return fmt.Errorf("store: insert batch %q: %w", b.ID, err)
	}

	for _, it := range items {
		if it == nil || it.ID == "" {
			return errors.New("store: batch item id is required")
		}
		if _, err := tx.Exec(ctx, `
			INSERT INTO batch_items (id, batch_id, custom_id, line_no, state, blob_ref)
			VALUES ($1, $2, $3, $4, $5, $6)`,
			it.ID, b.ID, it.CustomID, it.LineNo, ItemPending, it.BlobRef,
		); err != nil {
			return fmt.Errorf("store: insert batch item %q: %w", it.ID, err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return fmt.Errorf("store: commit create batch tx: %w", err)
	}
	return nil
}

// GetBatch returns the batch if it belongs to accountID.
func (s *PostgresStore) GetBatch(accountID, id string) (*Batch, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	b, err := scanBatch(s.pool.QueryRow(ctx,
		`SELECT `+batchColumns+` FROM batches WHERE id = $1 AND account_id = $2`, id, accountID))
	if err != nil {
		return nil, false
	}
	return b, true
}

// ListBatches returns an account's batches newest first. The cursor is
// (created_at, id) of the batch named by after, so a page boundary is stable
// even when two batches share a timestamp.
func (s *PostgresStore) ListBatches(accountID string, limit int, after string) ([]*Batch, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if limit <= 0 {
		limit = 20
	}

	query := `SELECT ` + batchColumns + ` FROM batches WHERE account_id = $1`
	args := []any{accountID}
	if after != "" {
		query += ` AND (created_at, id) < (SELECT created_at, id FROM batches WHERE id = $2)`
		args = append(args, after)
	}
	query += ` ORDER BY created_at DESC, id DESC LIMIT $` + fmt.Sprint(len(args)+1)
	args = append(args, limit)

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: list batches: %w", err)
	}
	defer rows.Close()

	out := []*Batch{}
	for rows.Next() {
		b, err := scanBatch(rows)
		if err != nil {
			return nil, fmt.Errorf("store: scan batch: %w", err)
		}
		out = append(out, b)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate batches: %w", err)
	}
	return out, nil
}

// SetBatchStatus compare-and-sets the status and stamps the matching timestamp
// in the same UPDATE, so a losing racer changes nothing.
func (s *PostgresStore) SetBatchStatus(id string, from, to BatchStatus, at time.Time) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tag, err := s.pool.Exec(ctx, `
		UPDATE batches SET
			status = $3,
			in_progress_at = CASE WHEN $3 = 'in_progress' THEN $4 ELSE in_progress_at END,
			completed_at = CASE WHEN $3 IN ('completed', 'failed', 'expired') THEN $4 ELSE completed_at END,
			cancelled_at = CASE WHEN $3 = 'cancelled' THEN $4 ELSE cancelled_at END
		WHERE id = $1 AND status = $2`, id, from, to, at)
	if err != nil {
		return false, fmt.Errorf("store: set batch %q status: %w", id, err)
	}
	return tag.RowsAffected() == 1, nil
}

// AttachOutputFiles records the assembled result files. The NULL guard on both
// columns is what makes it first-writer-wins.
func (s *PostgresStore) AttachOutputFiles(id string, outputFileID, errorFileID *string, at time.Time) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tag, err := s.pool.Exec(ctx, `
		UPDATE batches SET output_file_id = $2, error_file_id = $3
		 WHERE id = $1 AND output_file_id IS NULL AND error_file_id IS NULL`,
		id, outputFileID, errorFileID)
	if err != nil {
		return false, fmt.Errorf("store: attach output files to batch %q: %w", id, err)
	}
	return tag.RowsAffected() == 1, nil
}

// ListOpenBatches returns every in_progress or cancelling batch, oldest first.
func (s *PostgresStore) ListOpenBatches() ([]*Batch, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	rows, err := s.pool.Query(ctx,
		`SELECT `+batchColumns+` FROM batches
		  WHERE status IN ('in_progress', 'cancelling')
		  ORDER BY created_at, id`)
	if err != nil {
		return nil, fmt.Errorf("store: list open batches: %w", err)
	}
	defer rows.Close()

	out := []*Batch{}
	for rows.Next() {
		b, err := scanBatch(rows)
		if err != nil {
			return nil, fmt.Errorf("store: scan open batch: %w", err)
		}
		out = append(out, b)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate open batches: %w", err)
	}
	return out, nil
}

// CompletionRate counts items that finished in [now-window, now] across every
// batch and divides by the window.
func (s *PostgresStore) CompletionRate(window time.Duration, now time.Time) (float64, bool) {
	if window <= 0 {
		return 0, false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var finished int
	err := s.pool.QueryRow(ctx, `
		SELECT COUNT(*) FROM batch_items
		 WHERE finished_at >= $1 AND finished_at <= $2`, now.Add(-window), now,
	).Scan(&finished)
	if err != nil || finished == 0 {
		return 0, false
	}
	return float64(finished) / window.Seconds(), true
}

// --- BatchItemStore ---

// ClaimPendingItems moves up to limit pending items to inflight in line_no
// order. SKIP LOCKED lets concurrent ticks split the queue instead of blocking
// on each other, and the EXISTS guard keeps a cancelling, expired, or still
// validating batch from dispatching anything.
func (s *PostgresStore) ClaimPendingItems(batchID string, limit int, at time.Time) ([]*BatchItem, error) {
	if limit <= 0 {
		return []*BatchItem{}, nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	rows, err := s.pool.Query(ctx, `
		UPDATE batch_items SET state = 'inflight', attempts = attempts + 1, submitted_at = $3
		 WHERE id IN (
			SELECT id FROM batch_items
			 WHERE batch_id = $1 AND state = 'pending'
			   AND EXISTS (SELECT 1 FROM batches WHERE id = $1 AND status = 'in_progress')
			 ORDER BY line_no
			 LIMIT $2
			 FOR UPDATE SKIP LOCKED)
		RETURNING `+batchItemColumns, batchID, limit, at)
	if err != nil {
		return nil, fmt.Errorf("store: claim pending items for batch %q: %w", batchID, err)
	}
	defer rows.Close()

	claimed := []*BatchItem{}
	for rows.Next() {
		it, err := scanBatchItem(rows)
		if err != nil {
			return nil, fmt.Errorf("store: scan claimed item: %w", err)
		}
		claimed = append(claimed, it)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate claimed items: %w", err)
	}
	// RETURNING has no ordering guarantee; the dispatcher expects line_no order.
	sortBatchItemsByLineNo(claimed)
	return claimed, nil
}

// ReleaseItem returns one inflight item to pending and un-counts its claim.
func (s *PostgresStore) ReleaseItem(itemID string) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	tag, err := s.pool.Exec(ctx, `
		UPDATE batch_items
		   SET state = 'pending', attempts = GREATEST(attempts - 1, 0), submitted_at = NULL
		 WHERE id = $1 AND state = 'inflight'`, itemID)
	if err != nil {
		return false, fmt.Errorf("store: release batch item %q: %w", itemID, err)
	}
	return tag.RowsAffected() == 1, nil
}

// RequeueInflightItems returns every inflight item of a batch to pending,
// keeping the attempt count.
func (s *PostgresStore) RequeueInflightItems(batchID string) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tag, err := s.pool.Exec(ctx, `
		UPDATE batch_items SET state = 'pending', submitted_at = NULL
		 WHERE batch_id = $1 AND state = 'inflight'`, batchID)
	if err != nil {
		return 0, fmt.Errorf("store: requeue inflight items for batch %q: %w", batchID, err)
	}
	return int(tag.RowsAffected()), nil
}

// FinishItem settles one inflight item and moves the batch counter in the same
// transaction. The state guard on the item UPDATE is the idempotency gate: a
// duplicate or late result finds no inflight row, so neither statement runs.
func (s *PostgresStore) FinishItem(r ItemResult, at time.Time) (bool, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	state := ItemFailed
	completed, failed := 0, 1
	if r.Succeeded {
		state = ItemSucceeded
		completed, failed = 1, 0
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return false, fmt.Errorf("store: begin finish item tx: %w", err)
	}
	defer tx.Rollback(ctx)

	var batchID string
	err = tx.QueryRow(ctx, `
		UPDATE batch_items SET state = $2, last_error_code = $3, prompt_tokens = $4,
			completion_tokens = $5, request_id = $6, result_blob_ref = $7, finished_at = $8
		 WHERE id = $1 AND state = 'inflight'
		RETURNING batch_id`,
		r.ItemID, state, r.ErrorCode, r.PromptTokens, r.CompletionTokens,
		r.RequestID, r.ResultBlobRef, at,
	).Scan(&batchID)
	if errors.Is(err, pgx.ErrNoRows) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("store: finish batch item %q: %w", r.ItemID, err)
	}

	if _, err := tx.Exec(ctx, `
		UPDATE batches SET counts_completed = counts_completed + $2, counts_failed = counts_failed + $3
		 WHERE id = $1`, batchID, completed, failed); err != nil {
		return false, fmt.Errorf("store: move batch %q counts: %w", batchID, err)
	}

	if err := tx.Commit(ctx); err != nil {
		return false, fmt.Errorf("store: commit finish item tx: %w", err)
	}
	return true, nil
}

// ExpireOpenItems moves pending and inflight items to expired.
func (s *PostgresStore) ExpireOpenItems(batchID string, at time.Time) (int, error) {
	return s.closeOpenItems(batchID, ItemExpired, at)
}

// CancelOpenItems moves pending and inflight items to cancelled.
func (s *PostgresStore) CancelOpenItems(batchID string, at time.Time) (int, error) {
	return s.closeOpenItems(batchID, ItemCancelled, at)
}

// closeOpenItems is the shared body of ExpireOpenItems and CancelOpenItems.
// Neither destination state touches counts_completed or counts_failed, so the
// batch counters keep OpenAI's completed + failed ≤ total invariant.
func (s *PostgresStore) closeOpenItems(batchID string, to ItemState, at time.Time) (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	tag, err := s.pool.Exec(ctx, `
		UPDATE batch_items SET state = $2, finished_at = $3
		 WHERE batch_id = $1 AND state IN ('pending', 'inflight')`, batchID, to, at)
	if err != nil {
		return 0, fmt.Errorf("store: close open items for batch %q as %q: %w", batchID, to, err)
	}
	return int(tag.RowsAffected()), nil
}

// ListItems returns a batch's items in line_no order, filtered to states when
// any are given.
func (s *PostgresStore) ListItems(batchID string, states ...ItemState) ([]*BatchItem, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	query := `SELECT ` + batchItemColumns + ` FROM batch_items WHERE batch_id = $1`
	args := []any{batchID}
	if len(states) > 0 {
		wanted := make([]string, 0, len(states))
		for _, st := range states {
			wanted = append(wanted, string(st))
		}
		query += ` AND state = ANY($2)`
		args = append(args, wanted)
	}
	query += ` ORDER BY line_no`

	rows, err := s.pool.Query(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("store: list items for batch %q: %w", batchID, err)
	}
	defer rows.Close()

	out := []*BatchItem{}
	for rows.Next() {
		it, err := scanBatchItem(rows)
		if err != nil {
			return nil, fmt.Errorf("store: scan batch item: %w", err)
		}
		out = append(out, it)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("store: iterate batch items: %w", err)
	}
	return out, nil
}

// CountItems returns the item tallies for one batch in a single round trip.
func (s *PostgresStore) CountItems(batchID string) (total, pending, inflight, succeeded, failed int, err error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	err = s.pool.QueryRow(ctx, `
		SELECT COUNT(*),
		       COUNT(*) FILTER (WHERE state = 'pending'),
		       COUNT(*) FILTER (WHERE state = 'inflight'),
		       COUNT(*) FILTER (WHERE state = 'succeeded'),
		       COUNT(*) FILTER (WHERE state = 'failed')
		  FROM batch_items WHERE batch_id = $1`, batchID,
	).Scan(&total, &pending, &inflight, &succeeded, &failed)
	if err != nil {
		return 0, 0, 0, 0, 0, fmt.Errorf("store: count items for batch %q: %w", batchID, err)
	}
	return total, pending, inflight, succeeded, failed, nil
}
