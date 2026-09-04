package store

// Batch-lane persistence: the OpenAI-wire Batch API's files, batches, and
// per-request items (docs/design/tidal-batch-lane.md §3.2).
//
// These tables are metadata only. Request bodies and results live in the sealed
// blob store (store/sealedblob) keyed by item or file id, so no column here —
// and no log line or error string produced by either backend — carries prompt
// or response content, a hash of it, a filename, a custom_id, or a metadata
// value outside the one column designed to hold it.

import (
	"errors"
	"sort"
	"time"
)

// ErrDuplicateCustomID is returned by CreateBatch when two items in the same
// batch share a custom_id. Both backends check this in Go before any insert,
// so the duplicate never reaches the UNIQUE (batch_id, custom_id) index and
// its value never reaches a Postgres error log.
var ErrDuplicateCustomID = errors.New("store: duplicate custom_id in batch")

// checkDuplicateCustomIDs returns ErrDuplicateCustomID if two items share a
// custom_id. It is the Go-side backstop for the UNIQUE (batch_id, custom_id)
// index, checked before either backend writes anything.
func checkDuplicateCustomIDs(items []*BatchItem) error {
	seen := make(map[string]struct{}, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		if _, dup := seen[it.CustomID]; dup {
			return ErrDuplicateCustomID
		}
		seen[it.CustomID] = struct{}{}
	}
	return nil
}

// sortBatchItemsByLineNo puts items in dispatch order. The memory backend keeps
// each batch's slice sorted on write; the Postgres backend re-sorts a claim's
// RETURNING rows, which carry no ordering guarantee of their own.
func sortBatchItemsByLineNo(items []*BatchItem) {
	sort.Slice(items, func(i, j int) bool { return items[i].LineNo < items[j].LineNo })
}

// BatchStatus is the lifecycle of a batch:
// validating → in_progress → completed | expired, or → cancelling → cancelled,
// or validating → failed.
type BatchStatus string

const (
	BatchValidating BatchStatus = "validating"
	BatchInProgress BatchStatus = "in_progress"
	BatchCompleted  BatchStatus = "completed"
	BatchFailed     BatchStatus = "failed"
	BatchExpired    BatchStatus = "expired"
	BatchCancelling BatchStatus = "cancelling"
	BatchCancelled  BatchStatus = "cancelled"
)

// ItemState is the lifecycle of one request inside a batch:
// pending → inflight → succeeded | failed, or → expired | cancelled.
// counts_completed counts only succeeded and counts_failed only failed, so
// expired and cancelled items make completed + failed ≤ total (OpenAI
// semantics).
type ItemState string

const (
	ItemPending   ItemState = "pending"
	ItemInflight  ItemState = "inflight"
	ItemSucceeded ItemState = "succeeded"
	ItemFailed    ItemState = "failed"
	ItemExpired   ItemState = "expired"
	ItemCancelled ItemState = "cancelled"
)

// BatchFile is an uploaded input file or an assembled output/error file. The
// bytes themselves are sealed on disk under BlobRef; only the shape is here.
type BatchFile struct {
	ID        string     `json:"id"` // "file-" + 24 hex
	AccountID string     `json:"account_id"`
	Purpose   string     `json:"purpose"` // "batch" | "batch_output" | "batch_error"
	Filename  string     `json:"filename"`
	SizeBytes int64      `json:"size_bytes"`
	CreatedAt time.Time  `json:"created_at"`
	BlobRef   string     `json:"blob_ref"`  // sealedblob key (= file id)
	SealedBy  string     `json:"sealed_by"` // always "coordinator" in v1
	PurgedAt  *time.Time `json:"purged_at,omitempty"`
}

// Batch is one submitted job. Counts move only through FinishItem so a
// duplicate or late result can never advance them twice.
type Batch struct {
	ID               string            `json:"id"` // "batch_" + 24 hex
	AccountID        string            `json:"account_id"`
	InputFileID      string            `json:"input_file_id"`
	Endpoint         string            `json:"endpoint"`
	Status           BatchStatus       `json:"status"`
	CompletionWindow string            `json:"completion_window"` // "24h"
	CreatedAt        time.Time         `json:"created_at"`
	ExpiresAt        time.Time         `json:"expires_at"`
	InProgressAt     *time.Time        `json:"in_progress_at,omitempty"`
	CompletedAt      *time.Time        `json:"completed_at,omitempty"`
	CancelledAt      *time.Time        `json:"cancelled_at,omitempty"`
	CountsTotal      int               `json:"counts_total"`
	CountsCompleted  int               `json:"counts_completed"`
	CountsFailed     int               `json:"counts_failed"`
	OutputFileID     *string           `json:"output_file_id,omitempty"`
	ErrorFileID      *string           `json:"error_file_id,omitempty"`
	ResultPublicKey  string            `json:"result_public_key,omitempty"` // base64 X25519, validated as 32 bytes at creation
	SealedTo         string            `json:"sealed_to"`                   // "coordinator" | "consumer"
	Source           string            `json:"source"`                      // "file" | "inline"
	Model            string            `json:"model,omitempty"`             // inline form only
	Metadata         map[string]string `json:"metadata,omitempty"`
}

// BatchItem is one request line. BlobRef and ResultBlobRef are sealedblob keys
// derived from the item id, never from CustomID.
type BatchItem struct {
	ID               string     `json:"id"` // "bitem_" + 24 hex
	BatchID          string     `json:"batch_id"`
	CustomID         string     `json:"custom_id"`
	LineNo           int        `json:"line_no"`
	State            ItemState  `json:"state"`
	Attempts         int        `json:"attempts"`
	LastErrorCode    string     `json:"last_error_code,omitempty"`
	PromptTokens     int        `json:"prompt_tokens"`
	CompletionTokens int        `json:"completion_tokens"`
	SubmittedAt      *time.Time `json:"submitted_at,omitempty"`
	FinishedAt       *time.Time `json:"finished_at,omitempty"`
	RequestID        string     `json:"request_id,omitempty"`
	BlobRef          string     `json:"blob_ref"`                  // sealed request body
	ResultBlobRef    string     `json:"result_blob_ref,omitempty"` // sealed result
}

// ItemResult is the terminal outcome the dispatcher settles an item with.
// ErrorCode is one of a fixed set ("request_failed", "batch_expired") — never
// provider text.
type ItemResult struct {
	ItemID           string
	Succeeded        bool
	ErrorCode        string
	PromptTokens     int
	CompletionTokens int
	RequestID        string
	ResultBlobRef    string
}

// BatchFileStore covers uploaded input files and assembled output/error files.
type BatchFileStore interface {
	// CreateBatchFile records a file whose sealed bytes are already on disk.
	CreateBatchFile(f *BatchFile) error

	// GetBatchFile returns the file if it belongs to accountID. Handlers always
	// pass the authenticated account so one consumer can never read another's.
	GetBatchFile(accountID, id string) (*BatchFile, bool)

	// MarkBatchFilePurged stamps purged_at after the blob has been deleted, so a
	// retention sweep is idempotent and a purged file reports as gone rather
	// than as a read error.
	MarkBatchFilePurged(id string, at time.Time) error

	// ListPurgeableFiles returns files created before the cutoff that have not
	// been purged yet. The caller decides the cutoff (outputRetention after
	// completion) and deletes each blob before marking it.
	ListPurgeableFiles(before time.Time) ([]*BatchFile, error)
}

// BatchStore covers the batch object and its status machine.
type BatchStore interface {
	// CreateBatch atomically writes the batch and all of its items, and sets the
	// batch status to validating.
	CreateBatch(b *Batch, items []*BatchItem) error

	// GetBatch returns the batch if it belongs to accountID.
	GetBatch(accountID, id string) (*Batch, bool)

	// ListBatches returns an account's batches newest first, at most limit rows,
	// starting strictly after the batch id in after (empty for the first page).
	ListBatches(accountID string, limit int, after string) ([]*Batch, error)

	// SetBatchStatus is a compare-and-set on status: it returns false and
	// changes nothing when the current status is not from. It stamps
	// in_progress_at, completed_at, or cancelled_at to match the destination.
	SetBatchStatus(id string, from, to BatchStatus, at time.Time) (bool, error)

	// AttachOutputFiles records the assembled result files. First writer wins:
	// once either file id is set, a later call returns false and changes
	// nothing, so a retried finalize cannot orphan the first pair of files.
	AttachOutputFiles(id string, outputFileID, errorFileID *string) (bool, error)

	// ListOpenBatches returns every in_progress or cancelling batch, for the
	// dispatcher's tick and for restart recovery.
	ListOpenBatches() ([]*Batch, error)

	// CompletionRate returns the fleet-wide batch item completion rate over the
	// trailing window, for laxity. known is false when no item finished in the
	// window, so the caller can fall back to floorItemsPerSec rather than
	// treating an empty window as a rate of zero.
	CompletionRate(window time.Duration, now time.Time) (itemsPerSec float64, known bool)
}

// BatchItemStore covers per-item claim and settle. Every method here is
// dispatcher-only — no request handler calls one.
type BatchItemStore interface {
	// ClaimPendingItems atomically moves up to limit pending items to inflight
	// in line_no order, counting an attempt on each and stamping submitted_at.
	// It claims nothing unless the batch is in_progress, so a cancelling or
	// expired batch drains instead of dispatching.
	ClaimPendingItems(batchID string, limit int, at time.Time) ([]*BatchItem, error)

	// ReleaseItem returns one inflight item to pending and un-counts the attempt
	// its claim charged, for a claim that found no capacity and never
	// dispatched. It returns false when the item is not inflight.
	ReleaseItem(itemID string) (bool, error)

	// RequeueInflightItems returns every inflight item of a batch to pending
	// after a coordinator restart. Unlike ReleaseItem it keeps the attempt
	// count: the dispatch really did happen, its outcome is just unknown.
	RequeueInflightItems(batchID string) (int, error)

	// FinishItem settles one inflight item and moves counts_completed or
	// counts_failed in the same transaction. It is idempotent: a duplicate or
	// late result for an item that is already terminal (including expired or
	// cancelled) returns false and moves nothing.
	FinishItem(r ItemResult, at time.Time) (bool, error)

	// ExpireOpenItems moves every pending or inflight item of a batch to
	// expired, leaving terminal items alone. Expired items count in neither
	// counts_completed nor counts_failed.
	ExpireOpenItems(batchID string, at time.Time) (int, error)

	// CancelOpenItems moves every pending or inflight item of a batch to
	// cancelled, leaving terminal items alone.
	CancelOpenItems(batchID string, at time.Time) (int, error)

	// ListItems returns a batch's items in line_no order, filtered to the given
	// states when any are supplied.
	ListItems(batchID string, states ...ItemState) ([]*BatchItem, error)

	// CountItems returns the item tallies for one batch. total counts every
	// item; the rest count only their own state.
	CountItems(batchID string) (total, pending, inflight, succeeded, failed int, err error)

	// BatchItemExists reports whether an item row exists, in any state. It is
	// the cheap existence probe the dispatcher's orphan sweep needs: a crash
	// between sealing an item body and committing its rows leaves a blob no row
	// references, and only a per-ref lookup can tell that apart from a live
	// item. Deliberately unscoped and index-only — it returns no content, not
	// even the batch the item belongs to.
	BatchItemExists(itemID string) (bool, error)
}
