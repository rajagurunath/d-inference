package api

// /v1/batches — create, list, retrieve, cancel (docs/design/tidal-batch-lane.md §3.6).
//
// Two consumer shapes reach one implementation. The OpenAI form points at an
// uploaded input_file_id; the OpenRouter form carries a model plus an inline
// requests array and reads its results back off the batch object. The stored
// batch always records which it was (source) and who the results are sealed to
// (sealed_to), so retrieval can serve both without a second code path.
//
// Ownership: every handler loads the batch with the authenticated account id
// and 404s on a miss before it touches an unscoped mutator.

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// Admission policy, verbatim from the plan's Global Constraints.
const (
	// batchCompletionWindowDuration is what completion_window: "24h" means.
	batchCompletionWindowDuration = 24 * time.Hour
	// batchAdmissionMargin is the fraction of the window a new batch may fill
	// at the currently observed completion rate. Anything slower is refused at
	// submission rather than expiring 24 hours later.
	batchAdmissionMargin = 0.8
	// batchRateWindow is the trailing window the observed completion rate is
	// measured over.
	batchRateWindow = time.Hour

	// Sealing destinations recorded on the batch.
	sealedToCoordinator = "coordinator"
	sealedToConsumer    = "consumer"

	// Batch sources.
	batchSourceFile   = "file"
	batchSourceInline = "inline"

	defaultBatchListLimit = 20
	maxBatchListLimit     = 100
)

// createBatchRequest is the union of the OpenAI and OpenRouter create bodies.
// Exactly one of InputFileID and Requests may be present.
type createBatchRequest struct {
	InputFileID      string            `json:"input_file_id,omitempty"` // OpenAI form
	Endpoint         string            `json:"endpoint"`
	CompletionWindow string            `json:"completion_window"` // must be "24h"
	Metadata         map[string]string `json:"metadata,omitempty"`
	// ResultPublicKey is a base64 32-byte X25519 key. When present, results are
	// sealed to it and the coordinator can no longer read them.
	ResultPublicKey string          `json:"result_public_key,omitempty"`
	Model           string          `json:"model,omitempty"`    // OpenRouter inline form
	Requests        []inlineRequest `json:"requests,omitempty"` // OpenRouter inline form
}

// inlineResult is one element of the results array an inline batch returns
// once it reaches a terminal state (OpenRouter shape).
type inlineResult struct {
	CustomID string         `json:"custom_id"`
	Response *itemResponse  `json:"response"`
	Error    *itemErrorBody `json:"error"`
	ID       string         `json:"id,omitempty"`
}

// itemResponse is the response half of an output line or an inline result.
// Body is the plain OpenAI response JSON when the batch is sealed to the
// coordinator, and an e2e.EncryptedPayload object when it is sealed to the
// consumer.
type itemResponse struct {
	StatusCode int             `json:"status_code"`
	RequestID  string          `json:"request_id,omitempty"`
	Body       json.RawMessage `json:"body"`
}

// itemErrorBody is the error half. Message is a fixed string per code and never
// carries provider text.
type itemErrorBody struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// handleBatchCreate handles POST /v1/batches.
func (s *Server) handleBatchCreate(w http.ResponseWriter, r *http.Request) {
	blobs, err := s.batchStore()
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	accountID := s.resolveAccountID(r)

	var req createBatchRequest
	if !decodeCappedJSON(w, r, maxFileBytes*4/3+multipartSlackBytes, &req) {
		return
	}

	items, source, inputFileID, err := s.parseCreateBatch(accountID, blobs, &req)
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	resultKey, sealedTo, err := parseResultPublicKey(req.ResultPublicKey)
	if err != nil {
		s.writeBatchError(w, err)
		return
	}
	if err := validateBatchMetadata(req.Metadata); err != nil {
		s.writeBatchError(w, err)
		return
	}

	now := time.Now().UTC()
	if err := s.checkBatchFeasible(len(items), now); err != nil {
		s.writeBatchError(w, err)
		return
	}

	batchID, err := newBatchID("batch_")
	if err != nil {
		s.writeBatchError(w, internalBatchError(err))
		return
	}
	endpoint := req.Endpoint
	if endpoint == "" && len(items) > 0 {
		endpoint = items[0].Line.URL
	}

	records := make([]*store.BatchItem, 0, len(items))
	written := make([]string, 0, len(items))
	for _, it := range items {
		itemID, err := newBatchID("bitem_")
		if err != nil {
			s.rollbackItemBlobs(blobs, written)
			s.writeBatchError(w, internalBatchError(err))
			return
		}
		if err := blobs.PutPlain(BatchItemInputRef(itemID), it.Raw); err != nil {
			s.logger.Error("batch: sealing an item body failed", "batch_id", batchID, "item_id", itemID, "error", err)
			s.rollbackItemBlobs(blobs, written)
			s.writeBatchError(w, internalBatchError(err))
			return
		}
		written = append(written, BatchItemInputRef(itemID))
		records = append(records, &store.BatchItem{
			ID:       itemID,
			BatchID:  batchID,
			CustomID: it.Line.CustomID,
			LineNo:   it.LineNo,
			State:    store.ItemPending,
			BlobRef:  BatchItemInputRef(itemID),
		})
	}

	batch := &store.Batch{
		ID:               batchID,
		AccountID:        accountID,
		InputFileID:      inputFileID,
		Endpoint:         endpoint,
		Status:           store.BatchValidating,
		CompletionWindow: batchCompletionWindow,
		CreatedAt:        now,
		ExpiresAt:        now.Add(batchCompletionWindowDuration),
		CountsTotal:      len(records),
		ResultPublicKey:  resultKey,
		SealedTo:         sealedTo,
		Source:           source,
		Model:            req.Model,
		Metadata:         req.Metadata,
	}
	if err := s.store.CreateBatch(batch, records); err != nil {
		s.rollbackItemBlobs(blobs, written)
		s.logger.Error("batch: create failed", "batch_id", batchID, "error", err)
		s.writeBatchError(w, internalBatchError(err))
		return
	}

	// The response reports the batch as the consumer created it (validating);
	// the transition below is what the next GET observes.
	response := batchObject(batch, nil)

	if ok, err := s.store.SetBatchStatus(batchID, store.BatchValidating, store.BatchInProgress, now); err != nil {
		s.logger.Error("batch: admitting to in_progress failed", "batch_id", batchID, "error", err)
	} else if !ok {
		s.logger.Warn("batch: admission CAS lost a race", "batch_id", batchID)
	}

	// The input file has been fanned out into per-item blobs; its own blob is
	// now a second copy of the same prompts and is dropped immediately.
	if inputFileID != "" {
		s.purgeBatchFileBlob(blobs, inputFileID)
	}

	s.logger.Info("batch: created",
		"batch_id", batchID, "account_id", accountID, "source", source,
		"sealed_to", sealedTo, "endpoint", endpoint, "requests", len(records))

	status := http.StatusOK
	if source == batchSourceInline {
		status = http.StatusAccepted
	}
	writeJSON(w, status, response)
}

// parseCreateBatch validates the create body's shape and returns the parsed
// items. It enforces "exactly one of input_file_id / requests" and, for the
// file form, that the file belongs to the caller and still has its content.
func (s *Server) parseCreateBatch(accountID string, blobs *sealedblob.Store, req *createBatchRequest) ([]parsedItem, string, string, error) {
	hasFile := strings.TrimSpace(req.InputFileID) != ""
	hasInline := len(req.Requests) > 0
	switch {
	case hasFile && hasInline:
		return nil, "", "", batchErr("invalid_request", "requests",
			"supply either input_file_id or requests, not both")
	case !hasFile && !hasInline:
		return nil, "", "", batchErr("invalid_request", "input_file_id",
			"one of input_file_id or requests is required")
	}

	window := strings.TrimSpace(req.CompletionWindow)
	if window == "" {
		window = batchCompletionWindow
	}
	if window != batchCompletionWindow {
		return nil, "", "", batchErr("invalid_completion_window", "completion_window",
			"completion_window must be %q", batchCompletionWindow)
	}
	req.CompletionWindow = window

	endpoint := strings.TrimSpace(req.Endpoint)
	if endpoint == "" {
		endpoint = "/v1/chat/completions"
	}
	if !batchEndpoints[endpoint] {
		return nil, "", "", batchErr("invalid_endpoint", "endpoint",
			"endpoint %q is not available for batch — use /v1/chat/completions or /v1/completions", endpoint)
	}
	req.Endpoint = endpoint

	if hasInline {
		items, err := parseInlineRequests(req.Requests, endpoint, req.Model, maxInlineRequests, s.batchModelResolver())
		return items, batchSourceInline, "", err
	}

	fileID := strings.TrimSpace(req.InputFileID)
	f, ok := s.store.GetBatchFile(accountID, fileID)
	if !ok {
		return nil, "", "", batchNotFound("file")
	}
	if f.Purpose != batchFilePurposeInput {
		return nil, "", "", batchErr("invalid_request", "input_file_id",
			"input_file_id must name a file uploaded with purpose %q", batchFilePurposeInput)
	}
	if f.PurgedAt != nil {
		return nil, "", "", batchErr("invalid_request", "input_file_id",
			"this input file's content has been purged and can no longer be batched")
	}
	content, err := blobs.Open(f.BlobRef)
	if err != nil {
		s.logger.Error("batch: opening an input file blob failed", "file_id", f.ID, "error", err)
		return nil, "", "", batchErr("invalid_request", "input_file_id",
			"this input file's content is no longer available")
	}
	items, err := parseBatchJSONL(strings.NewReader(string(content)), endpoint, maxFileLines, s.batchModelResolver())
	return items, batchSourceFile, fileID, err
}

// parseResultPublicKey validates the optional consumer sealing key. It is
// strict: a key that is not exactly 32 bytes of base64 is a 400, never a
// silent fallback to coordinator sealing, because the consumer would otherwise
// believe their results were private when they are not.
func parseResultPublicKey(raw string) (string, string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", sealedToCoordinator, nil
	}
	key, err := base64.StdEncoding.DecodeString(raw)
	if err != nil || len(key) != 32 {
		return "", "", batchErr("invalid_result_public_key", "result_public_key",
			"result_public_key must be a base64-encoded 32-byte X25519 public key")
	}
	return raw, sealedToConsumer, nil
}

// checkBatchFeasible refuses a batch the fleet demonstrably cannot finish
// inside the completion window. The rate is only advisory — when nothing has
// finished in the trailing window there is no rate to reason about and the
// batch is admitted.
func (s *Server) checkBatchFeasible(items int, now time.Time) error {
	rate, known := s.store.CompletionRate(batchRateWindow, now)
	if !known || rate <= 0 {
		return nil
	}
	needed := float64(items) / rate
	budget := batchCompletionWindowDuration.Seconds() * batchAdmissionMargin
	if needed <= budget {
		return nil
	}
	return batchErr("batch_infeasible", "requests",
		"this batch needs about %.0f seconds at the current fleet completion rate, more than the %.0f seconds available in the completion window",
		needed, budget)
}

// rollbackItemBlobs removes the item blobs written before a later step failed,
// so a rejected create leaves nothing sealed on disk.
func (s *Server) rollbackItemBlobs(blobs *sealedblob.Store, refs []string) {
	for _, ref := range refs {
		if err := blobs.Delete(ref); err != nil {
			s.logger.Error("batch: rolling back an item blob failed", "item_id", ref, "error", err)
		}
	}
}

// purgeBatchFileBlob deletes a file's sealed bytes and marks the row purged, so
// the metadata still answers GET /v1/files/{id} while the content is gone.
func (s *Server) purgeBatchFileBlob(blobs *sealedblob.Store, fileID string) {
	if err := blobs.Delete(fileID); err != nil {
		s.logger.Error("batch: deleting a file blob failed", "file_id", fileID, "error", err)
		return
	}
	if err := s.store.MarkBatchFilePurged(fileID, time.Now().UTC()); err != nil {
		s.logger.Error("batch: marking a file purged failed", "file_id", fileID, "error", err)
	}
}

// handleBatchList handles GET /v1/batches.
func (s *Server) handleBatchList(w http.ResponseWriter, r *http.Request) {
	if _, err := s.batchStore(); err != nil {
		s.writeBatchError(w, err)
		return
	}
	accountID := s.resolveAccountID(r)

	limit := defaultBatchListLimit
	if raw := strings.TrimSpace(r.URL.Query().Get("limit")); raw != "" {
		n, err := strconv.Atoi(raw)
		if err != nil || n < 1 {
			s.writeBatchError(w, batchErr("invalid_request", "limit", "limit must be a positive integer"))
			return
		}
		limit = min(n, maxBatchListLimit)
	}

	// Ask for one extra row so has_more is exact without a second query.
	batches, err := s.store.ListBatches(accountID, limit+1, strings.TrimSpace(r.URL.Query().Get("after")))
	if err != nil {
		s.logger.Error("batch: list failed", "account_id", accountID, "error", err)
		s.writeBatchError(w, internalBatchError(err))
		return
	}
	hasMore := len(batches) > limit
	if hasMore {
		batches = batches[:limit]
	}

	data := make([]map[string]any, 0, len(batches))
	for _, b := range batches {
		data = append(data, batchObject(b, nil))
	}
	body := map[string]any{"object": "list", "data": data, "has_more": hasMore}
	if len(batches) > 0 {
		body["first_id"] = batches[0].ID
		body["last_id"] = batches[len(batches)-1].ID
	}
	writeJSON(w, http.StatusOK, body)
}

// handleBatchGet handles GET /v1/batches/{id}.
func (s *Server) handleBatchGet(w http.ResponseWriter, r *http.Request) {
	if _, err := s.batchStore(); err != nil {
		s.writeBatchError(w, err)
		return
	}
	b, ok := s.store.GetBatch(s.resolveAccountID(r), strings.TrimSpace(r.PathValue("id")))
	if !ok {
		s.writeBatchError(w, batchNotFound("batch"))
		return
	}

	var results []inlineResult
	if b.Source == batchSourceInline && batchIsTerminal(b.Status) {
		var err error
		if results, err = s.inlineBatchResults(b); err != nil {
			s.logger.Error("batch: assembling inline results failed", "batch_id", b.ID, "error", err)
			s.writeBatchError(w, internalBatchError(err))
			return
		}
	}
	writeJSON(w, http.StatusOK, batchObject(b, results))
}

// handleBatchCancel handles POST /v1/batches/{id}/cancel. It moves the batch to
// cancelling and cancels every open item; the batch reaches cancelled when the
// dispatcher's next finalize observes that nothing is left in flight.
func (s *Server) handleBatchCancel(w http.ResponseWriter, r *http.Request) {
	if _, err := s.batchStore(); err != nil {
		s.writeBatchError(w, err)
		return
	}
	accountID := s.resolveAccountID(r)
	batchID := strings.TrimSpace(r.PathValue("id"))
	b, ok := s.store.GetBatch(accountID, batchID)
	if !ok {
		s.writeBatchError(w, batchNotFound("batch"))
		return
	}

	switch b.Status {
	case store.BatchCancelling, store.BatchCancelled:
		// Idempotent: report the batch as it already is.
	case store.BatchValidating, store.BatchInProgress:
		now := time.Now().UTC()
		if ok, err := s.store.SetBatchStatus(batchID, b.Status, store.BatchCancelling, now); err != nil {
			s.logger.Error("batch: cancel CAS failed", "batch_id", batchID, "error", err)
			s.writeBatchError(w, internalBatchError(err))
			return
		} else if !ok {
			// Someone else moved it first; re-read and report that.
			break
		}
		if _, err := s.store.CancelOpenItems(batchID, now); err != nil {
			s.logger.Error("batch: cancelling open items failed", "batch_id", batchID, "error", err)
		}
		s.logger.Info("batch: cancellation requested", "batch_id", batchID, "account_id", accountID)
	default:
		s.writeBatchError(w, &batchError{
			Status: http.StatusConflict, Type: "invalid_request_error", Code: "batch_not_cancellable",
			Message: fmt.Sprintf("a batch in status %q can no longer be cancelled", b.Status),
		})
		return
	}

	current, ok := s.store.GetBatch(accountID, batchID)
	if !ok {
		s.writeBatchError(w, batchNotFound("batch"))
		return
	}
	writeJSON(w, http.StatusOK, batchObject(current, nil))
}

// batchIsTerminal reports whether the batch will not change again.
func batchIsTerminal(status store.BatchStatus) bool {
	switch status {
	case store.BatchCompleted, store.BatchFailed, store.BatchExpired, store.BatchCancelled:
		return true
	}
	return false
}

// batchObject renders the OpenAI batch object plus the two Darkbloom fields
// (source, sealed_to) and, for an inline batch, the model and its results.
func batchObject(b *store.Batch, results []inlineResult) map[string]any {
	obj := map[string]any{
		"object":            "batch",
		"id":                b.ID,
		"endpoint":          b.Endpoint,
		"input_file_id":     b.InputFileID,
		"completion_window": b.CompletionWindow,
		"status":            string(b.Status),
		"created_at":        b.CreatedAt.Unix(),
		"expires_at":        b.ExpiresAt.Unix(),
		"request_counts": map[string]any{
			"total":     b.CountsTotal,
			"completed": b.CountsCompleted,
			"failed":    b.CountsFailed,
		},
		"metadata":  b.Metadata,
		"source":    b.Source,
		"sealed_to": b.SealedTo,
	}
	obj["output_file_id"] = stringOrNil(b.OutputFileID)
	obj["error_file_id"] = stringOrNil(b.ErrorFileID)
	obj["in_progress_at"] = unixOrNil(b.InProgressAt)
	obj["completed_at"] = unixOrNil(b.CompletedAt)
	obj["cancelled_at"] = unixOrNil(b.CancelledAt)
	if b.Source == batchSourceInline {
		obj["model"] = b.Model
		if results != nil {
			obj["results"] = results
		}
	}
	return obj
}

func stringOrNil(v *string) any {
	if v == nil {
		return nil
	}
	return *v
}

func unixOrNil(t *time.Time) any {
	if t == nil {
		return nil
	}
	return t.Unix()
}
