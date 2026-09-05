package api

import (
	"encoding/json"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// outputLine is the shape of one assembled output or error JSONL line.
type outputLine struct {
	ID       string `json:"id"`
	CustomID string `json:"custom_id"`
	Response *struct {
		StatusCode int             `json:"status_code"`
		RequestID  string          `json:"request_id"`
		Body       json.RawMessage `json:"body"`
	} `json:"response"`
	Error *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// fetchFileLines downloads a result file and decodes its JSONL body.
func (e *batchEnv) fetchFileLines(fileID string) []outputLine {
	e.t.Helper()
	status, raw := e.request(http.MethodGet, "/v1/files/"+fileID+"/content", "", nil, e.key)
	if status != http.StatusOK {
		e.t.Fatalf("GET file content = %d: %s", status, raw)
	}
	lines := make([]outputLine, 0, 4)
	for _, text := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		if strings.TrimSpace(text) == "" {
			continue
		}
		var line outputLine
		if err := json.Unmarshal([]byte(text), &line); err != nil {
			e.t.Fatalf("decode output line %q: %v", text, err)
		}
		lines = append(lines, line)
	}
	return lines
}

// createFileBatch uploads n lines and creates the batch, returning its id.
func (e *batchEnv) createFileBatch(n int) string {
	e.t.Helper()
	status, fileObj := e.uploadMultipart(jsonlLines(n), "batch", "input.jsonl", e.key)
	if status != http.StatusOK {
		e.t.Fatalf("upload = %d %v", status, fileObj)
	}
	fileID, _ := fileObj["id"].(string)
	status, created := e.postJSON("/v1/batches", map[string]any{
		"input_file_id":     fileID,
		"endpoint":          "/v1/chat/completions",
		"completion_window": "24h",
	}, e.key)
	if status != http.StatusOK {
		e.t.Fatalf("create = %d %v", status, created)
	}
	id, _ := created["id"].(string)
	return id
}

func TestFinalizeWritesOutputAndErrorFiles(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(3)

	items := env.claimAll(batchID)
	if len(items) != 3 {
		t.Fatalf("claimed %d items, want 3", len(items))
	}
	env.settleSucceeded(items[0], []byte(`{"id":"chatcmpl-0","object":"chat.completion"}`))
	env.settleSucceeded(items[1], []byte(`{"id":"chatcmpl-1","object":"chat.completion"}`))
	env.settleFailed(items[2])

	// A batch with work still in flight is not finalized.
	if res, err := env.srv.FinalizeBatchIfDone("batch_nonexistent", time.Now().UTC()); err != nil || res != nil {
		t.Fatalf("unknown batch = %+v, %v", res, err)
	}

	now := time.Now().UTC()
	res, err := env.srv.FinalizeBatchIfDone(batchID, now)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if res == nil || res.Status != store.BatchCompleted {
		t.Fatalf("finalize = %+v, want completed", res)
	}
	if res.OutputFileID == nil || res.ErrorFileID == nil {
		t.Fatalf("both files must exist: %+v", res)
	}

	// The batch object now points at both files with settled counts.
	status, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if status != http.StatusOK || got["status"] != "completed" {
		t.Fatalf("get = %d %v", status, got)
	}
	if got["output_file_id"] != *res.OutputFileID || got["error_file_id"] != *res.ErrorFileID {
		t.Fatalf("batch does not carry the assembled file ids: %v", got)
	}
	if total, completed, failed := requestCounts(t, got); total != 3 || completed != 2 || failed != 1 {
		t.Fatalf("request_counts = %d/%d/%d, want 3/2/1", total, completed, failed)
	}
	if got["completed_at"] == nil {
		t.Fatalf("completed_at not stamped: %v", got)
	}

	// Output file: two lines, in line_no order, carrying the plain response.
	output := env.fetchFileLines(*res.OutputFileID)
	if len(output) != 2 {
		t.Fatalf("output has %d lines, want 2", len(output))
	}
	if output[0].CustomID != "req-0" || output[1].CustomID != "req-1" {
		t.Fatalf("output lines out of order: %+v", output)
	}
	for _, line := range output {
		if line.Response == nil || line.Response.StatusCode != http.StatusOK {
			t.Fatalf("output line has no 200 response: %+v", line)
		}
		if line.Error != nil {
			t.Fatalf("output line carries an error: %+v", line)
		}
		if !strings.Contains(string(line.Response.Body), "chat.completion") {
			t.Fatalf("output body not carried through: %s", line.Response.Body)
		}
	}

	// Error file: one line with the fixed request_failed message.
	errorLines := env.fetchFileLines(*res.ErrorFileID)
	if len(errorLines) != 1 {
		t.Fatalf("error file has %d lines, want 1", len(errorLines))
	}
	if errorLines[0].CustomID != "req-2" || errorLines[0].Response != nil {
		t.Fatalf("unexpected error line: %+v", errorLines[0])
	}
	if errorLines[0].Error == nil || errorLines[0].Error.Code != batchRequestFailedCode {
		t.Fatalf("error code = %+v", errorLines[0].Error)
	}
	if errorLines[0].Error.Message != batchItemErrorMessages[batchRequestFailedCode] {
		t.Fatalf("error message is not the fixed string: %q", errorLines[0].Error.Message)
	}

	// The file rows describe themselves as batch results.
	outFile, ok := env.st.GetBatchFile(env.account, *res.OutputFileID)
	if !ok || outFile.Purpose != batchFilePurposeOutput {
		t.Fatalf("output file row = %+v", outFile)
	}
	errFile, _ := env.st.GetBatchFile(env.account, *res.ErrorFileID)
	if errFile.Purpose != batchFilePurposeError {
		t.Fatalf("error file purpose = %q", errFile.Purpose)
	}

	// A second finalize is a no-op: the batch is no longer open.
	if again, err := env.srv.FinalizeBatchIfDone(batchID, now); err != nil || again != nil {
		t.Fatalf("second finalize = %+v, %v", again, err)
	}
}

func TestFinalizeSkipsBatchesWithWorkOutstanding(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(2)

	// Nothing settled yet — both items are pending.
	if res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC()); err != nil || res != nil {
		t.Fatalf("pending batch = %+v, %v", res, err)
	}
	items := env.claimAll(batchID)
	env.settleSucceeded(items[0], []byte(`{"id":"a"}`))

	// One item is still in flight.
	if res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC()); err != nil || res != nil {
		t.Fatalf("in-flight batch = %+v, %v", res, err)
	}
	env.settleFailed(items[1])
	res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC())
	if err != nil || res == nil || res.Status != store.BatchCompleted {
		t.Fatalf("settled batch = %+v, %v", res, err)
	}
}

func TestFinalizeReportsExpiredItemsWithTheirOwnCode(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(2)

	if _, err := env.st.ExpireOpenItems(batchID, time.Now().UTC()); err != nil {
		t.Fatalf("expire: %v", err)
	}
	res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC())
	if err != nil || res == nil {
		t.Fatalf("finalize = %+v, %v", res, err)
	}
	if res.OutputFileID != nil {
		t.Fatalf("no item succeeded, so there is no output file: %+v", res)
	}
	if res.ErrorFileID == nil {
		t.Fatal("expired items belong in the error file")
	}
	lines := env.fetchFileLines(*res.ErrorFileID)
	if len(lines) != 2 {
		t.Fatalf("error file has %d lines, want 2", len(lines))
	}
	for _, line := range lines {
		if line.Error == nil || line.Error.Code != batchItemErrorExpired {
			t.Fatalf("expired line = %+v", line)
		}
		if line.Error.Message != batchItemErrorMessages[batchItemErrorExpired] {
			t.Fatalf("expired message is not the fixed string: %q", line.Error.Message)
		}
	}

	// Counts stay at 0/0 for expiry (OpenAI semantics), and the terminal status
	// finalize picks is expired rather than completed.
	if res.Status != store.BatchExpired {
		t.Fatalf("finalize status = %s, want expired", res.Status)
	}
	_, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if got["status"] != "expired" {
		t.Fatalf("batch status = %v, want expired", got["status"])
	}
	if total, completed, failed := requestCounts(t, got); total != 2 || completed != 0 || failed != 0 {
		t.Fatalf("request_counts = %d/%d/%d, want 2/0/0", total, completed, failed)
	}
}

// The dispatcher's expiry sweep expires the open items and then calls finalize
// with the batch STILL in_progress, so the output and error files are attached
// before it goes terminal. Finalize owns the in_progress → expired transition;
// a crash before it leaves the batch open and the next tick retries the pass.
func TestFinalizeExpiresAPartiallyCompleteBatch(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(5)

	// Two items succeeded; the other three never left pending.
	items := env.claimAll(batchID)
	env.settleSucceeded(items[0], []byte(`{"id":"chatcmpl-0","object":"chat.completion"}`))
	env.settleSucceeded(items[1], []byte(`{"id":"chatcmpl-1","object":"chat.completion"}`))
	for _, it := range items[2:] {
		if ok, err := env.st.ReleaseItem(it.ID); err != nil || !ok {
			t.Fatalf("release %s: ok=%v err=%v", it.ID, ok, err)
		}
	}

	now := time.Now().UTC()
	// This is the sweep's order: close the items, leave the batch open.
	n, err := env.st.ExpireOpenItems(batchID, now)
	if err != nil || n != 3 {
		t.Fatalf("ExpireOpenItems = %d, %v, want 3", n, err)
	}
	if b, ok := env.st.GetBatchByID(batchID); !ok || b.Status != store.BatchInProgress {
		t.Fatalf("the sweep must leave the batch open for finalize: %+v", b)
	}

	res, err := env.srv.FinalizeBatchIfDone(batchID, now)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if res == nil || res.Status != store.BatchExpired {
		t.Fatalf("finalize = %+v, want expired", res)
	}
	if res.OutputFileID == nil || res.ErrorFileID == nil {
		t.Fatalf("an expired batch still carries both files: %+v", res)
	}

	output := env.fetchFileLines(*res.OutputFileID)
	if len(output) != 2 {
		t.Fatalf("output has %d lines, want the 2 that succeeded", len(output))
	}
	for _, line := range output {
		if line.Response == nil || line.Response.StatusCode != http.StatusOK {
			t.Fatalf("output line = %+v", line)
		}
	}
	errLines := env.fetchFileLines(*res.ErrorFileID)
	if len(errLines) != 3 {
		t.Fatalf("error file has %d lines, want the 3 that expired", len(errLines))
	}
	for _, line := range errLines {
		if line.Error == nil || line.Error.Code != batchItemErrorExpired {
			t.Fatalf("expired line = %+v", line)
		}
	}

	_, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if got["status"] != "expired" {
		t.Fatalf("batch status = %v, want expired", got["status"])
	}
	if got["output_file_id"] != *res.OutputFileID || got["error_file_id"] != *res.ErrorFileID {
		t.Fatalf("the expired batch does not carry its files: %v", got)
	}
	// Expiry moves neither counter.
	if total, completed, failed := requestCounts(t, got); total != 5 || completed != 2 || failed != 0 {
		t.Fatalf("request_counts = %d/%d/%d, want 5/2/0", total, completed, failed)
	}
}

// The cancellation drain is the same shape: cancel the open items, leave the
// batch cancelling, and let finalize attach the files and CAS to cancelled. A
// result that lands after the drain is ignored — FinishItem refuses an item
// that is already terminal — so it changes neither the files nor the counts.
func TestFinalizeCancelsWithALateResultIgnored(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(5)

	items := env.claimAll(batchID)
	env.settleSucceeded(items[0], []byte(`{"id":"chatcmpl-0","object":"chat.completion"}`))
	env.settleSucceeded(items[1], []byte(`{"id":"chatcmpl-1","object":"chat.completion"}`))
	// items[2] stays inflight — a dispatch that is still out. The other two are
	// back in pending.
	for _, it := range items[3:] {
		if ok, err := env.st.ReleaseItem(it.ID); err != nil || !ok {
			t.Fatalf("release %s: ok=%v err=%v", it.ID, ok, err)
		}
	}

	now := time.Now().UTC()
	if ok, err := env.st.SetBatchStatus(batchID, store.BatchInProgress, store.BatchCancelling, now); err != nil || !ok {
		t.Fatalf("cancel: ok=%v err=%v", ok, err)
	}
	n, err := env.st.CancelOpenItems(batchID, now)
	if err != nil || n != 3 {
		t.Fatalf("CancelOpenItems = %d, %v, want 3", n, err)
	}

	res, err := env.srv.FinalizeBatchIfDone(batchID, now)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if res == nil || res.Status != store.BatchCancelled {
		t.Fatalf("finalize = %+v, want cancelled", res)
	}
	if res.OutputFileID == nil {
		t.Fatal("the two items that succeeded before the cancel belong in the output file")
	}
	if res.ErrorFileID != nil {
		t.Fatalf("a cancelled item was never attempted, so it belongs in no file: %+v", res)
	}
	if lines := env.fetchFileLines(*res.OutputFileID); len(lines) != 2 {
		t.Fatalf("output has %d lines, want 2", len(lines))
	}

	// The dispatch that was still out reports back. It is ignored.
	ok, err := env.st.FinishItem(store.ItemResult{
		ItemID: items[2].ID, Succeeded: true, RequestID: "req_late",
		ResultBlobRef: batchlane.ResultBlobRef(items[2].ID),
	}, now.Add(time.Second))
	if err != nil {
		t.Fatalf("late finish: %v", err)
	}
	if ok {
		t.Fatal("a late result settled an item the cancel had already closed")
	}

	_, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if got["status"] != "cancelled" {
		t.Fatalf("batch status = %v, want cancelled", got["status"])
	}
	if got["output_file_id"] != *res.OutputFileID {
		t.Fatalf("the cancelled batch does not carry its output file: %v", got)
	}
	if total, completed, failed := requestCounts(t, got); total != 5 || completed != 2 || failed != 0 {
		t.Fatalf("request_counts = %d/%d/%d, want 5/2/0", total, completed, failed)
	}

	// A second finalize changes nothing: the batch is terminal.
	if again, err := env.srv.FinalizeBatchIfDone(batchID, now.Add(2*time.Second)); err != nil || again != nil {
		t.Fatalf("second finalize = %+v, %v", again, err)
	}
}

// TestFinalizeRetriesALostStatusCAS reproduces the wedge where one finalize
// pass's AttachOutputFiles lands but its own status CAS is lost or errored
// (e.g. racing a cancel between the two calls). The next finalize pass must
// not treat AttachOutputFiles losing the race as "someone else already
// finished the job" — it must re-read the batch's current status and files
// and retry the CAS, or the batch would stay wedged open forever with output
// files already attached.
func TestFinalizeRetriesALostStatusCAS(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(1)
	items := env.claimAll(batchID)
	env.settleSucceeded(items[0], []byte(`{"id":"a"}`))

	now := time.Now().UTC()

	// Simulate a prior finalize pass whose attach won but whose status CAS
	// never landed: attach a real output file directly, without moving the
	// batch out of in_progress.
	outputID, err := newBatchID("file-")
	if err != nil {
		t.Fatalf("mint id: %v", err)
	}
	content := []byte(`{"id":"x","custom_id":"req-0","response":{"status_code":200,"body":{"id":"a"}},"error":null}` + "\n")
	if err := env.srv.BatchBlobs().PutPlain(outputID, content); err != nil {
		t.Fatalf("seal output: %v", err)
	}
	if err := env.st.CreateBatchFile(&store.BatchFile{
		ID: outputID, AccountID: env.account, Purpose: batchFilePurposeOutput,
		Filename: batchID + "_batch_output.jsonl", SizeBytes: int64(len(content)), CreatedAt: now,
		BlobRef: outputID, SealedBy: "coordinator",
	}); err != nil {
		t.Fatalf("record output file: %v", err)
	}
	if attached, err := env.st.AttachOutputFiles(batchID, &outputID, nil); err != nil || !attached {
		t.Fatalf("attach = %v, %v", attached, err)
	}

	// Precondition: the batch is wedged — files attached, status still open.
	if b, ok := env.st.GetBatch(env.account, batchID); !ok || b.Status != store.BatchInProgress {
		t.Fatalf("precondition: batch = %+v, ok=%v", b, ok)
	}

	res, err := env.srv.FinalizeBatchIfDone(batchID, now)
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if res == nil || res.Status != store.BatchCompleted {
		t.Fatalf("finalize = %+v, want completed", res)
	}
	if res.OutputFileID == nil || *res.OutputFileID != outputID {
		t.Fatalf("finalize must report the already-attached output file, got %+v", res)
	}

	status, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if status != http.StatusOK || got["status"] != "completed" {
		t.Fatalf("get = %d %v", status, got)
	}
	if got["output_file_id"] != outputID {
		t.Fatalf("batch does not carry the already-attached output file: %v", got)
	}

	// The item input blob is still cleaned up even though this pass never
	// built the output file it reports.
	if names := blobFiles(t, env.blobDir); len(names) != 2 { // the result blob + the output file
		t.Fatalf("unexpected blobs left after finalize: %v", names)
	}
}

func TestPurgeExpiredBatchFiles(t *testing.T) {
	env := newBatchEnv(t)
	batchID := env.createFileBatch(1)
	items := env.claimAll(batchID)
	env.settleSucceeded(items[0], []byte(`{"id":"a"}`))
	res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC())
	if err != nil || res == nil || res.OutputFileID == nil {
		t.Fatalf("finalize = %+v, %v", res, err)
	}
	outputID := *res.OutputFileID

	// Inside the retention window nothing is dropped.
	if purged, err := env.srv.PurgeExpiredBatchFiles(time.Now().UTC()); err != nil || purged != 0 {
		t.Fatalf("early sweep purged %d (%v)", purged, err)
	}
	if lines := env.fetchFileLines(outputID); len(lines) != 1 {
		t.Fatalf("output still readable? got %d lines", len(lines))
	}

	// Past the horizon the content goes and the metadata stays.
	future := time.Now().UTC().Add(batchlane.DefaultOutputRetention + time.Hour)
	purged, err := env.srv.PurgeExpiredBatchFiles(future)
	if err != nil {
		t.Fatalf("sweep: %v", err)
	}
	if purged == 0 {
		t.Fatal("the sweep must drop content past the retention horizon")
	}
	if status, _ := env.request(http.MethodGet, "/v1/files/"+outputID+"/content", "", nil, env.key); status != http.StatusNotFound {
		t.Fatalf("purged content = %d, want 404", status)
	}
	status, meta := env.getJSON("/v1/files/"+outputID, env.key)
	if status != http.StatusOK || meta["id"] != outputID {
		t.Fatalf("purged file metadata = %d %v", status, meta)
	}
	if _, err := env.srv.BatchBlobs().Raw(outputID); err == nil {
		t.Fatal("the sweep must delete the assembled file's blob, not just mark the row")
	}
	// The item's result blob is not a file row and is swept by the batch
	// dispatcher (PR3b), not by this pass; assert it is the only leftover so a
	// future change that widens the sweep has a failing expectation to update.
	if names := blobFiles(t, env.blobDir); len(names) != 1 || names[0] != batchlane.ResultBlobRef(items[0].ID) {
		t.Fatalf("unexpected blobs left after the sweep: %v", names)
	}

	// The sweep is idempotent.
	if again, err := env.srv.PurgeExpiredBatchFiles(future); err != nil || again != 0 {
		t.Fatalf("second sweep purged %d (%v)", again, err)
	}
}
