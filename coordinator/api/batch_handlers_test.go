package api

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/internal/e2e"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
)

// batchTestModel is the one catalog entry the batch test fleet knows, so model
// validation exercises the real resolver instead of a permissive stub.
const batchTestModel = "test-known-model"

// batchTestPrompt is the string every privacy assertion looks for: it must
// never appear in a log line or in a blob on disk.
const batchTestPrompt = "summarize-this-secret-document"

type batchEnv struct {
	t       *testing.T
	srv     *Server
	st      *store.MemoryStore
	ts      *httptest.Server
	blobDir string
	logs    *syncBuffer
	key     string
	account string
}

// newBatchEnv builds a coordinator with the batch lane enabled on a temp
// directory and a process-local key, plus one API key bound to a real account.
// Log output is captured so privacy assertions can read every line the request
// path emitted.
func newBatchEnv(t *testing.T) *batchEnv {
	t.Helper()
	logs := &syncBuffer{}
	logger := slog.New(slog.NewTextHandler(logs, &slog.HandlerOptions{Level: slog.LevelDebug}))

	st := store.NewMemory(store.Config{})
	reg := registry.New(logger)
	reg.SetModelCatalog([]registry.CatalogEntry{{ID: batchTestModel}})

	dir := t.TempDir()
	cfg := ServerConfig{Batch: BatchConfig{BlobDir: dir, DevInsecureKey: true}}
	srv := NewServer(reg, st, cfg, logger)
	blobs, err := NewBatchBlobStore(cfg.Batch, "", logger)
	if err != nil {
		t.Fatalf("batch blob store: %v", err)
	}
	srv.SetBatchBlobStore(blobs)

	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)

	env := &batchEnv{t: t, srv: srv, st: st, ts: ts, blobDir: dir, logs: logs}
	env.key, env.account = env.newKey("acct-primary")
	return env
}

func (e *batchEnv) newKey(accountID string) (string, string) {
	e.t.Helper()
	raw, _, err := e.st.CreateAPIKey(accountID, store.APIKeyCreate{})
	if err != nil {
		e.t.Fatalf("create api key: %v", err)
	}
	return raw, accountID
}

func (e *batchEnv) request(method, path, contentType string, body io.Reader, key string) (int, []byte) {
	e.t.Helper()
	req, err := http.NewRequest(method, e.ts.URL+path, body)
	if err != nil {
		e.t.Fatalf("build request: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+key)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		e.t.Fatalf("%s %s: %v", method, path, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		e.t.Fatalf("read body: %v", err)
	}
	return resp.StatusCode, raw
}

func (e *batchEnv) postJSON(path string, body any, key string) (int, map[string]any) {
	e.t.Helper()
	encoded, err := json.Marshal(body)
	if err != nil {
		e.t.Fatalf("encode body: %v", err)
	}
	status, raw := e.request(http.MethodPost, path, "application/json", bytes.NewReader(encoded), key)
	return status, decodeObject(e.t, raw)
}

func (e *batchEnv) getJSON(path, key string) (int, map[string]any) {
	e.t.Helper()
	status, raw := e.request(http.MethodGet, path, "", nil, key)
	return status, decodeObject(e.t, raw)
}

func decodeObject(t *testing.T, raw []byte) map[string]any {
	t.Helper()
	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		t.Fatalf("decode response %q: %v", string(raw), err)
	}
	return obj
}

// uploadMultipart posts content as an OpenAI-shaped multipart upload.
func (e *batchEnv) uploadMultipart(content, purpose, filename, key string) (int, map[string]any) {
	e.t.Helper()
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("purpose", purpose); err != nil {
		e.t.Fatalf("write purpose: %v", err)
	}
	part, err := mw.CreateFormFile("file", filename)
	if err != nil {
		e.t.Fatalf("create file part: %v", err)
	}
	if _, err := part.Write([]byte(content)); err != nil {
		e.t.Fatalf("write file part: %v", err)
	}
	if err := mw.Close(); err != nil {
		e.t.Fatalf("close multipart: %v", err)
	}
	status, raw := e.request(http.MethodPost, "/v1/files", mw.FormDataContentType(), &body, key)
	return status, decodeObject(e.t, raw)
}

// jsonlLines builds a valid batch input file with n lines.
func jsonlLines(n int) string {
	var b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&b, `{"custom_id":"req-%d","method":"POST","url":"/v1/chat/completions","body":{"model":%q,"messages":[{"role":"user","content":%q}]}}`+"\n",
			i, batchTestModel, batchTestPrompt)
	}
	return b.String()
}

func errorCode(t *testing.T, body map[string]any) string {
	t.Helper()
	detail, ok := body["error"].(map[string]any)
	if !ok {
		t.Fatalf("response has no error envelope: %v", body)
	}
	code, _ := detail["code"].(string)
	return code
}

func requestCounts(t *testing.T, batch map[string]any) (total, completed, failed int) {
	t.Helper()
	counts, ok := batch["request_counts"].(map[string]any)
	if !ok {
		t.Fatalf("batch has no request_counts: %v", batch)
	}
	toInt := func(k string) int {
		v, _ := counts[k].(float64)
		return int(v)
	}
	return toInt("total"), toInt("completed"), toInt("failed")
}

// blobFiles lists the sealed blob directory, skipping the temp files a
// concurrent write might leave.
func blobFiles(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read blob dir: %v", err)
	}
	names := make([]string, 0, len(entries))
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), ".tmp-") {
			continue
		}
		names = append(names, entry.Name())
	}
	return names
}

// assertNoPlaintextOnDisk fails when any blob file contains needle.
func assertNoPlaintextOnDisk(t *testing.T, dir, needle string) {
	t.Helper()
	for _, name := range blobFiles(t, dir) {
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read blob %s: %v", name, err)
		}
		if bytes.Contains(raw, []byte(needle)) {
			t.Fatalf("blob %s holds %q in the clear", name, needle)
		}
	}
}

func TestBatchLifecycleOverHTTP(t *testing.T) {
	env := newBatchEnv(t)

	// 1. Upload three valid lines.
	status, fileObj := env.uploadMultipart(jsonlLines(3), "batch", "input.jsonl", env.key)
	if status != http.StatusOK {
		t.Fatalf("upload status = %d, body %v", status, fileObj)
	}
	fileID, _ := fileObj["id"].(string)
	if !strings.HasPrefix(fileID, "file-") {
		t.Fatalf("file id = %q, want a file- prefix", fileID)
	}
	if fileObj["object"] != "file" || fileObj["purpose"] != "batch" {
		t.Fatalf("unexpected file object: %v", fileObj)
	}
	if got, _ := fileObj["bytes"].(float64); int(got) != len(jsonlLines(3)) {
		t.Fatalf("bytes = %v, want %d", fileObj["bytes"], len(jsonlLines(3)))
	}
	assertNoPlaintextOnDisk(t, env.blobDir, batchTestPrompt)

	// 2. Create the batch.
	status, created := env.postJSON("/v1/batches", map[string]any{
		"input_file_id":     fileID,
		"endpoint":          "/v1/chat/completions",
		"completion_window": "24h",
		"metadata":          map[string]string{"job": "nightly"},
	}, env.key)
	if status != http.StatusOK {
		t.Fatalf("create status = %d, body %v", status, created)
	}
	if created["object"] != "batch" || created["status"] != "validating" {
		t.Fatalf("create must report a validating batch object: %v", created)
	}
	if created["source"] != "file" || created["sealed_to"] != "coordinator" {
		t.Fatalf("source/sealed_to = %v/%v", created["source"], created["sealed_to"])
	}
	batchID, _ := created["id"].(string)
	if !strings.HasPrefix(batchID, "batch_") {
		t.Fatalf("batch id = %q", batchID)
	}

	// The next read observes the admitted batch.
	status, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if status != http.StatusOK {
		t.Fatalf("get status = %d, body %v", status, got)
	}
	if got["status"] != "in_progress" {
		t.Fatalf("status = %v, want in_progress", got["status"])
	}
	if total, completed, failed := requestCounts(t, got); total != 3 || completed != 0 || failed != 0 {
		t.Fatalf("request_counts = %d/%d/%d, want 3/0/0", total, completed, failed)
	}
	if got["output_file_id"] != nil || got["error_file_id"] != nil {
		t.Fatalf("an open batch has no result files: %v", got)
	}
	if got["in_progress_at"] == nil || got["expires_at"] == nil {
		t.Fatalf("timestamps missing: %v", got)
	}

	// 3. Every item is ciphertext on disk and the input file's blob is gone.
	assertNoPlaintextOnDisk(t, env.blobDir, batchTestPrompt)
	if names := blobFiles(t, env.blobDir); len(names) != 3 {
		t.Fatalf("want exactly the 3 item blobs on disk, got %v", names)
	}
	if _, err := env.srv.BatchBlobs().Raw(fileID); err == nil {
		t.Fatal("the input file blob must be deleted once the items exist")
	}
	f, ok := env.st.GetBatchFile(env.account, fileID)
	if !ok || f.PurgedAt == nil {
		t.Fatalf("the input file row must be marked purged, got %+v", f)
	}
	if status, _ := env.request(http.MethodGet, "/v1/files/"+fileID+"/content", "", nil, env.key); status != http.StatusNotFound {
		t.Fatalf("purged content status = %d, want 404", status)
	}

	// 4. Listing shows it; another account sees none of it.
	status, list := env.getJSON("/v1/batches", env.key)
	if status != http.StatusOK {
		t.Fatalf("list status = %d", status)
	}
	data, _ := list["data"].([]any)
	if len(data) != 1 {
		t.Fatalf("list returned %d batches, want 1", len(data))
	}

	otherKey, _ := env.newKey("acct-other")
	for _, probe := range []struct {
		method, path string
	}{
		{http.MethodGet, "/v1/batches/" + batchID},
		{http.MethodPost, "/v1/batches/" + batchID + "/cancel"},
		{http.MethodGet, "/v1/files/" + fileID},
		{http.MethodGet, "/v1/files/" + fileID + "/content"},
	} {
		status, _ := env.request(probe.method, probe.path, "application/json", nil, otherKey)
		if status != http.StatusNotFound {
			t.Fatalf("%s %s from another account = %d, want 404", probe.method, probe.path, status)
		}
	}
	status, otherList := env.getJSON("/v1/batches", otherKey)
	if otherData, _ := otherList["data"].([]any); status != http.StatusOK || len(otherData) != 0 {
		t.Fatalf("another account's list = %d %v", status, otherList)
	}

	// 5. Cancel moves the batch to cancelling; finalize settles it as cancelled
	//    with no result files.
	status, cancelled := env.postJSON("/v1/batches/"+batchID+"/cancel", map[string]any{}, env.key)
	if status != http.StatusOK || cancelled["status"] != "cancelling" {
		t.Fatalf("cancel = %d %v", status, cancelled)
	}
	items, err := env.st.ListItems(batchID)
	if err != nil {
		t.Fatalf("list items: %v", err)
	}
	for _, it := range items {
		if it.State != store.ItemCancelled {
			t.Fatalf("item %s state = %s, want cancelled", it.ID, it.State)
		}
	}
	res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC())
	if err != nil {
		t.Fatalf("finalize: %v", err)
	}
	if res == nil || res.Status != store.BatchCancelled {
		t.Fatalf("finalize = %+v, want cancelled", res)
	}
	if res.OutputFileID != nil || res.ErrorFileID != nil {
		t.Fatalf("a cancelled batch produces no result files: %+v", res)
	}
	status, final := env.getJSON("/v1/batches/"+batchID, env.key)
	if status != http.StatusOK || final["status"] != "cancelled" || final["cancelled_at"] == nil {
		t.Fatalf("final batch = %d %v", status, final)
	}
	if names := blobFiles(t, env.blobDir); len(names) != 0 {
		t.Fatalf("finalize must drop every item input blob, still have %v", names)
	}

	// 6. Nothing the request path logged carries a prompt or a custom_id.
	logs := env.logs.String()
	if strings.Contains(logs, batchTestPrompt) {
		t.Fatalf("a log line carries the prompt:\n%s", logs)
	}
	if strings.Contains(logs, "req-0") || strings.Contains(logs, "custom_id") {
		t.Fatalf("a log line carries a custom_id:\n%s", logs)
	}
	if strings.Contains(logs, "input.jsonl") {
		t.Fatalf("a log line carries the uploaded filename:\n%s", logs)
	}
	if strings.Contains(logs, "nightly") {
		t.Fatalf("a log line carries batch metadata:\n%s", logs)
	}
}

func TestInlineBatchOpenRouterForm(t *testing.T) {
	env := newBatchEnv(t)

	status, created := env.postJSON("/v1/batches", map[string]any{
		"endpoint": "/v1/chat/completions",
		"model":    batchTestModel,
		"requests": []map[string]any{
			{"custom_id": "a", "body": map[string]any{
				"messages": []map[string]any{{"role": "user", "content": batchTestPrompt}},
			}},
		},
	}, env.key)
	if status != http.StatusAccepted {
		t.Fatalf("inline create status = %d, body %v", status, created)
	}
	if created["source"] != "inline" || created["model"] != batchTestModel {
		t.Fatalf("inline batch object = %v", created)
	}
	if total, _, _ := requestCounts(t, created); total != 1 {
		t.Fatalf("request_counts.total = %d, want 1", total)
	}
	batchID, _ := created["id"].(string)

	// Settle the single item the way the dispatcher will, then read the
	// inline results back off the batch object.
	items := env.claimAll(batchID)
	body := []byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"ok"}}]}`)
	env.settleSucceeded(items[0], body)

	res, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC())
	if err != nil || res == nil || res.Status != store.BatchCompleted {
		t.Fatalf("finalize = %+v, %v", res, err)
	}

	status, got := env.getJSON("/v1/batches/"+batchID, env.key)
	if status != http.StatusOK || got["status"] != "completed" {
		t.Fatalf("get = %d %v", status, got)
	}
	results, _ := got["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("results = %v, want one entry", got["results"])
	}
	first, _ := results[0].(map[string]any)
	if first["custom_id"] != "a" {
		t.Fatalf("results[0].custom_id = %v", first["custom_id"])
	}
	response, ok := first["response"].(map[string]any)
	if !ok {
		t.Fatalf("results[0].response = %v", first["response"])
	}
	if code, _ := response["status_code"].(float64); int(code) != 200 {
		t.Fatalf("status_code = %v", response["status_code"])
	}
	inner, _ := response["body"].(map[string]any)
	if inner["id"] != "chatcmpl-1" {
		t.Fatalf("response body was not carried through: %v", response["body"])
	}
}

// claimAll moves every pending item of a batch to inflight, standing in for the
// dispatcher's claim.
func (e *batchEnv) claimAll(batchID string) []*store.BatchItem {
	e.t.Helper()
	items, err := e.st.ClaimPendingItems(batchID, maxInlineRequests, time.Now().UTC())
	if err != nil {
		e.t.Fatalf("claim: %v", err)
	}
	return items
}

// settleSucceeded writes the result blob the way the dispatcher does (keyed by
// item id, before the item is settled) and then finishes the item.
func (e *batchEnv) settleSucceeded(item *store.BatchItem, body []byte) {
	e.t.Helper()
	if err := e.srv.BatchBlobs().PutPlain(BatchItemResultRef(item.ID), body); err != nil {
		e.t.Fatalf("write result blob: %v", err)
	}
	ok, err := e.st.FinishItem(store.ItemResult{
		ItemID: item.ID, Succeeded: true, RequestID: "req_" + item.ID,
		ResultBlobRef: BatchItemResultRef(item.ID), PromptTokens: 3, CompletionTokens: 5,
	}, time.Now().UTC())
	if err != nil || !ok {
		e.t.Fatalf("finish item: ok=%v err=%v", ok, err)
	}
}

// settleFailed settles an item as failed with the fixed request_failed code.
func (e *batchEnv) settleFailed(item *store.BatchItem) {
	e.t.Helper()
	ok, err := e.st.FinishItem(store.ItemResult{
		ItemID: item.ID, Succeeded: false, ErrorCode: batchRequestFailedCode,
	}, time.Now().UTC())
	if err != nil || !ok {
		e.t.Fatalf("finish failed item: ok=%v err=%v", ok, err)
	}
}

func TestResultPublicKeyIsStrict(t *testing.T) {
	env := newBatchEnv(t)

	short := base64.StdEncoding.EncodeToString(make([]byte, 31))
	status, body := env.postJSON("/v1/batches", map[string]any{
		"endpoint":          "/v1/chat/completions",
		"model":             batchTestModel,
		"result_public_key": short,
		"requests":          []map[string]any{{"custom_id": "a", "body": map[string]any{"messages": []any{}}}},
	}, env.key)
	if status != http.StatusBadRequest || errorCode(t, body) != "invalid_result_public_key" {
		t.Fatalf("31-byte key = %d %v", status, body)
	}

	if status, body := env.postJSON("/v1/batches", map[string]any{
		"endpoint":          "/v1/chat/completions",
		"model":             batchTestModel,
		"result_public_key": "not base64!!",
		"requests":          []map[string]any{{"custom_id": "a", "body": map[string]any{"messages": []any{}}}},
	}, env.key); status != http.StatusBadRequest || errorCode(t, body) != "invalid_result_public_key" {
		t.Fatalf("malformed key = %d %v", status, body)
	}

	// A well-formed consumer key marks the batch sealed_to consumer, and a
	// result sealed to that key is opaque to the coordinator's own store key.
	consumer, err := e2e.GenerateSessionKeys()
	if err != nil {
		t.Fatalf("consumer keys: %v", err)
	}
	status, created := env.postJSON("/v1/batches", map[string]any{
		"endpoint":          "/v1/chat/completions",
		"model":             batchTestModel,
		"result_public_key": base64.StdEncoding.EncodeToString(consumer.PublicKey[:]),
		"requests": []map[string]any{{"custom_id": "a", "body": map[string]any{
			"messages": []map[string]any{{"role": "user", "content": batchTestPrompt}},
		}}},
	}, env.key)
	if status != http.StatusAccepted {
		t.Fatalf("create with a valid key = %d %v", status, created)
	}
	if created["sealed_to"] != "consumer" {
		t.Fatalf("sealed_to = %v, want consumer", created["sealed_to"])
	}
	batchID, _ := created["id"].(string)

	// Seal a result to the consumer key the way the dispatcher will, then check
	// the store cannot read it while assembly still carries it verbatim.
	items := env.claimAll(batchID)
	plain := []byte(`{"id":"chatcmpl-secret"}`)
	if err := env.srv.BatchBlobs().PutTo(BatchItemResultRef(items[0].ID), plain, consumer.PublicKey); err != nil {
		t.Fatalf("seal to consumer: %v", err)
	}
	if _, err := env.srv.BatchBlobs().Open(BatchItemResultRef(items[0].ID)); err == nil {
		t.Fatal("the coordinator must not be able to open a consumer-sealed result")
	}
	ok, err := env.st.FinishItem(store.ItemResult{
		ItemID: items[0].ID, Succeeded: true, ResultBlobRef: BatchItemResultRef(items[0].ID),
	}, time.Now().UTC())
	if err != nil || !ok {
		t.Fatalf("finish: ok=%v err=%v", ok, err)
	}
	if _, err := env.srv.FinalizeBatchIfDone(batchID, time.Now().UTC()); err != nil {
		t.Fatalf("finalize: %v", err)
	}

	_, got := env.getJSON("/v1/batches/"+batchID, env.key)
	results, _ := got["results"].([]any)
	if len(results) != 1 {
		t.Fatalf("results = %v", got["results"])
	}
	response, _ := results[0].(map[string]any)["response"].(map[string]any)
	envelope, _ := response["body"].(map[string]any)
	if envelope["ciphertext"] == nil || envelope["ephemeral_public_key"] == nil {
		t.Fatalf("a consumer-sealed result body must be an EncryptedPayload, got %v", response["body"])
	}

	// And the consumer can actually open it.
	var payload e2e.EncryptedPayload
	raw, err := json.Marshal(envelope)
	if err != nil {
		t.Fatalf("re-encode payload: %v", err)
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("decode payload: %v", err)
	}
	opened, err := e2e.DecryptWithPrivateKey(&payload, consumer.PrivateKey)
	if err != nil {
		t.Fatalf("consumer decrypt: %v", err)
	}
	if string(opened) != string(plain) {
		t.Fatalf("decrypted %q, want %q", opened, plain)
	}
}

func TestCustomIDAndMetadataConstraints(t *testing.T) {
	env := newBatchEnv(t)

	inline := func(customID string, metadata map[string]string) (int, map[string]any) {
		body := map[string]any{
			"endpoint": "/v1/chat/completions",
			"model":    batchTestModel,
			"requests": []map[string]any{{"custom_id": customID, "body": map[string]any{"messages": []any{}}}},
		}
		if metadata != nil {
			body["metadata"] = metadata
		}
		return env.postJSON("/v1/batches", body, env.key)
	}

	if status, body := inline(strings.Repeat("x", 65), nil); status != http.StatusBadRequest || errorCode(t, body) != "invalid_custom_id" {
		t.Fatalf("65-char custom_id = %d %v", status, body)
	}
	if status, body := inline("a b", nil); status != http.StatusBadRequest || errorCode(t, body) != "invalid_custom_id" {
		t.Fatalf("spaced custom_id = %d %v", status, body)
	}

	tooManyKeys := make(map[string]string, maxMetadataKeys+1)
	for i := 0; i <= maxMetadataKeys; i++ {
		tooManyKeys[fmt.Sprintf("k%d", i)] = "v"
	}
	if status, body := inline("ok", tooManyKeys); status != http.StatusBadRequest || errorCode(t, body) != "invalid_metadata" {
		t.Fatalf("17 metadata keys = %d %v", status, body)
	}
}

func TestBatchRejectsWrongWindowAndEndpoint(t *testing.T) {
	env := newBatchEnv(t)

	base := func() map[string]any {
		return map[string]any{
			"endpoint":          "/v1/chat/completions",
			"completion_window": "24h",
			"model":             batchTestModel,
			"requests":          []map[string]any{{"custom_id": "a", "body": map[string]any{"messages": []any{}}}},
		}
	}

	body := base()
	body["completion_window"] = "48h"
	if status, got := env.postJSON("/v1/batches", body, env.key); status != http.StatusBadRequest ||
		errorCode(t, got) != "invalid_completion_window" {
		t.Fatalf("48h window = %d %v", status, got)
	}

	body = base()
	body["endpoint"] = "/v1/embeddings"
	if status, got := env.postJSON("/v1/batches", body, env.key); status != http.StatusBadRequest ||
		errorCode(t, got) != "invalid_endpoint" {
		t.Fatalf("embeddings endpoint = %d %v", status, got)
	}

	// Neither form supplied, and both forms supplied, are equally invalid.
	if status, got := env.postJSON("/v1/batches", map[string]any{"endpoint": "/v1/chat/completions"}, env.key); status != http.StatusBadRequest {
		t.Fatalf("empty create = %d %v", status, got)
	}
	body = base()
	body["input_file_id"] = "file-deadbeef"
	if status, got := env.postJSON("/v1/batches", body, env.key); status != http.StatusBadRequest {
		t.Fatalf("both forms = %d %v", status, got)
	}
}

func TestFilesRejectsOversize(t *testing.T) {
	env := newBatchEnv(t)

	// One line comfortably over the 16 MiB cap.
	oversize := fmt.Sprintf(`{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":%q,"messages":[{"role":"user","content":%q}]}}`,
		batchTestModel, strings.Repeat("x", maxFileBytes+1024))
	status, body := env.uploadMultipart(oversize, "batch", "big.jsonl", env.key)
	if status != http.StatusRequestEntityTooLarge {
		t.Fatalf("oversize upload = %d %v", status, body)
	}
	if names := blobFiles(t, env.blobDir); len(names) != 0 {
		t.Fatalf("a rejected upload must store nothing, got %v", names)
	}
}

func TestFilesRejectsWrongPurposeAndInvalidLines(t *testing.T) {
	env := newBatchEnv(t)

	if status, body := env.uploadMultipart(jsonlLines(1), "fine-tune", "x.jsonl", env.key); status != http.StatusBadRequest ||
		errorCode(t, body) != "invalid_purpose" {
		t.Fatalf("wrong purpose = %d %v", status, body)
	}
	bad := `{"custom_id":"a","method":"POST","url":"/v1/chat/completions","body":{"model":"unknown-model"}}`
	if status, body := env.uploadMultipart(bad, "batch", "x.jsonl", env.key); status != http.StatusBadRequest ||
		errorCode(t, body) != "model_not_found" {
		t.Fatalf("unknown model = %d %v", status, body)
	}
	if names := blobFiles(t, env.blobDir); len(names) != 0 {
		t.Fatalf("a rejected upload must store nothing, got %v", names)
	}
}

func TestSealedEnvelopeUploadUnsealsInTransitThenSealsPerItem(t *testing.T) {
	env := newBatchEnv(t)
	coordKey, err := e2e.DeriveCoordinatorKey(senderTestMnemonic)
	if err != nil {
		t.Fatalf("derive coordinator key: %v", err)
	}
	env.srv.SetCoordinatorKey(coordKey)

	content := jsonlLines(2)
	envelope, err := json.Marshal(sealedFileEnvelope{
		Purpose:       "batch",
		Filename:      "sealed.jsonl",
		ContentBase64: base64.StdEncoding.EncodeToString([]byte(content)),
	})
	if err != nil {
		t.Fatalf("encode envelope: %v", err)
	}
	sealed, _, ephemPriv := sealRequest(t, envelope, coordKey.PublicKey, coordKey.KID)

	status, raw := env.request(http.MethodPost, "/v1/files", SealedContentType, bytes.NewReader(sealed), env.key)
	if status != http.StatusOK {
		t.Fatalf("sealed upload = %d %s", status, raw)
	}
	fileObj := decodeObject(t, unsealResponse(t, raw, coordKey.PublicKey, ephemPriv))
	fileID, _ := fileObj["id"].(string)
	if !strings.HasPrefix(fileID, "file-") {
		t.Fatalf("sealed upload returned %v", fileObj)
	}
	if fileObj["filename"] != "sealed.jsonl" {
		t.Fatalf("filename = %v", fileObj["filename"])
	}

	// The stored file is batch-store ciphertext and the plaintext never landed
	// anywhere in the blob directory.
	stored, err := env.srv.BatchBlobs().Raw(fileID)
	if err != nil {
		t.Fatalf("raw: %v", err)
	}
	if bytes.Contains(stored, []byte(batchTestPrompt)) {
		t.Fatal("the stored file blob is not ciphertext")
	}
	assertNoPlaintextOnDisk(t, env.blobDir, batchTestPrompt)
	opened, err := env.srv.BatchBlobs().Open(fileID)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if string(opened) != content {
		t.Fatal("the sealed upload did not round-trip through the batch store key")
	}
}

func TestAdmissionFeasibility(t *testing.T) {
	env := newBatchEnv(t)

	// Seed a very slow observed completion rate: four items finished in the
	// trailing hour is about 0.0011 items/s, so a thousand more cannot fit in
	// 80% of a 24-hour window.
	seedID := env.seedCompletedBatch(4)
	if rate, known := env.st.CompletionRate(batchRateWindow, time.Now().UTC()); !known || rate > 0.01 {
		t.Fatalf("seeded rate = %v known=%v, want a small known rate (batch %s)", rate, known, seedID)
	}

	requests := make([]map[string]any, 0, 1000)
	for i := 0; i < 1000; i++ {
		requests = append(requests, map[string]any{
			"custom_id": fmt.Sprintf("r%d", i),
			"body":      map[string]any{"messages": []any{}},
		})
	}
	status, body := env.postJSON("/v1/batches", map[string]any{
		"endpoint": "/v1/chat/completions",
		"model":    batchTestModel,
		"requests": requests,
	}, env.key)
	if status != http.StatusBadRequest || errorCode(t, body) != "batch_infeasible" {
		t.Fatalf("infeasible batch = %d %v", status, body)
	}

	// A single request still fits.
	if status, body := env.postJSON("/v1/batches", map[string]any{
		"endpoint": "/v1/chat/completions",
		"model":    batchTestModel,
		"requests": requests[:1],
	}, env.key); status != http.StatusAccepted {
		t.Fatalf("one feasible request = %d %v", status, body)
	}
}

// seedCompletedBatch creates and fully settles n items so CompletionRate has
// something to report.
func (e *batchEnv) seedCompletedBatch(n int) string {
	e.t.Helper()
	requests := make([]map[string]any, 0, n)
	for i := 0; i < n; i++ {
		requests = append(requests, map[string]any{
			"custom_id": fmt.Sprintf("seed%d", i),
			"body":      map[string]any{"messages": []any{}},
		})
	}
	status, created := e.postJSON("/v1/batches", map[string]any{
		"endpoint": "/v1/chat/completions",
		"model":    batchTestModel,
		"requests": requests,
	}, e.key)
	if status != http.StatusAccepted {
		e.t.Fatalf("seed create = %d %v", status, created)
	}
	batchID, _ := created["id"].(string)
	for _, it := range e.claimAll(batchID) {
		e.settleSucceeded(it, []byte(`{"id":"seed"}`))
	}
	if _, err := e.srv.FinalizeBatchIfDone(batchID, time.Now().UTC()); err != nil {
		e.t.Fatalf("seed finalize: %v", err)
	}
	return batchID
}

func TestBatchRoutesRequireAuth(t *testing.T) {
	env := newBatchEnv(t)
	for _, probe := range []struct{ method, path string }{
		{http.MethodPost, "/v1/files"},
		{http.MethodGet, "/v1/files/file-x"},
		{http.MethodGet, "/v1/files/file-x/content"},
		{http.MethodPost, "/v1/batches"},
		{http.MethodGet, "/v1/batches"},
		{http.MethodGet, "/v1/batches/batch_x"},
		{http.MethodPost, "/v1/batches/batch_x/cancel"},
	} {
		req, err := http.NewRequest(probe.method, env.ts.URL+probe.path, nil)
		if err != nil {
			t.Fatalf("build: %v", err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", probe.method, probe.path, err)
		}
		resp.Body.Close()
		if resp.StatusCode != http.StatusUnauthorized {
			t.Fatalf("%s %s unauthenticated = %d, want 401", probe.method, probe.path, resp.StatusCode)
		}
	}
}

func TestBatchRoutesReport503WithoutABlobStore(t *testing.T) {
	logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	st := store.NewMemory(store.Config{})
	srv := NewServer(registry.New(logger), st, ServerConfig{}, logger)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	raw, _, err := st.CreateAPIKey("acct", store.APIKeyCreate{})
	if err != nil {
		t.Fatalf("create key: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, ts.URL+"/v1/batches", nil)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	req.Header.Set("Authorization", "Bearer "+raw)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("do: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
	body, _ := io.ReadAll(resp.Body)
	if code := errorCode(t, decodeObject(t, body)); code != "batch_unavailable" {
		t.Fatalf("code = %q, want batch_unavailable", code)
	}
}

// blobRefsUnderDir is a small guard that the ids we mint are legal blob refs,
// so consumer input can never become a path.
func TestMintedIDsAreLegalBlobRefs(t *testing.T) {
	dir := t.TempDir()
	key, err := sealedblob.RandomKey()
	if err != nil {
		t.Fatalf("key: %v", err)
	}
	bs, err := sealedblob.New(dir, key)
	if err != nil {
		t.Fatalf("store: %v", err)
	}
	for _, prefix := range []string{"file-", "batch_", "bitem_"} {
		id, err := newBatchID(prefix)
		if err != nil {
			t.Fatalf("mint %s: %v", prefix, err)
		}
		if err := bs.PutPlain(id, []byte("x")); err != nil {
			t.Fatalf("%q is not a legal blob ref: %v", id, err)
		}
	}
}
