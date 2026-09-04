package testbed

// batchclient.go — the Batch API surface an e2e test drives.
//
// Everything here speaks the same HTTP the OpenAI SDK speaks against a running
// Suite's coordinator (docs/design/tidal-batch-lane.md §3.6): upload a JSONL
// input file, create a batch from it, poll it to a terminal state, read the
// output file back. Nothing reaches into the store or the dispatcher, so a
// test written against this client proves the consumer-visible contract rather
// than an internal one.
//
// Privacy: prompts are consumer content. This client never logs a request
// body, a custom_id, or a completion — only ids, counts, and HTTP status.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"strings"
	"time"
)

// batchClientTimeout bounds a single Batch API call. Every route here is
// metadata-only or a blob read; none of them wait on a provider, so a request
// that has not answered in 60 s is a fault, not slow inference.
const batchClientTimeout = 60 * time.Second

// BatchClient issues Batch API calls against a Suite's coordinator as one
// account. Build it with NewBatchClient so it inherits the suite's base URL and
// one of the suite's real API keys — the batch lane resolves the owning account
// from the key, and every read is scoped to that account.
type BatchClient struct {
	Suite  *Suite
	APIKey string
	HTTP   *http.Client
}

// NewBatchClient binds a client to a running suite and the given API key
// (typically suite.Users[i].APIKey).
func NewBatchClient(s *Suite, apiKey string) *BatchClient {
	return &BatchClient{
		Suite:  s,
		APIKey: apiKey,
		HTTP:   &http.Client{Timeout: batchClientTimeout},
	}
}

// BatchInputLine is one JSONL request line in an input file.
type BatchInputLine struct {
	CustomID  string
	Model     string
	Prompt    string
	MaxTokens int
	// URL is the batch endpoint the line targets; empty means
	// "/v1/chat/completions".
	URL string
}

// MarshalJSONL renders one input line in the OpenAI batch input shape.
func (l BatchInputLine) MarshalJSONL() ([]byte, error) {
	url := l.URL
	if url == "" {
		url = BatchChatCompletionsEndpoint
	}
	return json.Marshal(map[string]any{
		"custom_id": l.CustomID,
		"method":    http.MethodPost,
		"url":       url,
		"body": map[string]any{
			"model":       l.Model,
			"messages":    []map[string]string{{"role": "user", "content": l.Prompt}},
			"max_tokens":  l.MaxTokens,
			"temperature": 0.0,
		},
	})
}

// BatchChatCompletionsEndpoint is the only endpoint these helpers submit to.
const BatchChatCompletionsEndpoint = "/v1/chat/completions"

// BatchInputJSONL renders a whole input file: one JSON object per line,
// newline-terminated.
func BatchInputJSONL(lines []BatchInputLine) ([]byte, error) {
	var buf bytes.Buffer
	for i, line := range lines {
		encoded, err := line.MarshalJSONL()
		if err != nil {
			return nil, fmt.Errorf("marshal batch input line %d: %w", i, err)
		}
		buf.Write(encoded)
		buf.WriteByte('\n')
	}
	return buf.Bytes(), nil
}

// BatchRequestCounts mirrors the batch object's request_counts.
type BatchRequestCounts struct {
	Completed int `json:"completed"`
	Failed    int `json:"failed"`
	Total     int `json:"total"`
}

// Settled is the number of items that reached a terminal state.
func (c BatchRequestCounts) Settled() int { return c.Completed + c.Failed }

// BatchObject is the subset of the OpenAI batch object these tests read.
type BatchObject struct {
	ID            string             `json:"id"`
	Object        string             `json:"object"`
	Status        string             `json:"status"`
	Endpoint      string             `json:"endpoint"`
	Source        string             `json:"source"`
	Model         string             `json:"model"`
	InputFileID   string             `json:"input_file_id"`
	OutputFileID  string             `json:"output_file_id"`
	ErrorFileID   string             `json:"error_file_id"`
	RequestCounts BatchRequestCounts `json:"request_counts"`
	CreatedAt     int64              `json:"created_at"`
	InProgressAt  int64              `json:"in_progress_at"`
	CompletedAt   int64              `json:"completed_at"`
	FailedAt      int64              `json:"failed_at"`
	CancelledAt   int64              `json:"cancelled_at"`
}

// BatchIsTerminal reports whether a batch status can no longer change.
func BatchIsTerminal(status string) bool {
	switch status {
	case "completed", "failed", "cancelled", "expired":
		return true
	default:
		return false
	}
}

// UploadJSONL uploads a batch input file through the multipart form every
// OpenAI SDK sends and returns the minted file id.
func (c *BatchClient) UploadJSONL(ctx context.Context, filename string, content []byte) (string, error) {
	var body bytes.Buffer
	mw := multipart.NewWriter(&body)
	if err := mw.WriteField("purpose", "batch"); err != nil {
		return "", fmt.Errorf("write purpose field: %w", err)
	}
	part, err := mw.CreateFormFile("file", filename)
	if err != nil {
		return "", fmt.Errorf("create file part: %w", err)
	}
	if _, err := part.Write(content); err != nil {
		return "", fmt.Errorf("write file part: %w", err)
	}
	if err := mw.Close(); err != nil {
		return "", fmt.Errorf("close multipart writer: %w", err)
	}

	var file struct {
		ID    string `json:"id"`
		Bytes int64  `json:"bytes"`
	}
	if err := c.do(ctx, http.MethodPost, "/v1/files", mw.FormDataContentType(), body.Bytes(), &file); err != nil {
		return "", err
	}
	if file.ID == "" {
		return "", fmt.Errorf("upload returned no file id")
	}
	return file.ID, nil
}

// CreateBatch creates a batch from an uploaded input file (the OpenAI form).
// endpoint defaults to /v1/chat/completions.
func (c *BatchClient) CreateBatch(ctx context.Context, inputFileID, endpoint string) (BatchObject, error) {
	if endpoint == "" {
		endpoint = BatchChatCompletionsEndpoint
	}
	payload, err := json.Marshal(map[string]any{
		"input_file_id":     inputFileID,
		"endpoint":          endpoint,
		"completion_window": "24h",
	})
	if err != nil {
		return BatchObject{}, fmt.Errorf("marshal create body: %w", err)
	}
	var batch BatchObject
	if err := c.do(ctx, http.MethodPost, "/v1/batches", "application/json", payload, &batch); err != nil {
		return BatchObject{}, err
	}
	return batch, nil
}

// SubmitBatch is upload + create in one call, the shape every phase of the
// co-serving benchmark uses.
func (c *BatchClient) SubmitBatch(ctx context.Context, filename string, lines []BatchInputLine) (BatchObject, error) {
	content, err := BatchInputJSONL(lines)
	if err != nil {
		return BatchObject{}, err
	}
	fileID, err := c.UploadJSONL(ctx, filename, content)
	if err != nil {
		return BatchObject{}, err
	}
	return c.CreateBatch(ctx, fileID, BatchChatCompletionsEndpoint)
}

// GetBatch retrieves one batch.
func (c *BatchClient) GetBatch(ctx context.Context, batchID string) (BatchObject, error) {
	var batch BatchObject
	if err := c.do(ctx, http.MethodGet, "/v1/batches/"+batchID, "", nil, &batch); err != nil {
		return BatchObject{}, err
	}
	return batch, nil
}

// CancelBatch requests cancellation and returns the batch as the coordinator
// rendered it in the cancel response.
func (c *BatchClient) CancelBatch(ctx context.Context, batchID string) (BatchObject, error) {
	var batch BatchObject
	if err := c.do(ctx, http.MethodPost, "/v1/batches/"+batchID+"/cancel", "application/json", []byte(`{}`), &batch); err != nil {
		return BatchObject{}, err
	}
	return batch, nil
}

// FileContent downloads a batch output or error file.
func (c *BatchClient) FileContent(ctx context.Context, fileID string) ([]byte, error) {
	raw, err := c.raw(ctx, http.MethodGet, "/v1/files/"+fileID+"/content", "", nil)
	if err != nil {
		return nil, err
	}
	return raw, nil
}

// PollWhile samples the batch every interval until it reaches a terminal
// status, observe returns false, or ctx ends. observe is called once per
// sample, terminal samples included, and returning false stops the loop
// without treating the batch as finished. The last observed batch is returned.
func (c *BatchClient) PollWhile(
	ctx context.Context,
	batchID string,
	interval time.Duration,
	observe func(BatchObject) bool,
) (BatchObject, error) {
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	var last BatchObject
	for {
		batch, err := c.GetBatch(ctx, batchID)
		if err != nil {
			return last, err
		}
		last = batch
		if observe != nil && !observe(batch) {
			return last, nil
		}
		if BatchIsTerminal(batch.Status) {
			return last, nil
		}
		select {
		case <-ctx.Done():
			return last, ctx.Err()
		case <-ticker.C:
		}
	}
}

// PollUntilTerminal polls until the batch settles or timeout elapses.
func (c *BatchClient) PollUntilTerminal(
	ctx context.Context,
	batchID string,
	timeout, interval time.Duration,
) (BatchObject, error) {
	pollCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	batch, err := c.PollWhile(pollCtx, batchID, interval, nil)
	if err != nil {
		return batch, fmt.Errorf("poll batch %s: %w (last status %q, counts %+v)",
			batchID, err, batch.Status, batch.RequestCounts)
	}
	return batch, nil
}

// do issues one request and decodes a JSON response into out.
func (c *BatchClient) do(ctx context.Context, method, path, contentType string, body []byte, out any) error {
	raw, err := c.raw(ctx, method, path, contentType, body)
	if err != nil {
		return err
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(raw, out); err != nil {
		return fmt.Errorf("%s %s: decode response: %w", method, path, err)
	}
	return nil
}

// raw issues one request and returns the response body, turning any non-2xx
// into an error that carries the status and the coordinator's own error
// envelope. Batch error messages are fixed strings by construction
// (docs/design/tidal-batch-lane.md §3.6), so echoing one leaks nothing.
func (c *BatchClient) raw(ctx context.Context, method, path, contentType string, body []byte) ([]byte, error) {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.Suite.Coordinator.BaseURL()+path, reader)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w", method, path, err)
	}
	req.Header.Set("Authorization", "Bearer "+c.APIKey)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("%s %s: read response: %w", method, path, err)
	}
	if resp.StatusCode/100 != 2 {
		return nil, fmt.Errorf("%s %s: status %d: %s",
			method, path, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	return raw, nil
}
