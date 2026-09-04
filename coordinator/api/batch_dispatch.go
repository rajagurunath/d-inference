package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"

	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batch_dispatch.go is the api-layer surface of the batch lane
// (docs/design/tidal-batch-lane.md §3.4): the vocabulary the no-capacity
// terminal answers with, and the entry point the batch dispatcher uses to run
// one item through the ordinary dispatch funnel.

const (
	// batchNoCapacityCode is the bounded error/rejection code a batch-lane
	// request gets when no provider slot has headroom for it. It is also the
	// ErrCode BatchOutcome carries, so the batch dispatcher can tell "nothing
	// free right now, re-offer me next tick" apart from a real failure without
	// parsing prose.
	batchNoCapacityCode = "no_capacity"
	// batchNoCapacityRetryAfterSec is the Retry-After the batch lane advertises
	// on that 429. It matches the dispatcher's 1 Hz tick scale: a few seconds is
	// long enough for online traffic to drain a slot and short enough that a
	// paced client (OpenRouter's synchronous service_tier=batch calls) keeps
	// making progress.
	batchNoCapacityRetryAfterSec = 5
)

// BatchOutcome is the result of running one batch item through the dispatch
// funnel. Exactly one of a delivered response or an ErrCode is meaningful:
// ErrCode is empty on success, batchNoCapacityCode ("no_capacity") when no
// provider slot had batch headroom, and "request_failed" for every other
// non-success terminal — the same bounded vocabulary the batch output/error
// files use (docs/design/tidal-batch-lane.md §3.6).
type BatchOutcome struct {
	// RequestID is the coordinator-owned inference job id of the attempt that
	// committed, echoed from the X-Inference-Job-ID header. Empty when nothing
	// reached a provider.
	RequestID string
	// PromptTokens / CompletionTokens are the provider-reported usage of the
	// committed attempt, read back from the assembled response body.
	PromptTokens, CompletionTokens int
	// ResponseBody is the complete non-streaming OpenAI response body on
	// success, and the OpenAI-shaped error body otherwise.
	ResponseBody []byte
	// ErrCode is "" on success; see the type comment for the failure vocabulary.
	ErrCode string
}

// batchRequestFailedCode is the bounded error code for a batch attempt that
// reached a terminal that is not a capacity refusal.
const batchRequestFailedCode = "request_failed"

// DispatchBatchItem runs one batch request through the standard consumer
// dispatch funnel on registry.LaneBatch and waits for it to complete, returning
// the assembled non-streaming response. It is the single entry point the batch
// dispatcher (coordinator/batchlane, PR3b) uses; nothing about the request
// reaches a provider differently from an online one — only the lane differs,
// and the lane is what restricts placement to headroom slots, suppresses the
// wait queue and the hedge, and keeps the attempt out of reputation and TTFT
// calibration.
//
// Implementation note (deliberate, see the PR body): the item is driven through
// the real handleChatCompletions with a synthetic request and an
// httptest.ResponseRecorder rather than through an extracted prelude. The chat
// handler's prelude is ~350 lines of interleaved parsing, alias resolution,
// media inlining, tool-constraint validation, token admission and balance
// reservation whose intermediate values are re-derived by several closures; a
// split that preserved all of it exactly was not achievable inside this PR's
// budget, and a partial copy would be the drift the plan's "no parallel
// formula" rule exists to prevent. Driving the real handler is therefore the
// path that GUARANTEES a batch item is admitted, billed, resolved and dispatched
// by exactly the same code as an online one. Extracting the prelude is tracked
// as a follow-up.
//
// accountID is the billing/consumer identity (the same value requireAuth stamps
// on an online request) and apiKeyID the attributing key, "" when the caller has
// none. body is the item's OpenAI chat-completions body; its model field is
// overwritten with model and streaming is forced off, because a batch item is
// assembled, never streamed.
func (s *Server) DispatchBatchItem(
	ctx context.Context,
	accountID, apiKeyID, model string,
	body []byte,
) (BatchOutcome, error) {
	if s == nil {
		return BatchOutcome{}, errors.New("nil server")
	}
	if model == "" {
		return BatchOutcome{}, errors.New("batch dispatch: model is required")
	}
	parsed, err := decodeInferenceJSONObject(body)
	if err != nil {
		return BatchOutcome{}, fmt.Errorf("batch dispatch: %w", err)
	}
	parsed["model"] = model
	// A batch item is assembled from CompleteCh, never relayed as SSE. Deleting
	// rather than setting false keeps the forwarded body byte-minimal.
	delete(parsed, "stream")
	itemBody, err := marshalForwardBody(parsed)
	if err != nil {
		return BatchOutcome{}, fmt.Errorf("batch dispatch: %w", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(itemBody))
	req.Header.Set("Content-Type", "application/json")
	rctx := withRequestLane(ctx, registry.LaneBatch)
	rctx = context.WithValue(rctx, ctxKeyConsumer, accountID)
	if apiKeyID != "" {
		rctx = context.WithValue(rctx, ctxKeyAPIKey, &store.APIKey{ID: apiKeyID})
	}
	req = req.WithContext(rctx)

	rec := httptest.NewRecorder()
	s.handleChatCompletions(rec, req)
	result := rec.Result()
	defer result.Body.Close()
	respBody := rec.Body.Bytes()

	outcome := BatchOutcome{
		RequestID:    result.Header.Get("X-Inference-Job-ID"),
		ResponseBody: respBody,
	}
	if result.StatusCode/100 == 2 {
		outcome.PromptTokens, outcome.CompletionTokens = batchUsageFromResponse(respBody)
		return outcome, nil
	}
	if result.StatusCode == http.StatusTooManyRequests &&
		responseErrorCode(respBody) == batchNoCapacityCode {
		outcome.ErrCode = batchNoCapacityCode
		return outcome, nil
	}
	outcome.ErrCode = batchRequestFailedCode
	return outcome, nil
}

// batchUsageFromResponse reads the provider-reported usage back off an assembled
// non-streaming response. A body without a usage object yields zeros rather than
// an error: the attempt succeeded, and the batch item records what it was told.
func batchUsageFromResponse(body []byte) (promptTokens, completionTokens int) {
	var envelope struct {
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			InputTokens      int `json:"input_tokens"`
			OutputTokens     int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return 0, 0
	}
	promptTokens = envelope.Usage.PromptTokens
	if promptTokens == 0 {
		// Responses-API shaped usage.
		promptTokens = envelope.Usage.InputTokens
	}
	completionTokens = envelope.Usage.CompletionTokens
	if completionTokens == 0 {
		completionTokens = envelope.Usage.OutputTokens
	}
	return promptTokens, completionTokens
}

// responseErrorCode extracts error.code from an OpenAI-shaped error body, or ""
// when the body is not one.
func responseErrorCode(body []byte) string {
	var envelope struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}
	return envelope.Error.Code
}
