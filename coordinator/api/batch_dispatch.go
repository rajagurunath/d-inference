package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	"github.com/eigeninference/d-inference/coordinator/batchlane"
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
	// ErrCode the Outcome carries, so the batch dispatcher can tell "nothing
	// free right now, re-offer me next tick" apart from a real failure without
	// parsing prose. Defined AS the dispatcher's own constant: the HTTP error
	// body and the settle vocabulary must never drift apart.
	batchNoCapacityCode = batchlane.ErrCodeNoCapacity
	// batchNoCapacityRetryAfterSec is the Retry-After the batch lane advertises
	// on that 429. It matches the dispatcher's 1 Hz tick scale: a few seconds is
	// long enough for online traffic to drain a slot and short enough that a
	// paced client (OpenRouter's synchronous service_tier=batch calls) keeps
	// making progress.
	batchNoCapacityRetryAfterSec = 5
)

// batchRequestFailedCode is the bounded error code for a batch attempt that
// reached a terminal that is not a capacity refusal.
const batchRequestFailedCode = batchlane.ErrCodeRequestFailed

// batchCancelledCode is the bounded error code for an attempt that ended
// because the CALLER'S context did, not because anything went wrong with the
// request. Handed back separately from "request_failed" because it must not
// burn one of the item's three attempts: a coordinator shutdown or an operator
// cancelling the batch mid-flight says nothing about whether the item can be
// served, and charging an attempt for it would retire a perfectly good item
// after three restarts.
const batchCancelledCode = batchlane.ErrCodeCancelled

// errBatchAPIKeyUnusable is returned when the attributing key ID does not
// resolve to a live key owned by accountID: unknown, revoked, disabled, expired
// or belonging to someone else. Typed so the dispatcher can fail the ITEM
// (permanently — the key is not coming back this tick) rather than treating it
// as a transient provider fault.
var errBatchAPIKeyUnusable = errors.New("batch dispatch: api key is not usable")

// DispatchBatchItem runs one batch request through the standard consumer
// dispatch funnel on registry.LaneBatch and waits for it to complete, returning
// the assembled non-streaming response. Its signature IS batchlane.DispatchFn,
// so cmd/coordinator wires the method value straight into the dispatcher with
// no adapter. It is the single entry point the batch
// dispatcher (coordinator/batchlane) uses; nothing about the request
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
) (batchlane.Outcome, error) {
	if s == nil {
		return batchlane.Outcome{}, errors.New("nil server")
	}
	if model == "" {
		return batchlane.Outcome{}, errors.New("batch dispatch: model is required")
	}
	parsed, err := decodeInferenceJSONObject(body)
	if err != nil {
		return batchlane.Outcome{}, fmt.Errorf("batch dispatch: %w", err)
	}
	parsed["model"] = model
	// A batch item is assembled from CompleteCh, never relayed as SSE. Deleting
	// rather than setting false keeps the forwarded body byte-minimal.
	delete(parsed, "stream")
	itemBody, err := marshalForwardBody(parsed)
	if err != nil {
		return batchlane.Outcome{}, fmt.Errorf("batch dispatch: %w", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(itemBody))
	req.Header.Set("Content-Type", "application/json")
	rctx := withRequestLane(ctx, registry.LaneBatch)
	rctx = context.WithValue(rctx, ctxKeyConsumer, accountID)
	// Load the REAL key record. A stub carrying only the ID would satisfy
	// apiKeyFromContext while reporting no AllowedModels and no LimitMicroUSD,
	// so keyModelAllowed and checkKeySpendCap (api/apikey_handlers.go) would
	// wave every batch item through — a key restricted to one model, or capped
	// at $5, would be unrestricted the moment its traffic arrived on the batch
	// lane. The key is scoped to accountID exactly as GET /v1/keys/{id} is, so a
	// batch row cannot attribute itself to another account's key.
	//
	// apiKeyID == "" is a legitimate state, not a missing value: the batch was
	// created by a caller that has no API key (a Privy JWT session or the admin
	// key). Nothing is stamped on the context, so keyModelAllowed and
	// checkKeySpendCap see no key and the item runs under ACCOUNT-level limits
	// only — exactly what that caller's ONLINE requests do. The decision is
	// logged once per batch at creation (handleBatchCreate) rather than per
	// item here, because the key id is fixed when the batch row is written and
	// DispatchFn carries no batch id to deduplicate on.
	if apiKeyID != "" {
		keyRec, err := s.store.GetAPIKeyByID(accountID, apiKeyID)
		if err != nil || keyRec == nil {
			return batchlane.Outcome{ErrCode: batchRequestFailedCode},
				fmt.Errorf("%w: %q: %v", errBatchAPIKeyUnusable, apiKeyID, err)
		}
		if keyRec.Disabled {
			return batchlane.Outcome{ErrCode: batchRequestFailedCode},
				fmt.Errorf("%w: %q is revoked", errBatchAPIKeyUnusable, apiKeyID)
		}
		if keyRec.ExpiresAt != nil && time.Now().After(*keyRec.ExpiresAt) {
			return batchlane.Outcome{ErrCode: batchRequestFailedCode},
				fmt.Errorf("%w: %q is expired", errBatchAPIKeyUnusable, apiKeyID)
		}
		rctx = context.WithValue(rctx, ctxKeyAPIKey, keyRec)
	}
	req = req.WithContext(rctx)

	rec := httptest.NewRecorder()
	// The per-key RPM override lives in the rate-limit MIDDLEWARE
	// (rateLimitWithTier -> applyKeyRPMLimit), which a batch item never passes
	// through: DispatchBatchItem calls the handler directly. Without this the
	// per-key requests-per-minute cap was silently exempt on the batch lane —
	// a key throttled to 10 RPM online could have the dispatcher run it at the
	// fleet's whole batch allowance. Applying the same function against the
	// same synthetic request keeps ONE implementation of the limit.
	//
	// A throttled item comes back as no_capacity, not request_failed: it never
	// reached a provider and nothing about it was proven, so the dispatcher
	// must release the claim WITHOUT charging one of its three attempts and
	// re-offer it on a later tick, which is exactly what the key's own rate
	// limit is asking for.
	if !s.applyKeyRPMLimit(rec, req) {
		return batchlane.Outcome{
			ResponseBody: rec.Body.Bytes(),
			ErrCode:      batchNoCapacityCode,
		}, nil
	}
	s.handleChatCompletions(rec, req)
	result := rec.Result()
	defer result.Body.Close()
	respBody := rec.Body.Bytes()

	outcome := batchlane.Outcome{
		RequestID:    result.Header.Get("X-Inference-Job-ID"),
		ResponseBody: respBody,
	}
	// A served item is a 2xx WITH a body. The dispatch funnel can also return
	// having written nothing at all — its outcomeClientGone exits (the caller's
	// context fired while parked for a routing-scan slot, or while queued) refund
	// the reservation and, by contract, produce no response body — and an
	// httptest.ResponseRecorder reports that as a default 200 with an empty body.
	// Accepting it here would settle a CANCELLED item as a completed one with
	// zero tokens, so an empty response falls through to the cancellation check
	// below (and, if the context is fine, to request_failed).
	if result.StatusCode/100 == 2 && len(respBody) > 0 {
		outcome.PromptTokens, outcome.CompletionTokens = batchUsageFromResponse(respBody)
		return outcome, nil
	}
	if result.StatusCode == http.StatusTooManyRequests &&
		responseErrorCode(respBody) == batchNoCapacityCode {
		outcome.ErrCode = batchNoCapacityCode
		return outcome, nil
	}
	// The caller's own context ended (coordinator shutdown, the batch cancelled
	// under us, the dispatcher's per-item budget expiring). Every terminal the
	// handler can write in that situation — a client-gone abort, a first-content
	// deadline, a plain 500 — is indistinguishable from a provider failure by
	// status code alone, so ask the context directly. Checked AFTER the success
	// and no-capacity branches so a request that finished before the cancellation
	// landed still reports what it actually achieved.
	if err := ctx.Err(); err != nil {
		outcome.ErrCode = batchCancelledCode
		return outcome, fmt.Errorf("batch dispatch: %w", err)
	}
	outcome.ErrCode = batchRequestFailedCode
	return outcome, nil
}

// batchDiscardedRefundPrefix is the ledger reference prefix a discarded batch
// item's refund is written under: "batch_discarded:<request id>". Distinct from
// every other refund reference the funnel writes (reservation_refund,
// provider_error, the settlement difference keyed on the bare request id) so a
// ledger reader can tell the coordinator giving money back for a result it
// threw away from the ordinary refunds a served request produces.
const batchDiscardedRefundPrefix = "batch_discarded:"

// RefundBatchItem returns the money for a batch item that reached a provider
// and was charged for, but whose result the dispatcher then discarded — the
// batch went terminal under it, or the sweep had already closed the item.
// batchlane.Config.RefundItem is wired to this in cmd/coordinator; its
// signature IS that hook, so there is no adapter.
//
// The amount is the cost the settlement actually debited for this request, read
// back from the ledger's usage history rather than recomputed: the charge
// resolves a provider's custom price, the platform price, the account's
// wholesale role and the batch lane's 0.5 multiplier, and a second
// implementation of that formula here would be one more thing to keep in
// agreement. A refund is credited with the SAME primitive every other refund on
// this path uses (store.Credit with store.LedgerRefund — see
// refundReservedBalance and releaseInitialReservation), so it nets against the
// charge exactly.
//
// What is deliberately NOT reversed is the provider's earning. The provider
// served the request: it loaded the prompt, generated the tokens and reported
// them. That the coordinator then dropped the answer — because the consumer
// cancelled, or the item's batch expired mid-flight — is the coordinator's
// doing, not the provider's, so the platform absorbs the difference rather than
// clawing back a payout for work that was genuinely done.
//
// Errors are returned, never swallowed: the dispatcher logs them against the
// batch and item ids, which is the only place both are known.
func (s *Server) RefundBatchItem(accountID, requestID string, promptTokens, completionTokens int) error {
	if s == nil {
		return errors.New("nil server")
	}
	if accountID == "" || requestID == "" {
		return fmt.Errorf("batch refund: account id and request id are required")
	}
	charged, ok := s.settledChargeMicroUSD(accountID, requestID)
	if !ok {
		// The settlement never recorded a usage entry for this request, so
		// there is no charge to reverse and no amount that can be derived
		// without re-deriving the price. Reported rather than silently skipped:
		// it means either that the attempt was never billed (a zero-token
		// completion, an owned-machine self-route, an uncollected charge) or
		// that the entry aged out of the history, and only the second is a
		// consumer left out of pocket.
		return fmt.Errorf("batch refund: no settled charge recorded for request %q", requestID)
	}
	if charged <= 0 {
		return nil // billed nothing, so there is nothing to give back
	}
	if err := s.store.Credit(accountID, charged, store.LedgerRefund, batchDiscardedRefundPrefix+requestID); err != nil {
		s.ddIncr("billing.credit_failed", []string{"op:batch_discarded_refund"})
		return fmt.Errorf("batch refund: credit %d micro-USD for request %q: %w", charged, requestID, err)
	}
	s.ddIncr("billing.batch_discarded_refunds", nil)
	s.ddHistogram("billing.batch_discarded_refund_micro_usd", float64(charged), nil)
	s.logger.Info("batch lane: refunded a discarded result",
		"request_id", requestID, "refund_micro_usd", charged,
		"prompt_tokens", promptTokens, "completion_tokens", completionTokens)
	return nil
}

// settledChargeMicroUSD returns what the settlement charged accountID for
// requestID, and whether an entry for it was found.
//
// The ledger's in-memory usage history is the source because it is the only
// per-request record of the FINAL cost that is written synchronously, before
// the completion is handed back to the caller (handleComplete records usage
// ahead of closing CompleteCh), so it is already there by the time the
// dispatcher settles the outcome one tick later. The persisted usage row is
// written asynchronously and the ledger's own history is keyed on the
// reservation's reference, not the request id, so neither answers this
// question. The history is bounded per account (payments.usageHistoryLimit), so
// a request that has since been pushed out reports not-found rather than a
// wrong number.
func (s *Server) settledChargeMicroUSD(accountID, requestID string) (int64, bool) {
	if s.ledger == nil {
		return 0, false
	}
	entries := s.ledger.Usage(accountID)
	for i := len(entries) - 1; i >= 0; i-- {
		if entries[i].JobID == requestID {
			return entries[i].CostMicroUSD, true
		}
	}
	return 0, false
}

// laneModelAllowed is the per-key model allow-list check the request prelude
// runs, plus the one equivalence the batch lane needs.
//
// An ONLINE request is checked on the name the consumer typed, which is also
// the name they put on the key. A batch item cannot be: by the time it reaches
// the funnel, DispatchBatchItem has already replaced the body's model with the
// RESOLVED build id (deliberately — re-resolving an alias hours after
// submission would silently reroute a half-finished batch onto a different
// build). Comparing that build id against an alias-scoped allow-list would deny
// every alias-scoped key its own batches.
//
// So on the coordinator-stamped batch lane — laneFromContext, which only
// DispatchBatchItem sets, never a consumer's own service_tier — an allow-list
// entry also matches the build id it resolves to. This never WIDENS a key: the
// requested name was already checked against the same allow-list at
// handleBatchCreate, so a key holding only build ids still cannot run a batch
// that asks for an alias. It only stops the coordinator's own internal rename
// from denying work the key was granted.
func (s *Server) laneModelAllowed(ctx context.Context, model string) bool {
	if s.keyModelAllowed(ctx, model) {
		return true
	}
	if laneFromContext(ctx) != registry.LaneBatch {
		return false
	}
	k := apiKeyFromContext(ctx)
	if k == nil {
		// Unreachable: keyModelAllowed already allows a keyless request.
		return false
	}
	resolve := s.batchModelResolver()
	for _, allowed := range k.AllowedModels {
		if build, ok := resolve(allowed); ok && build == model {
			return true
		}
	}
	return false
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
