package api

import (
	"bytes"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/eigeninference/d-inference/coordinator/protocol"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batchRetryDispatchState builds the dispatchState the dispatch ladder holds on
// `attempt`, on a fleet whose only provider is warm for the model but closed to
// the batch lane (a waiting row). The reservation refund is a counter so the
// test can see the batch terminal do its own cleanup.
func batchRetryDispatchState(t *testing.T, attempt int, lane registry.Lane) (
	*dispatchState, *httptest.ResponseRecorder, *int,
) {
	t.Helper()
	const model = "batch-retry-model"
	logger := slog.New(slog.NewTextHandler(&bytes.Buffer{}, &slog.HandlerOptions{Level: slog.LevelError}))
	reg := registry.New(logger)
	srv := NewServer(reg, store.NewMemory(store.Config{AdminKey: "test-key"}), ServerConfig{}, logger)

	p := makeRoutableProvider(t, reg, "p-batch-retry", model)
	p.Mu().Lock()
	p.BackendCapacity = &protocol.BackendCapacity{
		TotalMemoryGB: 64,
		Slots: []protocol.BackendSlotCapacity{
			{Model: model, State: "running", NumRunning: 0, NumWaiting: 1},
		},
	}
	p.Mu().Unlock()

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		strings.NewReader(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`))
	refunds := 0
	d := &dispatchState{
		s: srv, w: w, r: r,
		model: model, publicModel: model,
		rawBody:               []byte(`{"model":"` + model + `","messages":[{"role":"user","content":"hi"}]}`),
		consumerKey:           "acct-batch",
		estimatedPromptTokens: 8,
		requestedMaxTokens:    64,
		deadline:              30 * time.Second,
		timing:                &registry.RequestTiming{ReceivedAt: time.Now()},
		lane:                  lane,
		attempt:               attempt,
		refundReservation:     func() { refunds++ },
		excludeProviders:      map[string]struct{}{},
	}
	// The batch lane finds nothing here because the only slot carries a waiting
	// row; the ONLINE lane is still served by that same slot, so an online
	// control has to reach "no provider available" the only other way the ladder
	// can — the attempt-0 provider already being excluded.
	if lane != registry.LaneBatch {
		d.excludeProviders[p.ID] = struct{}{}
	}
	// What attempt 0's outcomeRetry leaves latched: a real provider error from a
	// provider that has since been excluded.
	if attempt > 0 {
		d.setLastError("provider returned an error", http.StatusBadGateway)
	}
	return d, w, &refunds
}

// TestBatchNoCapacityOnRetryIsNotAFailure is the B3 regression. The batch lane's
// "no headroom, come back next tick" terminal used to sit BELOW the retry
// fail-fast, so a batch item that found a provider on attempt 0 and none on
// attempt 1 came back with attempt 0's latched provider error. The dispatcher
// reads that as "request_failed" and CHARGES one of the item's three attempts
// for a capacity refusal that proved nothing — three unlucky ticks retire a
// perfectly good item. Both attempts must answer 429 / no_capacity.
func TestBatchNoCapacityOnRetryIsNotAFailure(t *testing.T) {
	for _, attempt := range []int{0, 1} {
		t.Run(map[bool]string{true: "first attempt", false: "retry attempt"}[attempt == 0], func(t *testing.T) {
			d, w, refunds := batchRetryDispatchState(t, attempt, registry.LaneBatch)

			if got := d.dispatchPrimary(); got != outcomeResponseWritten {
				t.Fatalf("outcome=%v, want outcomeResponseWritten (a written 429), lastErr=%q/%d",
					got, d.lastErr, d.lastErrCode)
			}
			resp := w.Result()
			defer resp.Body.Close()
			if resp.StatusCode != http.StatusTooManyRequests {
				t.Fatalf("status=%d body=%s, want 429", resp.StatusCode, w.Body.Bytes())
			}
			if code := responseErrorCode(w.Body.Bytes()); code != batchNoCapacityCode {
				t.Fatalf("error.code=%q body=%s, want %q", code, w.Body.Bytes(), batchNoCapacityCode)
			}
			if got := resp.Header.Get("Retry-After"); got != "5" {
				t.Fatalf("Retry-After=%q, want %q", got, "5")
			}
			if *refunds != 1 {
				t.Fatalf("reservation refunds=%d, want 1", *refunds)
			}
		})
	}
}

// TestOnlineRetryStillFailsFastWithoutHeadroom is the control: the retry
// fail-fast the batch terminal now sits above is untouched for online traffic,
// which still stops on attempt 0's latched error rather than being handed the
// batch lane's 429.
func TestOnlineRetryStillFailsFastWithoutHeadroom(t *testing.T) {
	d, w, refunds := batchRetryDispatchState(t, 1, registry.LaneOnline)

	if got := d.dispatchPrimary(); got != outcomeFailFast {
		t.Fatalf("outcome=%v, want outcomeFailFast", got)
	}
	if w.Body.Len() != 0 {
		t.Fatalf("the online retry wrote a response body: %s", w.Body.Bytes())
	}
	if d.lastErrCode != http.StatusBadGateway {
		t.Fatalf("lastErrCode=%d, want the latched 502 from attempt 0", d.lastErrCode)
	}
	if *refunds != 0 {
		t.Fatalf("the online fail-fast refunded the reservation %d times, want 0", *refunds)
	}
}
