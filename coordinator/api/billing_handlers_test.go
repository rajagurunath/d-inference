package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/eigeninference/d-inference/coordinator/payments"
)

// TestGetPricingAdvertisesBatchLanePrices verifies GET /v1/pricing publishes
// the batch-lane rate alongside the online (list) rate for every priced model
// and for the fallback defaults, plus the batch_discount itself
// (docs/design/tidal-batch-lane.md §3.5). The advertised batch rate must be
// exactly what the settlement path charges, so the assertions derive it from
// payments.BatchPricePerMillion rather than restating 0.5 by hand.
func TestGetPricingAdvertisesBatchLanePrices(t *testing.T) {
	srv, st := testServer(t)

	const model = "pricing-batch-lane-model"
	const listIn, listOut int64 = 60_000, 240_000
	if err := st.SetModelPrice("platform", model, listIn, listOut); err != nil {
		t.Fatalf("SetModelPrice: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/pricing", nil)
	w := httptest.NewRecorder()
	srv.Handler().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body = %s", w.Code, w.Body.String())
	}

	type entryJSON struct {
		Model            string `json:"model"`
		InputPrice       int64  `json:"input_price"`
		OutputPrice      int64  `json:"output_price"`
		BatchInputPrice  int64  `json:"batch_input_price"`
		BatchOutputPrice int64  `json:"batch_output_price"`
		BatchInputUSD    string `json:"batch_input_usd"`
		BatchOutputUSD   string `json:"batch_output_usd"`
	}
	var body struct {
		Prices                   []entryJSON `json:"prices"`
		BatchDiscount            float64     `json:"batch_discount"`
		FallbackInputPrice       int64       `json:"fallback_input_price"`
		FallbackOutputPrice      int64       `json:"fallback_output_price"`
		FallbackBatchInputPrice  int64       `json:"fallback_batch_input_price"`
		FallbackBatchOutputPrice int64       `json:"fallback_batch_output_price"`
		FallbackBatchInputUSD    string      `json:"fallback_batch_input_usd"`
		FallbackBatchOutputUSD   string      `json:"fallback_batch_output_usd"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode: %v; body = %s", err, w.Body.String())
	}

	if body.BatchDiscount != payments.BatchDiscount {
		t.Errorf("batch_discount = %v, want %v", body.BatchDiscount, payments.BatchDiscount)
	}
	if body.BatchDiscount != 0.5 {
		t.Errorf("batch_discount = %v, want 0.5 (the published discount)", body.BatchDiscount)
	}

	var entry *entryJSON
	for i := range body.Prices {
		if body.Prices[i].Model == model {
			entry = &body.Prices[i]
		}
	}
	if entry == nil {
		t.Fatalf("no pricing entry for %q; body = %s", model, w.Body.String())
	}
	if entry.InputPrice != listIn || entry.OutputPrice != listOut {
		t.Errorf("list prices = (%d, %d), want (%d, %d)", entry.InputPrice, entry.OutputPrice, listIn, listOut)
	}
	if want := payments.BatchPricePerMillion(listIn); entry.BatchInputPrice != want {
		t.Errorf("batch_input_price = %d, want %d", entry.BatchInputPrice, want)
	}
	if want := payments.BatchPricePerMillion(listOut); entry.BatchOutputPrice != want {
		t.Errorf("batch_output_price = %d, want %d", entry.BatchOutputPrice, want)
	}
	if entry.BatchInputPrice != listIn/2 || entry.BatchOutputPrice != listOut/2 {
		t.Errorf("batch prices = (%d, %d), want half of list (%d, %d)",
			entry.BatchInputPrice, entry.BatchOutputPrice, listIn/2, listOut/2)
	}
	if entry.BatchInputUSD != "$0.0300" {
		t.Errorf("batch_input_usd = %q, want %q", entry.BatchInputUSD, "$0.0300")
	}
	if entry.BatchOutputUSD != "$0.1200" {
		t.Errorf("batch_output_usd = %q, want %q", entry.BatchOutputUSD, "$0.1200")
	}

	// Fallback prices (models with no DB row) carry the batch rate too.
	if want := payments.BatchPricePerMillion(payments.DefaultInputPricePerMillion); body.FallbackBatchInputPrice != want {
		t.Errorf("fallback_batch_input_price = %d, want %d", body.FallbackBatchInputPrice, want)
	}
	if want := payments.BatchPricePerMillion(payments.DefaultOutputPricePerMillion); body.FallbackBatchOutputPrice != want {
		t.Errorf("fallback_batch_output_price = %d, want %d", body.FallbackBatchOutputPrice, want)
	}
	if body.FallbackBatchInputPrice != body.FallbackInputPrice/2 {
		t.Errorf("fallback_batch_input_price = %d, want half of %d", body.FallbackBatchInputPrice, body.FallbackInputPrice)
	}
	if body.FallbackBatchOutputPrice != body.FallbackOutputPrice/2 {
		t.Errorf("fallback_batch_output_price = %d, want half of %d", body.FallbackBatchOutputPrice, body.FallbackOutputPrice)
	}
	if body.FallbackBatchInputUSD != "$0.0250" {
		t.Errorf("fallback_batch_input_usd = %q, want %q", body.FallbackBatchInputUSD, "$0.0250")
	}
	if body.FallbackBatchOutputUSD != "$0.1000" {
		t.Errorf("fallback_batch_output_usd = %q, want %q", body.FallbackBatchOutputUSD, "$0.1000")
	}
}
