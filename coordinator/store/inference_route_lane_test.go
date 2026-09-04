package store

import (
	"testing"
	"time"
)

// assertInferenceRouteLaneRoundTrips drives the shared scenario against any
// Store impl: a batch-lane route's Lane is stamped at settlement (via
// UpdateInferenceRouteOutcome, mirroring CostMicroUSD) and must come back on
// InferenceRouteRecordsSince. An online-lane row (Lane never set) must come
// back as "" — the zero value, matching registry.LaneOnline.
func assertInferenceRouteLaneRoundTrips(t *testing.T, s Store) {
	t.Helper()
	const batchReqID = "req-lane-batch"
	const onlineReqID = "req-lane-online"

	if err := s.RecordInferenceRoute(&InferenceRouteRecord{
		RequestID: batchReqID,
		Attempt:   0,
		Model:     "gpt-oss-20b",
		Outcome:   "selected",
		CreatedAt: time.Now(),
	}); err != nil {
		t.Fatalf("RecordInferenceRoute(batch): %v", err)
	}
	if err := s.UpdateInferenceRouteOutcome(batchReqID, 0, &InferenceRouteOutcome{
		FinalStatus:         "success",
		CompletionTokens:    10,
		CompletionTokensSet: true,
		CostMicroUSD:        1,
		Lane:                "batch",
	}); err != nil {
		t.Fatalf("UpdateInferenceRouteOutcome(batch): %v", err)
	}

	if err := s.RecordInferenceRoute(&InferenceRouteRecord{
		RequestID: onlineReqID,
		Attempt:   0,
		Model:     "gpt-oss-20b",
		Outcome:   "selected",
		CreatedAt: time.Now(),
	}); err != nil {
		t.Fatalf("RecordInferenceRoute(online): %v", err)
	}
	if err := s.UpdateInferenceRouteOutcome(onlineReqID, 0, &InferenceRouteOutcome{
		FinalStatus:         "success",
		CompletionTokens:    10,
		CompletionTokensSet: true,
		CostMicroUSD:        100,
	}); err != nil {
		t.Fatalf("UpdateInferenceRouteOutcome(online): %v", err)
	}

	all := s.InferenceRouteRecordsSince(time.Time{})
	var gotBatch, gotOnline *InferenceRouteRecord
	for i := range all {
		switch all[i].RequestID {
		case batchReqID:
			gotBatch = &all[i]
		case onlineReqID:
			gotOnline = &all[i]
		}
	}
	if gotBatch == nil {
		t.Fatalf("batch route row %q not found", batchReqID)
	}
	if gotBatch.Lane != "batch" {
		t.Fatalf("batch route lane = %q, want %q", gotBatch.Lane, "batch")
	}
	if gotOnline == nil {
		t.Fatalf("online route row %q not found", onlineReqID)
	}
	if gotOnline.Lane != "" {
		t.Fatalf("online route lane = %q, want \"\" (registry.LaneOnline)", gotOnline.Lane)
	}
}

func TestMemoryInferenceRouteLaneRoundTrips(t *testing.T) {
	assertInferenceRouteLaneRoundTrips(t, NewMemory(Config{}))
}

func TestPostgresInferenceRouteLaneRoundTrips(t *testing.T) {
	assertInferenceRouteLaneRoundTrips(t, testPostgresStore(t))
}

// assertProviderEarningLaneRoundTrips drives the shared scenario against any
// Store impl: a batch-lane earning credited through CreditProviderAccount (the
// settlement path) round-trips its Lane through both GetProviderEarnings and
// GetAccountEarnings; an earning recorded through RecordProviderEarning with
// no Lane set comes back as "".
func assertProviderEarningLaneRoundTrips(t *testing.T, s Store) {
	t.Helper()
	const batchJobID = "job-lane-batch"
	const onlineJobID = "job-lane-online"
	const providerKey = "pk-lane-earning"
	const accountID = "acct-lane-earning"

	if err := s.CreditProviderAccount(&ProviderEarning{
		AccountID:        accountID,
		ProviderID:       "prov-1",
		ProviderKey:      providerKey,
		JobID:            batchJobID,
		Model:            "gpt-oss-20b",
		AmountMicroUSD:   1,
		PromptTokens:     0,
		CompletionTokens: 10,
		CreatedAt:        time.Now(),
		Lane:             "batch",
	}); err != nil {
		t.Fatalf("CreditProviderAccount(batch): %v", err)
	}
	if err := s.RecordProviderEarning(&ProviderEarning{
		AccountID:        accountID,
		ProviderID:       "prov-1",
		ProviderKey:      providerKey,
		JobID:            onlineJobID,
		Model:            "gpt-oss-20b",
		AmountMicroUSD:   100,
		PromptTokens:     0,
		CompletionTokens: 10,
		CreatedAt:        time.Now(),
	}); err != nil {
		t.Fatalf("RecordProviderEarning(online): %v", err)
	}

	byProvider, err := s.GetProviderEarnings(providerKey, 10)
	if err != nil {
		t.Fatalf("GetProviderEarnings: %v", err)
	}
	var gotBatch, gotOnline *ProviderEarning
	for i := range byProvider {
		switch byProvider[i].JobID {
		case batchJobID:
			gotBatch = &byProvider[i]
		case onlineJobID:
			gotOnline = &byProvider[i]
		}
	}
	if gotBatch == nil {
		t.Fatalf("batch earning %q not found via GetProviderEarnings", batchJobID)
	}
	if gotBatch.Lane != "batch" {
		t.Fatalf("batch earning lane = %q, want %q", gotBatch.Lane, "batch")
	}
	if gotOnline == nil {
		t.Fatalf("online earning %q not found via GetProviderEarnings", onlineJobID)
	}
	if gotOnline.Lane != "" {
		t.Fatalf("online earning lane = %q, want \"\"", gotOnline.Lane)
	}

	byAccount, err := s.GetAccountEarnings(accountID, 10)
	if err != nil {
		t.Fatalf("GetAccountEarnings: %v", err)
	}
	found := false
	for _, e := range byAccount {
		if e.JobID == batchJobID {
			found = true
			if e.Lane != "batch" {
				t.Fatalf("GetAccountEarnings batch lane = %q, want %q", e.Lane, "batch")
			}
		}
	}
	if !found {
		t.Fatalf("batch earning %q not found via GetAccountEarnings", batchJobID)
	}
}

func TestMemoryProviderEarningLaneRoundTrips(t *testing.T) {
	assertProviderEarningLaneRoundTrips(t, NewMemory(Config{}))
}

func TestPostgresProviderEarningLaneRoundTrips(t *testing.T) {
	assertProviderEarningLaneRoundTrips(t, testPostgresStore(t))
}
