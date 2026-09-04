package main

// batch_lane.go starts the Tidal batch dispatcher
// (docs/design/tidal-batch-lane.md §3.4): the 1 Hz loop that fills provider
// slots the online quality cap is leaving empty with 24-hour batch work.
//
// It lives in main because it is the only place that may import both api and
// batchlane: batchlane must never import api (api imports batchlane), so the
// adapter from api.BatchOutcome to batchlane.Outcome belongs here.

import (
	"context"
	"log/slog"
	"time"

	"github.com/eigeninference/d-inference/coordinator/api"
	"github.com/eigeninference/d-inference/coordinator/batchlane"
	"github.com/eigeninference/d-inference/coordinator/env"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/saferun"
	"github.com/eigeninference/d-inference/coordinator/store"
)

// batchLaneEnabledEnv is the operator kill switch for the batch dispatcher.
// Default true: the lane is already gated on a sealed blob store existing, so
// a coordinator with no batch key never starts it regardless.
const batchLaneEnabledEnv = env.EnvPrefix + "_BATCH_LANE_ENABLED"

// startBatchDispatcher builds the dispatcher and runs it under saferun, or
// explains why it did not. Safe to call before any batch has ever been created.
func startBatchDispatcher(
	ctx context.Context,
	logger *slog.Logger,
	srv *api.Server,
	reg *registry.Registry,
	st store.Store,
) {
	if !env.EnvBool(batchLaneEnabledEnv, true) {
		logger.Info("batch lane dispatcher disabled by configuration", "env", batchLaneEnabledEnv)
		return
	}

	blobs := srv.BatchBlobs()
	if blobs == nil {
		// No key, so no sealed-at-rest storage, so no batch lane: prompts must
		// never sit on coordinator disk in the clear. api.NewBatchBlobStore has
		// already logged why, and every batch route answers 503.
		logger.Info("batch lane dispatcher not started — no sealed batch blob store is configured")
		return
	}

	d := batchlane.New(
		st,
		blobs,
		batchlane.NewRegistryView(reg),
		batchDispatchFn(srv),
		batchFinalizeFn(srv),
		batchlane.Config{
			Tick:        batchlane.DefaultTick,
			MaxAttempts: batchlane.DefaultMaxAttempts,
			// One retention boundary for the assembled files and for the
			// per-item result blobs behind them.
			OutputRetention: api.BatchOutputRetention,
			Purge:           srv.PurgeExpiredBatchFiles,
		},
		logger,
	)
	saferun.Go(logger, "batch_dispatcher", func() { d.Run(ctx) })
}

// batchDispatchFn adapts the api layer's batch dispatch entry to the
// dispatcher's funnel type. The two Outcome structs are field-for-field
// identical; they are separate types only to keep the import direction one-way.
func batchDispatchFn(srv *api.Server) batchlane.DispatchFn {
	return func(ctx context.Context, accountID, apiKeyID, model string, body []byte) (batchlane.Outcome, error) {
		out, err := srv.DispatchBatchItem(ctx, accountID, apiKeyID, model, body)
		return batchlane.Outcome{
			RequestID:        out.RequestID,
			PromptTokens:     out.PromptTokens,
			CompletionTokens: out.CompletionTokens,
			ResponseBody:     out.ResponseBody,
			ErrCode:          out.ErrCode,
		}, err
	}
}

// batchFinalizeFn adapts the assembler to the dispatcher's finalize hook. The
// dispatcher calls it after every settle and from its sweep; FinalizeBatchIfDone
// is a no-op unless the batch is open and has no pending or in-flight items, so
// the common tick costs one count query.
func batchFinalizeFn(srv *api.Server) batchlane.FinalizeFn {
	return func(batchID string, now time.Time) error {
		_, err := srv.FinalizeBatchIfDone(batchID, now)
		return err
	}
}
