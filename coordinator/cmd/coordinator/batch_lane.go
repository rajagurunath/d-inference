package main

// batch_lane.go starts the Tidal batch dispatcher
// (docs/design/tidal-batch-lane.md §3.4): the 1 Hz loop that fills provider
// slots the online quality cap is leaving empty with 24-hour batch work.
//
// It lives in main because it is the only place that holds both the api server
// and the dispatcher. The import direction is one-way — api imports batchlane,
// never the reverse — so api speaks the dispatcher's own vocabulary
// (batchlane.Outcome, batchlane.ResultBlobRef, batchlane.DefaultOutputRetention)
// and this file wires the two together with no adapter of its own.

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
		// DispatchBatchItem's signature IS batchlane.DispatchFn — no adapter.
		srv.DispatchBatchItem,
		batchFinalizeFn(srv),
		batchlane.Config{
			Tick:        batchlane.DefaultTick,
			MaxAttempts: batchlane.DefaultMaxAttempts,
			// One retention boundary for the assembled files and for the
			// per-item result blobs behind them. The assembler reads the same
			// constant, so there is one number, not two that must agree.
			OutputRetention: batchlane.DefaultOutputRetention,
			Purge:           srv.PurgeExpiredBatchFiles,
			// A successful item whose result the dispatcher then discards —
			// its batch went terminal mid-flight, or the sweep had already
			// closed it — was charged by the funnel before the dispatcher
			// could know that. RefundBatchItem gives that money back; without
			// it the consumer pays for tokens they can never read.
			RefundItem: srv.RefundBatchItem,
		},
		logger,
	)
	saferun.Go(logger, "batch_dispatcher", func() { d.Run(ctx) })
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
