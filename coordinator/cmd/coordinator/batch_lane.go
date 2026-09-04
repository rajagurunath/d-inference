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

	"github.com/eigeninference/d-inference/coordinator/api"
	"github.com/eigeninference/d-inference/coordinator/batchlane"
	"github.com/eigeninference/d-inference/coordinator/env"
	"github.com/eigeninference/d-inference/coordinator/registry"
	"github.com/eigeninference/d-inference/coordinator/saferun"
	"github.com/eigeninference/d-inference/coordinator/store"
	"github.com/eigeninference/d-inference/coordinator/store/sealedblob"
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

	blobs := batchBlobStore(srv)
	if blobs == nil {
		// No key, so no sealed-at-rest storage, so no batch lane: prompts must
		// never sit on coordinator disk in the clear. api.NewBatchBlobStore
		// (PR2) has already logged why.
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
		},
		logger,
	)
	saferun.Go(logger, "batch_dispatcher", func() { d.Run(ctx) })
}

// batchBlobStore reads the sealed batch blob store off the server.
//
// WIRING POINT (PR2). The accessor is (*api.Server).BatchBlobs, which lands
// with the Batch API on tidal/pr2-batch-api. The assertion below is what lets
// this file compile and behave correctly on BOTH sides of that merge: before
// PR2 the method does not exist, the assertion fails, and the dispatcher stays
// down (there would be nothing to dispatch — no route can create a batch yet);
// after PR2 it resolves and the lane comes up with no further change. Replace
// the assertion with a direct `srv.BatchBlobs()` once PR2 has merged.
func batchBlobStore(srv *api.Server) *sealedblob.Store {
	if srv == nil {
		return nil
	}
	if p, ok := any(srv).(interface{ BatchBlobs() *sealedblob.Store }); ok {
		return p.BatchBlobs()
	}
	return nil
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

// batchFinalizeFn returns the hook that assembles a finished batch's output and
// error files.
//
// WIRING POINT (PR2). The assembler is api/batch_assembler.go, which does not
// exist on this branch; until it does the hook is nil (a no-op), which leaves
// item rows correctly settled and the output file unwritten. Everything else in
// the loop — claim, dispatch, settle, expire, cancel — is unaffected.
func batchFinalizeFn(_ *api.Server) batchlane.FinalizeFn {
	return nil
}
